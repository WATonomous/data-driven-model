#!/usr/bin/env python3

import numpy as np
import casadi as ca
import rospy
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path

class CarMPC:
    def __init__(self):
        # MPC parameters
        self.dt = 0.1  # Time step [s]
        self.N = 10     # Prediction horizon
        
        # Constraints
        self.v_max = 2.0   # Maximum velocity [m/s]
        self.a_max = 1.0   # Maximum acceleration [m/s^2]
        
        # Weights for the cost function
        self.Q = np.diag([10.0, 10.0])  # Position weights [x, y]
        self.R = np.diag([0.1, 0.1])    # Velocity weights [vx, vy]
        
        # Initialize ROS node
        rospy.init_node('car_mpc_controller')
        
        # Publishers and subscribers
        self.cmd_pub = rospy.Publisher('/car/cmd_vel', Twist, queue_size=1)
        self.state_sub = rospy.Subscriber('/car/odom', Odometry, self.state_callback)
        self.ref_sub = rospy.Subscriber('/car/reference_path', Path, self.reference_callback)
        self.debug_pub = rospy.Publisher('/car/mpc_debug', Float64MultiArray, queue_size=1)
        
        # Initialize state and reference
        self.current_state = np.zeros(2)  # [x, y]
        self.reference = np.zeros((self.N, 2))  # N points of [x, y]
        
        self.setup_mpc()
        
        rospy.loginfo("Point mass MPC initialized")

    def setup_mpc(self):
        self.opti = ca.Opti()
        
        # State variables (position)
        self.X = self.opti.variable(2, self.N + 1)  # [x, y]
        # Control variables (velocity)
        self.U = self.opti.variable(2, self.N)       # [vx, vy]
        
        # Reference trajectory parameter
        self.P = self.opti.parameter(2, self.N)     # Reference [x, y]
        
        # Define the objective function
        obj = 0
        for k in range(self.N):
            state_error = self.X[:, k] - self.P[:, k]
            obj += ca.mtimes([state_error.T, self.Q, state_error]) + \
                  ca.mtimes([self.U[:, k].T, self.R, self.U[:, k]])
        
        self.opti.minimize(obj)
        
        # System dynamics (simple integrator)
        for k in range(self.N):
            x_next = self.X[0, k] + self.U[0, k] * self.dt
            y_next = self.X[1, k] + self.U[1, k] * self.dt
            self.opti.subject_to(self.X[:, k+1] == ca.vertcat(x_next, y_next))
        
        # Input constraints
        self.opti.subject_to(self.opti.bounded(-self.v_max, self.U[0, :], self.v_max))
        self.opti.subject_to(self.opti.bounded(-self.v_max, self.U[1, :], self.v_max))
        
        # Set solver options
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.opti.solver('ipopt', opts)

    def state_callback(self, msg):
        # Update current position
        self.current_state[0] = msg.pose.pose.position.x
        self.current_state[1] = msg.pose.pose.position.y
        self.solve_mpc()

    def reference_callback(self, msg):
        """Handle reference path updates."""
        try:
            ref_points = []
            for i in range(min(self.N, len(msg.poses))):
                pose = msg.poses[i].pose
                ref_points.append([pose.position.x, pose.position.y])
            
            # Repeat last point if needed
            while len(ref_points) < self.N:
                ref_points.append(ref_points[-1])
            
            self.reference = np.array(ref_points)
            
        except Exception as e:
            rospy.logerr("Failed to process reference path: %s", str(e))

    def solve_mpc(self):
        """Solve the MPC optimization problem."""
        # Set initial state constraint
        self.opti.set_value(self.P, self.reference.T)
        self.opti.subject_to(self.X[:, 0] == self.current_state)
        
        try:
            sol = self.opti.solve()
            u_optimal = sol.value(self.U)
            
            # Publish first control command
            self.publish_control(u_optimal[:, 0])
            
            # Publish debug information
            self.publish_debug_info(u_optimal[:, 0])
            
        except Exception as e:
            rospy.logwarn("MPC optimization failed: %s", str(e))
            self.publish_debug_info()

    def publish_control(self, u):
        cmd = Twist()
        cmd.linear.x = u[0]  # vx
        cmd.linear.y = u[1]  # vy
        self.cmd_pub.publish(cmd)

    def publish_debug_info(self, u=None):
        debug_msg = Float64MultiArray()
        if u is not None:
            debug_msg.data = [self.current_state[0], self.current_state[1], u[0], u[1]]
        else:
            debug_msg.data = [self.current_state[0], self.current_state[1], 0.0, 0.0]
        self.debug_pub.publish(debug_msg)

if __name__ == '__main__':
    try:
        mpc = CarMPC()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 