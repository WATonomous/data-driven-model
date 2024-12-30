#!/usr/bin/env python3

import numpy as np
import casadi as ca
import rospy
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sklearn.linear_model import Ridge
from collections import deque

class CarMPC:
    def __init__(self):
        # MPC parameters
        self.dt = 0.1  # Time step [s]
        self.N = 10    # Prediction horizon
        self.L = 2.7   # Wheelbase [m]
        
        # State constraints
        self.v_max = 20.0  # Maximum velocity [m/s]
        self.delta_max = np.pi/4  # Maximum steering angle [rad]
        self.a_max = 3.0  # Maximum acceleration [m/s^2]
        
        # Weights for the cost function
        self.Q = np.diag([10.0, 10.0, 1.0])  # State weights [x, y, yaw]
        self.R = np.diag([1.0, 1.0])         # Input weights [acceleration, steering]
        
        # Learning components
        self.buffer_size = 1000
        self.state_buffer = deque(maxlen=self.buffer_size)
        self.action_buffer = deque(maxlen=self.buffer_size)
        self.next_state_buffer = deque(maxlen=self.buffer_size)
        self.dynamics_model = Ridge(alpha=1.0)
        self.model_trained = False
        
        # Initialize ROS node
        rospy.init_node('car_mpc_controller')
        
        # Publishers and subscribers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.state_sub = rospy.Subscriber('/odom', Odometry, self.state_callback)
        self.ref_sub = rospy.Subscriber('/reference', Float64MultiArray, self.reference_callback)
        
        # Initialize state and reference
        self.current_state = np.zeros(4)  # [x, y, yaw, v]
        self.reference = np.zeros((self.N, 3))  # N points of [x, y, yaw]
        self.last_state = None
        self.last_control = None
        
        self.setup_mpc()
        
    def setup_mpc(self):
        # CasADi optimization variables
        self.opti = ca.Opti()
        
        # State variables
        self.X = self.opti.variable(4, self.N + 1)  # [x, y, yaw, v]
        self.U = self.opti.variable(2, self.N)      # [acceleration, steering_angle]
        
        # Parameters for reference trajectory and dynamics compensation
        self.P = self.opti.parameter(3, self.N)     # Reference [x, y, yaw]
        self.D = self.opti.parameter(4)             # Dynamics compensation
        
        # Define the objective function
        obj = 0
        for k in range(self.N):
            state_error = self.X[:3, k] - self.P[:, k]
            obj += ca.mtimes([state_error.T, self.Q, state_error]) + \
                  ca.mtimes([self.U[:, k].T, self.R, self.U[:, k]])
        
        self.opti.minimize(obj)
        
        # System dynamics constraints with learned compensation
        for k in range(self.N):
            # Nominal bicycle model
            x_next = self.X[0, k] + self.X[3, k] * ca.cos(self.X[2, k]) * self.dt
            y_next = self.X[1, k] + self.X[3, k] * ca.sin(self.X[2, k]) * self.dt
            yaw_next = self.X[2, k] + self.X[3, k] * ca.tan(self.U[1, k]) / self.L * self.dt
            v_next = self.X[3, k] + self.U[0, k] * self.dt
            
            # Add learned dynamics compensation (when k=0)
            if k == 0:
                x_next += self.D[0] * self.dt
                y_next += self.D[1] * self.dt
                yaw_next += self.D[2] * self.dt
                v_next += self.D[3] * self.dt
            
            self.opti.subject_to(self.X[:, k+1] == ca.vertcat(x_next, y_next, yaw_next, v_next))
        
        # Input constraints
        self.opti.subject_to(self.opti.bounded(-self.a_max, self.U[0, :], self.a_max))
        self.opti.subject_to(self.opti.bounded(-self.delta_max, self.U[1, :], self.delta_max))
        
        # Velocity constraints
        self.opti.subject_to(self.opti.bounded(0, self.X[3, :], self.v_max))
        
        # Set solver options
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.opti.solver('ipopt', opts)
    
    def update_dynamics_model(self):
        if len(self.state_buffer) < 100:  # Need minimum data to train
            return
        
        # Prepare training data
        X = np.array([(s, a[0], a[1]) for s, a in zip(self.state_buffer, self.action_buffer)])
        y = np.array(self.next_state_buffer) - np.array(self.state_buffer)
        
        # Train the model
        self.dynamics_model.fit(X, y)
        self.model_trained = True
        rospy.loginfo("Dynamics model updated")
    
    def predict_dynamics_error(self, state, action):
        if not self.model_trained:
            return np.zeros(4)
        
        X = np.array([[state[0], state[1], state[2], state[3], action[0], action[1]]])
        return self.dynamics_model.predict(X)[0]
    
    def state_callback(self, msg):
        # Extract state from odometry message
        self.current_state[0] = msg.pose.pose.position.x
        self.current_state[1] = msg.pose.pose.position.y
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        self.current_state[2] = np.arctan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        self.current_state[3] = np.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
        
        # Update learning buffers
        if self.last_state is not None and self.last_control is not None:
            self.state_buffer.append(self.last_state)
            self.action_buffer.append(self.last_control)
            self.next_state_buffer.append(self.current_state)
            
            if len(self.state_buffer) % 100 == 0:  # Update model periodically
                self.update_dynamics_model()
        
        self.last_state = self.current_state.copy()
        self.solve_mpc()
    
    def reference_callback(self, msg):
        # Update reference trajectory
        data = np.array(msg.data).reshape(-1, 3)
        self.reference = data[:self.N]
    
    def solve_mpc(self):
        # Set initial state constraint
        self.opti.set_value(self.P, self.reference.T)
        self.opti.subject_to(self.X[:, 0] == self.current_state)
        
        # Set dynamics compensation if model is trained
        if self.last_control is not None:
            dynamics_error = self.predict_dynamics_error(self.current_state, self.last_control)
            self.opti.set_value(self.D, dynamics_error)
        else:
            self.opti.set_value(self.D, np.zeros(4))
        
        try:
            sol = self.opti.solve()
            u_optimal = sol.value(self.U)
            self.last_control = u_optimal[:, 0]
            self.publish_control(self.last_control)
        except:
            rospy.logwarn("MPC optimization failed!")
    
    def publish_control(self, u):
        cmd = Twist()
        # Convert acceleration and steering commands to linear and angular velocity
        cmd.linear.x = self.current_state[3] + u[0] * self.dt  # v + a*dt
        cmd.angular.z = self.current_state[3] * np.tan(u[1]) / self.L  # v*tan(δ)/L
        self.cmd_pub.publish(cmd)

if __name__ == '__main__':
    try:
        mpc = CarMPC()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 