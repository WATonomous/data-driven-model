#!/usr/bin/env python3

import numpy as np
import casadi as ca
import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64MultiArray
from tf.transformations import euler_from_quaternion

class CarMPCController:
    def __init__(self):
        rospy.init_node('car_mpc_controller')
        
        # MPC Parameters
        self.dt = 0.1  # Time step [s]
        self.N = 10    # Prediction horizon
        self.wheelbase = 1.5  # Vehicle wheelbase [m]
        
        # State and input constraints
        self.v_max = 1.0    # Maximum velocity [m/s]
        self.v_min = 0.0    # Minimum velocity [m/s]
        self.delta_max = 0.6 # Maximum steering angle [rad]
        self.a_max = 0.5    # Maximum acceleration [m/s^2]
        
        # Cost weights
        self.Q_x = 10.0     # Position x cost
        self.Q_y = 10.0     # Position y cost
        self.Q_theta = 5.0  # Heading cost
        self.Q_v = 1.0      # Velocity cost
        self.R_a = 0.1      # Acceleration input cost
        self.R_delta = 0.1  # Steering input cost
        
        # Initialize state and reference
        self.current_state = np.zeros(4)  # [x, y, theta, v]
        self.reference_path = None
        self.reference_idx = 0
        self.state_update_count = 0  # Counter for state updates
        
        # Publishers and Subscribers
        self.cmd_pub = rospy.Publisher('/car/cmd_vel', Twist, queue_size=1)
        self.mpc_path_pub = rospy.Publisher('/car/mpc_prediction', Path, queue_size=1)
        self.debug_pub = rospy.Publisher('/car/mpc_debug', Float64MultiArray, queue_size=1)
        
        rospy.Subscriber('/car/odom', Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber('/car/reference_path', Path, self.reference_callback, queue_size=1)
        
        # Setup MPC solver
        self.setup_mpc()
        
        # Control timer
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.control_callback)
        
        rospy.loginfo("MPC Controller initialized")

    def setup_mpc(self):
        """Setup the MPC optimization problem."""
        # Create a fresh optimization problem
        self.opti = ca.Opti()
        
        # Decision variables
        self.X = self.opti.variable(4, self.N + 1)  # states [x, y, theta, v]
        self.U = self.opti.variable(2, self.N)      # inputs [acceleration, steering_angle]
        
        # Parameters for the reference trajectory
        self.P = self.opti.parameter(4, self.N + 1)
        
        # Cost function
        cost = 0
        for k in range(self.N + 1):
            # State error cost
            state_error = self.X[:, k] - self.P[:, k]
            cost += (state_error[0]**2 * self.Q_x +  # x position
                    state_error[1]**2 * self.Q_y +   # y position
                    state_error[2]**2 * self.Q_theta +  # heading
                    state_error[3]**2 * self.Q_v)    # velocity
            
            if k < self.N:
                # Control input cost
                cost += (self.U[0, k]**2 * self.R_a +      # acceleration
                        self.U[1, k]**2 * self.R_delta)    # steering
        
        self.opti.minimize(cost)
        
        # System dynamics constraints
        for k in range(self.N):
            # Current state
            x = self.X[0, k]
            y = self.X[1, k]
            theta = self.X[2, k]
            v = self.X[3, k]
            
            # Control inputs
            a = self.U[0, k]      # acceleration
            delta = self.U[1, k]  # steering angle
            
            # State equations
            x_next = x + v * ca.cos(theta) * self.dt
            y_next = y + v * ca.sin(theta) * self.dt
            theta_next = theta + v * ca.tan(delta) / self.wheelbase * self.dt
            v_next = v + a * self.dt
            
            # Next state constraint
            self.opti.subject_to(self.X[:, k+1] == ca.vertcat(x_next, y_next, theta_next, v_next))
        
        # Input constraints
        self.opti.subject_to(self.opti.bounded(-self.a_max, self.U[0, :], self.a_max))
        self.opti.subject_to(self.opti.bounded(-self.delta_max, self.U[1, :], self.delta_max))
        
        # State constraints
        self.opti.subject_to(self.opti.bounded(self.v_min, self.X[3, :], self.v_max))
        
        # Set solver options
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': 100
        }
        self.opti.solver('ipopt', opts)

    def odom_callback(self, msg):
        """Handle odometry updates."""
        # Extract state from odometry
        old_state = self.current_state.copy()
        
        self.current_state[0] = msg.pose.pose.position.x
        self.current_state[1] = msg.pose.pose.position.y
        
        # Get yaw from quaternion
        q = msg.pose.pose.orientation
        euler = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_state[2] = euler[2]  # yaw
        
        # Get velocity
        self.current_state[3] = np.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
        
        # Debug state changes every 100 updates
        self.state_update_count += 1
        if self.state_update_count % 100 == 0:
            dx = self.current_state[0] - old_state[0]
            dy = self.current_state[1] - old_state[1]
            dist = np.sqrt(dx*dx + dy*dy)
            rospy.loginfo("State update %d: moved %.3fm, current_vel=%.2f m/s, heading=%.2f rad",
                         self.state_update_count, dist, self.current_state[3], self.current_state[2])

    def reference_callback(self, msg):
        """Handle new reference path."""
        self.reference_path = msg.poses
        self.reference_idx = 0
        rospy.loginfo("Received new reference path with %d points", len(msg.poses))

    def get_reference_states(self):
        """Get reference states for MPC horizon."""
        if self.reference_path is None or len(self.reference_path) < 2:
            rospy.logwarn_throttle(1.0, "No valid reference path available")
            return np.tile(self.current_state, (self.N + 1, 1)).T
        
        ref_states = np.zeros((4, self.N + 1))
        n_points = len(self.reference_path)
        
        # Calculate desired velocity based on path curvature
        dx = self.reference_path[1].pose.position.x - self.reference_path[0].pose.position.x
        dy = self.reference_path[1].pose.position.y - self.reference_path[0].pose.position.y
        ds = np.sqrt(dx*dx + dy*dy)
        desired_velocity = 0.5  # m/s (constant for now)
        
        for i in range(self.N + 1):
            idx = (self.reference_idx + i) % n_points
            next_idx = (idx + 1) % n_points
            
            # Current pose
            pose = self.reference_path[idx].pose
            next_pose = self.reference_path[next_idx].pose
            
            # Position
            ref_states[0,i] = pose.position.x
            ref_states[1,i] = pose.position.y
            
            # Calculate heading from current to next point
            dx = next_pose.position.x - pose.position.x
            dy = next_pose.position.y - pose.position.y
            ref_states[2,i] = np.arctan2(dy, dx)
            
            # Desired velocity
            ref_states[3,i] = desired_velocity
        
        return ref_states

    def control_callback(self, event):
        """Main MPC control loop."""
        if self.reference_path is None:
            rospy.logwarn_throttle(1.0, "No reference path received")
            return
            
        try:
            # Reset the optimizer
            self.setup_mpc()
            
            # Set initial state constraint
            self.opti.subject_to(self.X[:, 0] == self.current_state)
            
            # Get reference trajectory
            ref_states = self.get_reference_states()
            self.opti.set_value(self.P, ref_states)
            
            # Solve optimization problem
            sol = self.opti.solve()
            
            # Extract optimal control inputs
            u_optimal = sol.value(self.U)
            x_optimal = sol.value(self.X)
            
            # Send control commands
            cmd = Twist()
            cmd.linear.x = x_optimal[3, 1]  # Use predicted velocity directly
            cmd.angular.z = u_optimal[1, 0]  # steering angle
            self.cmd_pub.publish(cmd)
            
            # Publish predicted trajectory
            self.publish_prediction(x_optimal)
            
            # Update reference index
            self.reference_idx = (self.reference_idx + 1) % len(self.reference_path)
            
            # Debug info
            rospy.loginfo_throttle(1.0, 
                f"MPC: v={cmd.linear.x:.2f} m/s, delta={cmd.angular.z:.2f} rad, cost={sol.value(self.opti.f):.2f}")
            
        except Exception as e:
            rospy.logwarn(f"MPC optimization failed: {str(e)}")
            # Send zero commands on failure
            cmd = Twist()
            self.cmd_pub.publish(cmd)

    def publish_prediction(self, prediction):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "map"
        
        for i in range(prediction.shape[1]):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = prediction[0,i]
            pose.pose.position.y = prediction[1,i]
            pose.pose.position.z = 0.0
            
            # Convert yaw to quaternion
            q = self.yaw_to_quaternion(prediction[2,i])
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            
            path.poses.append(pose)
        
        self.mpc_path_pub.publish(path)

    def yaw_to_quaternion(self, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        return [0.0, 0.0, sy, cy]

if __name__ == '__main__':
    try:
        controller = CarMPCController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 