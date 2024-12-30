#!/usr/bin/env python3

import numpy as np
import casadi as ca
import rospy
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sklearn.linear_model import Ridge
from collections import deque
from nav_msgs.msg import Path

class CarMPC:
    def __init__(self):
        # MPC parameters
        self.dt = 0.1  # Time step [s]
        self.N = 10    # Prediction horizon
        self.L = 2.7   # Wheelbase [m]
        
        # State constraints
        self.v_max = 2.0   # Maximum velocity [m/s] (reduced for safety)
        self.delta_max = 0.6  # Maximum steering angle [rad]
        self.a_max = 1.0   # Maximum acceleration [m/s^2]
        
        # Weights for the cost function
        self.Q = np.diag([10.0, 10.0, 1.0, 0.1])  # State weights [x, y, yaw, v]
        self.R = np.diag([1.0, 1.0])              # Input weights [acceleration, steering]
        
        # Learning components
        self.buffer_size = 1000
        self.state_buffer = deque(maxlen=self.buffer_size)
        self.action_buffer = deque(maxlen=self.buffer_size)
        self.next_state_buffer = deque(maxlen=self.buffer_size)
        self.error_buffer = deque(maxlen=self.buffer_size)  # Store prediction errors
        self.dynamics_model = Ridge(alpha=0.1)  # Reduced regularization for better fitting
        self.model_trained = False
        self.min_samples_for_training = 50  # Reduced for faster initial learning
        self.training_interval = 20  # Update model more frequently
        
        # Initialize ROS node
        rospy.init_node('car_mpc_controller')
        
        # Publishers and subscribers
        self.cmd_pub = rospy.Publisher('/car/cmd_vel', Twist, queue_size=1)
        self.state_sub = rospy.Subscriber('/car/odom', Odometry, self.state_callback)
        self.ref_sub = rospy.Subscriber('/car/reference_path', Path, self.reference_callback)
        
        # Debug publisher
        self.debug_pub = rospy.Publisher('/car/mpc_debug', Float64MultiArray, queue_size=1)
        
        # Initialize state and reference
        self.current_state = np.zeros(4)  # [x, y, yaw, v]
        self.reference = np.zeros((self.N, 3))  # N points of [x, y, yaw]
        self.last_state = None
        self.last_control = None
        
        self.setup_mpc()
        
        rospy.loginfo("Data-driven MPC initialized with buffer_size=%d, training_interval=%d", 
                     self.buffer_size, self.training_interval)
        
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
        """Update the learned dynamics model with collected data."""
        if len(self.state_buffer) < self.min_samples_for_training:
            rospy.logwarn_throttle(5.0, "Not enough samples for training (%d/%d)", 
                                 len(self.state_buffer), self.min_samples_for_training)
            return
        
        try:
            # Prepare training data
            states = np.array(self.state_buffer)
            actions = np.array(self.action_buffer)
            next_states = np.array(self.next_state_buffer)
            
            # Feature vector: [x, y, yaw, v, cos(yaw), sin(yaw), v*cos(yaw), v*sin(yaw), a, delta]
            X = np.column_stack([
                states,  # [x, y, yaw, v]
                np.cos(states[:, 2]),  # cos(yaw)
                np.sin(states[:, 2]),  # sin(yaw)
                states[:, 3] * np.cos(states[:, 2]),  # v*cos(yaw)
                states[:, 3] * np.sin(states[:, 2]),  # v*sin(yaw)
                actions  # [a, delta]
            ])
            
            # Target: state changes
            y = next_states - states
            
            # Train the model
            self.dynamics_model.fit(X, y)
            self.model_trained = True
            
            # Compute and store prediction errors
            y_pred = self.dynamics_model.predict(X)
            error = np.mean(np.abs(y - y_pred), axis=0)
            self.error_buffer.append(error)
            
            # Publish model error
            error_msg = Float64MultiArray()
            error_msg.data = error.tolist()
            self.model_error_pub.publish(error_msg)
            
            rospy.loginfo("Dynamics model updated with %d samples. Mean abs error: x=%.3f, y=%.3f, yaw=%.3f, v=%.3f", 
                         len(states), error[0], error[1], error[2], error[3])
                         
        except Exception as e:
            rospy.logerr("Failed to update dynamics model: %s", str(e))

    def predict_dynamics_error(self, state, action):
        """Predict the error in dynamics model."""
        if not self.model_trained:
            return np.zeros(4)
        
        try:
            # Create feature vector
            X = np.array([[
                state[0], state[1], state[2], state[3],  # [x, y, yaw, v]
                np.cos(state[2]),  # cos(yaw)
                np.sin(state[2]),  # sin(yaw)
                state[3] * np.cos(state[2]),  # v*cos(yaw)
                state[3] * np.sin(state[2]),  # v*sin(yaw)
                action[0], action[1]  # [a, delta]
            ]])
            
            # Predict state change
            state_change_pred = self.dynamics_model.predict(X)[0]
            
            # Calculate nominal state change from bicycle model
            v = state[3]
            nominal_change = np.array([
                v * np.cos(state[2]) * self.dt,
                v * np.sin(state[2]) * self.dt,
                v * np.tan(action[1]) / self.L * self.dt,
                action[0] * self.dt
            ])
            
            # Return the difference between predicted and nominal
            return state_change_pred - nominal_change
            
        except Exception as e:
            rospy.logerr("Failed to predict dynamics error: %s", str(e))
            return np.zeros(4)

    def state_callback(self, msg):
        """Handle odometry updates and collect data."""
        # Extract state from odometry message
        old_state = self.current_state.copy()
        
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
            
            # Update model periodically
            if len(self.state_buffer) % self.training_interval == 0:
                self.update_dynamics_model()
        
        # Store current state for next update
        self.last_state = self.current_state.copy()
        
        # Solve MPC
        self.solve_mpc()
        
        # Debug state changes periodically
        if len(self.state_buffer) % 100 == 0:
            dx = self.current_state[0] - old_state[0]
            dy = self.current_state[1] - old_state[1]
            dist = np.sqrt(dx*dx + dy*dy)
            rospy.loginfo("State update: moved %.3fm, vel=%.2f m/s, yaw=%.2f rad", 
                         dist, self.current_state[3], self.current_state[2])
    
    def reference_callback(self, msg):
        """Handle reference path updates."""
        try:
            # Extract N points from the path
            ref_points = []
            for i in range(min(self.N, len(msg.poses))):
                pose = msg.poses[i].pose
                # Extract yaw from quaternion
                q = pose.orientation
                yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
                ref_points.append([pose.position.x, pose.position.y, yaw])
            
            # If we got fewer points than N, repeat the last point
            while len(ref_points) < self.N:
                ref_points.append(ref_points[-1])
            
            self.reference = np.array(ref_points)
            
        except Exception as e:
            rospy.logerr("Failed to process reference path: %s", str(e))

    def publish_learning_stats(self):
        """Publish learning statistics for visualization."""
        if not self.error_buffer:
            return
            
        # Compute statistics
        recent_errors = np.array(list(self.error_buffer))
        mean_error = np.mean(recent_errors, axis=0)
        max_error = np.max(recent_errors, axis=0)
        
        # Create message
        stats_msg = Float64MultiArray()
        stats_msg.data = [
            mean_error[0],  # x error
            mean_error[1],  # y error
            mean_error[2],  # yaw error
            mean_error[3],  # velocity error
            max_error[0],   # max x error
            max_error[1],   # max y error
            max_error[2],   # max yaw error
            max_error[3],   # max velocity error
            len(self.state_buffer),  # number of samples
            float(self.model_trained)  # model status
        ]
        self.learning_stats_pub.publish(stats_msg)

    def publish_prediction_viz(self, predicted_states):
        """Publish predicted trajectory for visualization."""
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "odom"
        
        for state in predicted_states.T:  # states is 4xN
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = state[0]
            pose.pose.position.y = state[1]
            
            # Convert yaw to quaternion
            yaw = state[2]
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            pose.pose.orientation.z = sy
            pose.pose.orientation.w = cy
            
            path_msg.poses.append(pose)
        
        self.prediction_viz_pub.publish(path_msg)

    def publish_debug_info(self, u_optimal=None, error=None):
        """Publish debug information for visualization."""
        debug_msg = Float64MultiArray()
        debug_data = []
        
        # Current state
        debug_data.extend([
            self.current_state[0],  # x position
            self.current_state[1],  # y position
            self.current_state[2],  # yaw
            self.current_state[3],  # velocity
        ])
        
        # Control commands (if available)
        if u_optimal is not None:
            debug_data.extend([
                u_optimal[0],  # acceleration
                u_optimal[1],  # steering
            ])
        else:
            debug_data.extend([0.0, 0.0])
        
        # Learning progress
        debug_data.extend([
            len(self.state_buffer),  # number of samples
            float(self.model_trained),  # model status
        ])
        
        # Model error (if available)
        if error is not None:
            debug_data.extend(error)
        else:
            debug_data.extend([0.0, 0.0, 0.0, 0.0])
        
        debug_msg.data = debug_data
        self.debug_pub.publish(debug_msg)

    def solve_mpc(self):
        """Solve the MPC optimization problem."""
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
            x_optimal = sol.value(self.X)
            
            # Store control for learning
            self.last_control = u_optimal[:, 0]
            
            # Publish control command
            self.publish_control(self.last_control)
            
            # Publish debug information
            self.publish_debug_info(u_optimal[:, 0], dynamics_error if self.model_trained else None)
            
        except Exception as e:
            rospy.logwarn("MPC optimization failed: %s", str(e))
            # Publish debug info even on failure
            self.publish_debug_info()
    
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