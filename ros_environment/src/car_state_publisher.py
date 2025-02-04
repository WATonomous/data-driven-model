#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from tf.transformations import quaternion_from_euler
import tf2_ros

class CarStatePublisher:
    def __init__(self):
        rospy.init_node('car_state_publisher')
        
        # Car parameters
        self.wheelbase = 1.5  # meters
        self.max_steer = 0.6  # radians
        self.max_speed = 2.0  # m/s
        
        # State: [x, y, theta, v]
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.last_cmd = None
        self.last_cmd_time = None
        self.cmd_count = 0
        self.update_count = 0
        
        # Publishers
        self.odom_pub = rospy.Publisher('/car/odom', Odometry, queue_size=1)
        self.pose_pub = rospy.Publisher('/car/pose', PoseStamped, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('/car/cmd_vel', Twist, self.cmd_callback, queue_size=1)
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Timer for publishing state
        self.dt = 0.02  # 50 Hz for smooth visualization
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.timer_callback)
        
        rospy.loginfo("Car state publisher initialized with wheelbase=%.2fm, max_speed=%.2fm/s", 
                     self.wheelbase, self.max_speed)

    def cmd_callback(self, msg):
        """Handle velocity commands."""
        self.cmd_count += 1
        self.last_cmd = msg
        self.last_cmd_time = rospy.Time.now()
        
        # Debug every 50 commands
        if self.cmd_count % 50 == 0:
            rospy.loginfo("Received cmd %d: v=%.2f m/s, delta=%.2f rad", 
                         self.cmd_count, msg.linear.x, msg.angular.z)

    def update_state(self):
        """Update state based on current command."""
        if self.last_cmd is None:
            if self.update_count % 100 == 0:  # Debug every 100 updates
                rospy.logwarn("No command received yet")
            return

        # If command is too old, stop the car gradually
        cmd_age = (rospy.Time.now() - self.last_cmd_time).to_sec()
        if cmd_age > 0.5:
            if self.update_count % 100 == 0:  # Debug every 100 updates
                rospy.logwarn("Command too old (%.2f s), stopping car", cmd_age)
            # Gradually reduce velocity
            self.state[3] *= 0.95  # Decay factor
            if abs(self.state[3]) < 0.01:
                self.state[3] = 0.0
                self.last_cmd = None
            return

        # Extract commands
        target_v = np.clip(self.last_cmd.linear.x, -self.max_speed, self.max_speed)
        delta = np.clip(self.last_cmd.angular.z, -self.max_steer, self.max_steer)

        # Store old state for debugging
        old_state = self.state.copy()

        # Smoothly adjust velocity (acceleration limit)
        max_accel = 2.0  # m/s^2
        dv = np.clip(target_v - self.state[3], -max_accel * self.dt, max_accel * self.dt)
        self.state[3] += dv

        # Current velocity
        v = self.state[3]

        # Update state using bicycle model
        self.state[0] += v * np.cos(self.state[2]) * self.dt
        self.state[1] += v * np.sin(self.state[2]) * self.dt
        self.state[2] += v * np.tan(delta) / self.wheelbase * self.dt

        # Normalize angle to [-pi, pi]
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))

        # Debug state changes every 100 updates
        self.update_count += 1
        if self.update_count % 100 == 0:
            dx = self.state[0] - old_state[0]
            dy = self.state[1] - old_state[1]
            dist = np.sqrt(dx*dx + dy*dy)
            rospy.loginfo("State update %d: moved %.3fm, current_vel=%.2f m/s, heading=%.2f rad", 
                         self.update_count, dist, v, self.state[2])

    def timer_callback(self, event):
        """Publish car state."""
        # Update state
        self.update_state()
        
        now = rospy.Time.now()
        
        # Publish odometry
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"
        
        # Position
        odom.pose.pose.position.x = self.state[0]
        odom.pose.pose.position.y = self.state[1]
        odom.pose.pose.position.z = 0.0
        
        # Orientation
        q = quaternion_from_euler(0, 0, self.state[2])
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        
        # Velocity
        odom.twist.twist.linear.x = self.state[3]
        if self.last_cmd is not None:
            odom.twist.twist.angular.z = self.state[3] * np.tan(self.last_cmd.angular.z) / self.wheelbase
        
        self.odom_pub.publish(odom)
        
        # Publish pose
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self.pose_pub.publish(pose)
        
        # Broadcast transform from odom to base_footprint
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint"
        t.transform.translation.x = self.state[0]
        t.transform.translation.y = self.state[1]
        t.transform.translation.z = 0.0
        t.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)
        
        # Log TF publishing every 100 updates
        if self.update_count % 100 == 0:
            rospy.loginfo("Publishing TF: odom -> base_footprint at (%.2f, %.2f, %.2f)",
                         t.transform.translation.x,
                         t.transform.translation.y,
                         t.transform.translation.z)

if __name__ == '__main__':
    try:
        car_state_publisher = CarStatePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 