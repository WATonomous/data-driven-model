#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from tf.transformations import quaternion_from_euler
import tf2_ros
import geometry_msgs.msg

class CarStatePublisher:
    def __init__(self):
        rospy.init_node('car_state_publisher')
        
        # Car parameters
        self.wheelbase = 2.7  # meters
        self.max_steer = 0.6  # radians
        self.max_speed = 2.0  # m/s
        
        # State: [x, y, theta, v]
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Publishers
        self.odom_pub = rospy.Publisher('/car/odom', Odometry, queue_size=10)
        self.pose_pub = rospy.Publisher('/car/pose', PoseStamped, queue_size=10)
        
        # Subscribers
        rospy.Subscriber('/car/cmd_vel', Twist, self.cmd_callback)
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Timer for publishing state
        self.dt = 0.1  # 10 Hz
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.timer_callback)
        
        rospy.loginfo("Car state publisher initialized")

    def bicycle_model(self, state, v_cmd, delta):
        """Simple bicycle model."""
        x, y, theta, v = state
        
        # Simple dynamics
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = v * np.tan(delta) / self.wheelbase
        dv = (v_cmd - v) / 0.5  # Simple first-order velocity response
        
        return np.array([dx, dy, dtheta, dv])

    def cmd_callback(self, msg):
        """Handle velocity commands."""
        v_cmd = np.clip(msg.linear.x, -self.max_speed, self.max_speed)
        delta = np.clip(msg.angular.z, -self.max_steer, self.max_steer)
        
        # Integrate dynamics
        derivatives = self.bicycle_model(self.state, v_cmd, delta)
        self.state += derivatives * self.dt

    def timer_callback(self, event):
        """Publish car state."""
        now = rospy.Time.now()
        
        # Publish odometry
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        
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
        odom.twist.twist.angular.z = self.state[3] * np.tan(self.state[2]) / self.wheelbase
        
        self.odom_pub.publish(odom)
        
        # Publish pose
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self.pose_pub.publish(pose)
        
        # Broadcast transform
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = self.state[0]
        t.transform.translation.y = self.state[1]
        t.transform.translation.z = 0.0
        t.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

if __name__ == '__main__':
    try:
        car_state_publisher = CarStatePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 