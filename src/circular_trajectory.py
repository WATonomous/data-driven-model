#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class CircularTrajectory:
    def __init__(self):
        rospy.init_node('circular_trajectory')
        
        # Trajectory parameters
        self.radius = 5.0  # meters
        self.linear_velocity = 1.0  # m/s
        self.angular_velocity = self.linear_velocity / self.radius  # rad/s
        
        # Publishers
        self.cmd_pub = rospy.Publisher('/car/cmd_vel', Twist, queue_size=10)
        self.path_pub = rospy.Publisher('/car/reference_path', Path, queue_size=10)
        
        # Timer for publishing commands
        self.dt = 0.1  # 10 Hz
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.timer_callback)
        
        rospy.loginfo("Circular trajectory generator initialized")
        
        # Generate and publish reference path
        self.publish_reference_path()

    def publish_reference_path(self):
        """Generate and publish the reference circular path."""
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "odom"
        
        # Generate points along the circle
        t = np.linspace(0, 2*np.pi, 100)
        for angle in t:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = self.radius * np.cos(angle)
            pose.pose.position.y = self.radius * np.sin(angle)
            pose.pose.position.z = 0.0
            
            # Calculate orientation tangent to circle
            yaw = angle + np.pi/2  # Tangent angle
            qx, qy, qz, qw = self.get_quaternion_from_yaw(yaw)
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            
            path.poses.append(pose)
        
        self.path_pub.publish(path)

    def get_quaternion_from_yaw(self, yaw):
        """Convert yaw angle to quaternion."""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        return 0.0, 0.0, sy, cy

    def timer_callback(self, event):
        """Publish command velocities for circular motion."""
        cmd = Twist()
        cmd.linear.x = self.linear_velocity
        cmd.angular.z = self.angular_velocity
        self.cmd_pub.publish(cmd)

if __name__ == '__main__':
    try:
        circular_trajectory = CircularTrajectory()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 