#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist

class CircularTrajectory:
    def __init__(self):
        rospy.init_node('circular_trajectory')
        
        # Trajectory parameters
        self.radius = 3.0  # meters
        self.desired_velocity = 0.5  # m/s (increased slightly)
        self.num_points = 200
        
        # Publishers
        self.path_pub = rospy.Publisher('/car/reference_path', Path, queue_size=1, latch=True)
        self.cmd_pub = rospy.Publisher('/car/cmd_vel', Twist, queue_size=1)
        
        # Generate and publish initial path
        self.publish_path()
        
        # Timer for path updates (every 5 seconds)
        self.path_timer = rospy.Timer(rospy.Duration(5.0), self.path_timer_callback)
        
        # Timer for velocity commands (50 Hz)
        self.cmd_timer = rospy.Timer(rospy.Duration(0.02), self.cmd_timer_callback)
        
        # Current angle in the circle
        self.current_angle = 0.0
        
        rospy.loginfo("Circular trajectory generator initialized with radius %.1fm and velocity %.1fm/s", 
                     self.radius, self.desired_velocity)

    def get_quaternion_from_yaw(self, yaw):
        """Convert yaw angle to quaternion."""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        return [0.0, 0.0, sy, cy]

    def publish_path(self):
        """Generate and publish the reference circular path."""
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "odom"
        
        # Generate points along the circle
        angles = np.linspace(0, 2*np.pi, self.num_points, endpoint=False)
        
        for angle in angles:
            pose = PoseStamped()
            pose.header = path.header
            
            # Position
            pose.pose.position.x = self.radius * np.cos(angle)
            pose.pose.position.y = self.radius * np.sin(angle)
            pose.pose.position.z = 0.0
            
            # Orientation (tangent to circle)
            yaw = angle + np.pi/2  # Tangent angle
            q = self.get_quaternion_from_yaw(yaw)
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            
            path.poses.append(pose)
        
        self.path_pub.publish(path)
        rospy.loginfo("Published circular path: radius=%.1fm, points=%d", 
                     self.radius, self.num_points)

    def cmd_timer_callback(self, event):
        """Generate and publish velocity commands."""
        cmd = Twist()
        
        # Linear velocity is constant
        cmd.linear.x = self.desired_velocity
        
        # Angular velocity (v/r) for circular motion
        cmd.angular.z = self.desired_velocity / self.radius
        
        # Update current angle
        self.current_angle += (self.desired_velocity / self.radius) * 0.02  # dt = 0.02s
        if self.current_angle > 2*np.pi:
            self.current_angle -= 2*np.pi
        
        self.cmd_pub.publish(cmd)

    def path_timer_callback(self, event):
        """Periodically update the path."""
        self.publish_path()

if __name__ == '__main__':
    try:
        trajectory = CircularTrajectory()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 