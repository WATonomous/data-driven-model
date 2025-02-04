#!/usr/bin/env python3

import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class TrajectoryGenerator:
    def __init__(self):
        rospy.init_node('trajectory_generator')
        
        # Publishers
        self.ref_pub = rospy.Publisher('/reference', Float64MultiArray, queue_size=1)
        self.path_pub = rospy.Publisher('/reference_path', Path, queue_size=1)
        
        # Parameters
        self.update_rate = 10  # Hz
        self.N = 10  # Number of reference points (same as MPC horizon)
        
        # Trajectory parameters
        self.radius = 10.0  # meters
        self.v_ref = 5.0    # m/s
        self.t = 0.0
        
    def generate_circular_trajectory(self):
        # Generate N points along a circle
        omega = self.v_ref / self.radius  # angular velocity
        dt = 0.1  # time step (same as MPC)
        
        trajectory = []
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()
        
        for i in range(self.N):
            t = self.t + i * dt
            x = self.radius * np.cos(omega * t)
            y = self.radius * np.sin(omega * t)
            yaw = omega * t + np.pi/2  # tangent to circle
            
            trajectory.extend([x, y, yaw])
            
            # Add to path message for visualization
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            # Convert yaw to quaternion (simplified, only yaw)
            pose.pose.orientation.z = np.sin(yaw/2)
            pose.pose.orientation.w = np.cos(yaw/2)
            path_msg.poses.append(pose)
        
        # Publish reference trajectory
        ref_msg = Float64MultiArray()
        ref_msg.data = trajectory
        self.ref_pub.publish(ref_msg)
        self.path_pub.publish(path_msg)
        
        self.t += dt
    
    def run(self):
        rate = rospy.Rate(self.update_rate)
        
        while not rospy.is_shutdown():
            self.generate_circular_trajectory()
            rate.sleep()

if __name__ == '__main__':
    try:
        generator = TrajectoryGenerator()
        generator.run()
    except rospy.ROSInterruptException:
        pass 