cmake_minimum_required(VERSION 3.0.2)
project(car_mpc_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
  tf2_ros
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    nav_msgs
    geometry_msgs
    tf2_ros
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/car_mpc.py
  src/trajectory_generator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
) 