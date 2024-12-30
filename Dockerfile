FROM ros:noetic-ros-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-catkin-tools \
    python3-catkin-pkg \
    ros-noetic-tf2-ros \
    ros-noetic-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*

# Create catkin workspace
WORKDIR /catkin_ws
RUN mkdir -p /catkin_ws/src

# Copy the package
COPY . /catkin_ws/src/car_mpc_control/

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy>=1.18.0 \
    scipy>=1.4.0 \
    casadi>=3.5.5 \
    scikit-learn>=0.24.0

# Initialize catkin workspace
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash && \
    cd /catkin_ws && \
    catkin init && \
    catkin config --extend /opt/ros/noetic && \
    catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    catkin build'

# Source the workspace in bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

# Make Python files executable
RUN chmod +x /catkin_ws/src/car_mpc_control/src/*.py

# Create a launch script
RUN echo '#!/bin/bash\n\
source /opt/ros/noetic/setup.bash\n\
source /catkin_ws/devel/setup.bash\n\
export ROS_IP=$(hostname -I | cut -d" " -f1)\n\
roscore & \
sleep 2 && \
roslaunch car_mpc_control car_mpc.launch\n'\
> /start.sh && chmod +x /start.sh

# Expose Foxglove Bridge port
EXPOSE 8765

CMD ["/start.sh"] 