FROM ros:jazzy-ros-base-noble

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    python3-colcon-common-extensions \
    python3-pip \
    python3-pytest \
    python3-rosdep \
    python3-vcstool \
    ros-jazzy-cv-bridge \
    ros-jazzy-depth-image-proc \
    ros-jazzy-diagnostic-updater \
    ros-jazzy-image-transport \
    ros-jazzy-nav2-behaviors \
    ros-jazzy-nav2-bt-navigator \
    ros-jazzy-nav2-common \
    ros-jazzy-nav2-controller \
    ros-jazzy-nav2-dwb-controller \
    ros-jazzy-nav2-lifecycle-manager \
    ros-jazzy-nav2-navfn-planner \
    ros-jazzy-nav2-planner \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-rosbag2-storage-mcap \
    ros-jazzy-tf-transformations \
    ros-jazzy-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

COPY upstream.repos /tmp/upstream.repos
COPY containers/robotics/ianvs-jazzy-shutdown.patch /tmp/ianvs-jazzy-shutdown.patch

RUN mkdir -p /opt/hydra_ws/src \
    && vcs import --workers 1 /opt/hydra_ws/src < /tmp/upstream.repos \
    && git -C /opt/hydra_ws/src/ianvs apply /tmp/ianvs-jazzy-shutdown.patch

# The preceding image layer intentionally removes apt indexes.  rosdep invokes
# apt directly, so refresh the indexes in the same layer before resolving the
# pinned Hydra workspace.
RUN apt-get update \
    && source /opt/ros/jazzy/setup.bash \
    && rosdep update \
    && rosdep install --from-paths /opt/hydra_ws/src --ignore-src -r -y \
    && rm -rf /var/lib/apt/lists/*

RUN source /opt/ros/jazzy/setup.bash \
    && cd /opt/hydra_ws \
    && colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCONFIG_UTILS_ENABLE_ROS=OFF

COPY containers/robotics/entrypoint.sh /usr/local/bin/mns-robotics-entrypoint
RUN chmod +x /usr/local/bin/mns-robotics-entrypoint

WORKDIR /workspace/ros_ws
ENTRYPOINT ["/usr/local/bin/mns-robotics-entrypoint"]
CMD ["bash"]
