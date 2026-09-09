FROM nvcr.io/nvidia/isaac-lab:2.3.1

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# The simulator process uses Isaac Sim's internal Jazzy client libraries and
# only standard ROS interfaces. Project nodes remain in the robotics image.
RUN chmod +x /workspace/isaaclab/isaaclab.sh

WORKDIR /workspace
ENTRYPOINT []
