FROM multi-nav-scenegraph/robotics:jazzy-hydra-8f3b7e3

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG FORESTNAVIGATION_REVISION=0b29c399754f510499bfe9cc9d592cba215a7161
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

LABEL org.opencontainers.image.source="https://github.com/LxRoboticsLab/ForestNavigation" \
      org.opencontainers.image.revision="${FORESTNAVIGATION_REVISION}"

# The ML layer is optional; coverage and Nav2 keep using the smaller robotics
# image.  Versions are pinned and the CUDA wheels retain a CPU fallback for
# checkpoint/interface validation on hosts without a usable NVIDIA driver.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --break-system-packages \
      --index-url "${PYTORCH_INDEX_URL}" \
      torch==2.7.1 torchvision==0.22.1

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --break-system-packages \
      --ignore-installed --no-deps \
      diffusers==0.11.1 \
      efficientnet-pytorch==0.7.1 \
      einops==0.8.1 \
      h5py==3.12.1 \
      huggingface-hub==0.25.2 \
      requests==2.32.3 \
      certifi==2024.8.30 \
      charset-normalizer==3.4.0 \
      idna==3.10 \
      urllib3==2.2.3 \
      regex==2024.11.6 \
      tqdm==4.67.1

# Import only the package from the locally audited, pinned source checkout.
# The build target verifies its revision and cleanliness before Docker starts;
# LFS model assets remain runtime-mounted and are not baked into the image.
COPY models/ForestNavigation/forest_nav/setup.py /opt/forestnavigation/forest_nav/setup.py
COPY models/ForestNavigation/forest_nav/navdiffusion /opt/forestnavigation/forest_nav/navdiffusion
RUN python3 -m pip install --break-system-packages --no-deps \
      /opt/forestnavigation/forest_nav \
    && python3 -m pip check
