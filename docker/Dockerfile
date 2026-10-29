# ROS 2 Humble (Ubuntu 22.04, Python 3.10) + the adexpro conda env, so rclpy and
# torch share one interpreter. The repo itself is not baked in: it is
# bind-mounted at the same path as on the host (see scripts/docker_run.sh), so
# --symlink-install paths stay valid on both sides. Build context = repo root.
#
#   docker build -f docker/Dockerfile -t adexpro:humble .
FROM osrf/ros:humble-desktop-full

ARG USERNAME=jingfan
ARG USER_UID=1000
ARG USER_GID=1000
ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo tmux git wget curl less nano \
        python3-colcon-common-extensions python3-rosdep python3-pip \
        ros-humble-gazebo-ros-pkgs ros-humble-octomap-server ros-humble-joy \
    && rm -rf /var/lib/apt/lists/*

# System deps of every workspace package, resolved from their package.xml files
# only (.dockerignore keeps the rest of the sources out of the build context).
COPY dev_explore /tmp/src/dev_explore
COPY dev_baselines /tmp/src/dev_baselines
RUN apt-get update \
    && (rosdep init 2>/dev/null || true) && rosdep update --rosdistro humble \
    && rosdep install --from-paths /tmp/src --ignore-src -r -y --rosdistro humble \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && mkdir -p /opt/conda && chown ${USERNAME}:${USERNAME} /opt/conda

USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Miniconda + adexpro from docker/environment.yml. range-libc is not on PyPI: drop it from the pip
# list and build it from MapEx's bundled source instead.
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -u -p /opt/conda && rm /tmp/miniconda.sh
COPY --chown=${USERNAME}:${USERNAME} docker/environment.yml /tmp/environment.yml
RUN sed -i '/range-libc/d' /tmp/environment.yml \
    && /opt/conda/bin/conda env create -n adexpro -f /tmp/environment.yml \
    && /opt/conda/bin/conda clean -afy && rm -rf ~/.cache/pip
COPY --chown=${USERNAME}:${USERNAME} dev_baselines/mapex/MapEx/range_libc /tmp/range_libc
RUN cd /tmp/range_libc/pywrapper \
    && /opt/conda/envs/adexpro/bin/python -m pip install --no-build-isolation --no-cache-dir . \
    && rm -rf /tmp/range_libc

RUN /opt/conda/bin/conda init bash \
    && echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# voxblox declares protobuf-compiler, which is not a rosdep key, so rosdep skips it.
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends protobuf-compiler \
    && sudo rm -rf /var/lib/apt/lists/*

# --net=host gives the container the host's hostname, so mark the prompt to
# tell the two shells apart.
RUN echo 'PS1="[docker] $PS1"' >> ~/.bashrc

CMD ["bash"]
