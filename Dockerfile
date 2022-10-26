FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
# torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt update && apt -y install sudo
RUN sudo apt -y install vim build-essential git libgtk2.0-dev libgl1-mesa-dev
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tmux \
        python3-pip

COPY . /workspace

# Using SparseConvNet with cuda, requires cuda present in build time.
# So either set your docker runtime to nvidia and uncomment this, or
#  run run this command inside the container once running
#WORKDIR /workspace/SparseConvNet
#RUN bash develop.sh

WORKDIR /workspace
RUN pip install -e .
RUN pip install protobuf==3.20.*

RUN mkdir -p /workspace/persistent_storage/checkpoints/
RUN mkdir -p /workspace/persistent_storage/runs/
