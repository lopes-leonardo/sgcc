from pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update

RUN apt-get install nano -y

RUN pip3 install numpy sklearn scipy cython

RUN pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113>

RUN mkdir /workspaces

COPY ./* /workspaces/sgcc

WORKDIR /workspaces/sgcc
