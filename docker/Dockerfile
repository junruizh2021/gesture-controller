FROM ubuntu:22.04
ENV http_proxy=http://child-prc.intel.com:913
ENV https_proxy=http://child-prc.intel.com:913

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    git build-essential software-properties-common ca-certificates wget curl vim libgl1-mesa-glx libgl1-mesa-dri libglu1-mesa libglu1-mesa-dev

RUN apt-get update && apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --upgrade pip

# install intel level zero driver
RUN wget https://github.com/intel/linux-npu-driver/releases/download/v1.17.0/intel-driver-compiler-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb && \
    wget https://github.com/intel/linux-npu-driver/releases/download/v1.17.0/intel-fw-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb && \
    wget https://github.com/intel/linux-npu-driver/releases/download/v1.17.0/intel-level-zero-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb && \
    apt install -y libtbb12 && \
    dpkg -i *.deb && \
    wget https://github.com/oneapi-src/level-zero/releases/download/v1.21.9/level-zero_1.21.9+u22.04_amd64.deb && \ 
    dpkg -i level-zero*.deb

# install openvino
RUN pip install nncf onnx
RUN pip install git+https://github.com/huggingface/optimum-intel.git
RUN pip install transformers >=4.51.3
RUN pip install --pre -U openvino-genai openvino openvino-tokenizers --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
RUN pip install websockets opencv-python pyserial fashionstar_uart_sdk && \
    pip cache purge
RUN rm -rf /var/lib/apt/lists/*