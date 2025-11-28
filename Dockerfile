FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS builder
RUN apt-get update && apt-get install -y python3.12 python3-venv python3-dev git build-essential cmake
RUN git clone https://github.com/thu-ml/SageAttention.git /SageAttention
WORKDIR /SageAttention
RUN sed -i 's/HAS_SM80 = False/HAS_SM80 = True/' setup.py && sed -i 's/compute_capabilities = set()/compute_capabilities = set(["8.0"])/' setup.py
RUN python3 -m venv .venv
RUN . .venv/bin/activate && \
    pip3 install packaging && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    cd sageattention3_blackwell && python3 setup.py bdist_wheel && ls dist && cd /SageAttention && \
    python3 setup.py bdist_wheel && \
    mv dist/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl /sageattention-2.2.0-cp312-cp312-linux_x86_64.whl && \
    rm -rf dist build 

FROM alpine:edge
COPY --from=builder /sageattention-2.2.0-cp312-cp312-linux_x86_64.whl /sageattention-2.2.0-cp312-cp312-linux_x86_64.whl
