FROM lmsysorg/sglang:v0.5.16

WORKDIR /sgl-workspace

# env
ARG PIP_NO_CACHE_DIR=1
RUN pip config set global.break-system-packages true

# compile
RUN MAX_JOBS=128 pip install -v --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git
RUN MAX_JOBS=128 pip install --no-build-isolation causal-conv1d==1.6.1 mamba-ssm==2.3.1
RUN MAX_JOBS=128 NVTE_FRAMEWORK=pytorch NVTE_BUILD_THREADS_PER_JOB=4 pip install -v --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@v2.16

# pip only
RUN pip install ray[default]==2.54.0 omegaconf==2.3.1 hydra-core==1.3.4
RUN pip install --ignore-installed mlflow==3.14.0
RUN pip install wandb==0.28.0 weave==0.53.1
RUN pip install nvidia-resiliency-ext==0.6.0

# git
RUN git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git && cd Megatron-Bridge && git checkout 8a3462265 && pip install --no-deps --no-build-isolation -e .
RUN git clone https://github.com/NVIDIA/Megatron-LM.git && cd Megatron-LM && git checkout 43e45e130 && pip install --no-deps --no-build-isolation -e .

RUN pip install mini-swe-agent==2.4.5 litellm==1.80.0
RUN ARCH="$(uname -m)" && case "$ARCH" in \
      x86_64)  ARCH=amd64 ;; \
      aarch64) ARCH=arm64 ;; \
      *) echo "unsupported arch: $ARCH" && exit 1 ;; \
    esac && \
    curl -fsSLO "https://dl.k8s.io/release/v1.28.0/bin/linux/${ARCH}/kubectl" && \
    chmod +x kubectl && mv kubectl /usr/local/bin/kubectl

# for dsv4
RUN pip install --no-build-isolation git+https://github.com/Dao-AILab/fast-hadamard-transform.git@v1.1.0.post2
RUN pip install nvidia-cuda-cccl==13.3.3.4.1
RUN CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include/cccl pip install --no-build-isolation git+https://github.com/deepseek-ai/FlashMLA@nv_dev

# patch
COPY patch/sglang-0.5.16.patch /tmp/sglang-patch.patch
RUN cd /sgl-workspace/sglang && git apply /tmp/sglang-patch.patch && rm /tmp/sglang-patch.patch

# loongsage
RUN git clone https://github.com/baidu-baige/LoongSage.git && cd LoongSage && pip install --no-deps --no-build-isolation -e .
