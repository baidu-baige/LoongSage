FROM lmsysorg/sglang:v0.5.23

WORKDIR /sgl-workspace

# env
ARG PIP_NO_CACHE_DIR=1
RUN pip config set global.break-system-packages true

# compile
RUN MAX_JOBS=128 pip install -v --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git
RUN MAX_JOBS=128 pip install --no-build-isolation causal-conv1d==1.6.1
RUN MAX_JOBS=128 NVTE_FRAMEWORK=pytorch NVTE_BUILD_THREADS_PER_JOB=4 pip install -v --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@v2.16

# pip only
RUN pip install ray[default]==2.55.1 hydra-core==1.3.8
RUN pip install mlflow-skinny==3.16.0 wandb==0.28.0
RUN pip install nvidia-resiliency-ext==0.6.0

# git
RUN git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git && cd Megatron-Bridge && git checkout 17530a274 && pip install --no-deps --no-build-isolation -e .
RUN git clone https://github.com/NVIDIA/Megatron-LM.git && cd Megatron-LM && git checkout 2ab6361e4 && pip install --no-deps --no-build-isolation -e .

# sglang main 28053272d9 + sgl-project/sglang#51926 (Inkling-2-Mini, head 7ee83cfebc)
RUN rm -rf sglang && git clone https://github.com/sgl-project/sglang.git && cd sglang && git checkout 28053272d9 \
    && git fetch origin pull/51926/head && git -c user.name=loongsage -c user.email=loongsage@localhost merge --no-ff --no-edit 7ee83cfebc \
    && cd python && pip install --no-deps --no-build-isolation -e .

# kernels and deps pinned by sglang 28053272d9 + #51926 (python/pyproject.toml)
RUN pip install sglang-kernel==0.4.11 sgl-deep-gemm==0.2.9 sgl-flash-mla==1.0.5 sgl-deep-ep==0.1.3 sgl-fa3==0.1.0 \
    tml-renderers==0.4.2 humming-kernels==0.1.18 "flashinfer_python[cu13]==0.7.4"
RUN pip install flashinfer-cubin==0.7.4 --index-url https://flashinfer.ai/whl \
    && pip install flashinfer-jit-cache==0.7.4 --index-url https://flashinfer.ai/whl/cu130
# Bridge's Inkling2 needs Transformers' native inkling2 configs (>=5.23, <5.24)
RUN pip install transformers==5.23.0

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

# for qwen3.8
RUN pip install flash-linear-attention==0.5.2
RUN MAX_JOBS=16 pip install -v --no-build-isolation "flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention@main#subdirectory=hopper"

# for qwen-vl
RUN pip install qwen-vl-utils==0.0.14

# loongsage
RUN git clone https://github.com/baidu-baige/LoongSage.git && cd LoongSage && pip install --no-deps --no-build-isolation -e .

# patch
RUN cd /sgl-workspace/sglang && git apply ../LoongSage/docker/patch/sglang-28053272d9-pr51926.patch
RUN cd /sgl-workspace/Megatron-LM && git apply ../LoongSage/docker/patch/dsv41/megatron-core.patch
RUN cd /sgl-workspace/Megatron-Bridge && git apply ../LoongSage/docker/patch/dsv41/megatron-bridge.patch
RUN cd /sgl-workspace/Megatron-Bridge && git apply ../LoongSage/docker/patch/inkling2/megatron-bridge.patch
