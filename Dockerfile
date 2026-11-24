ARG CUDA_IMAGE=nvidia/cuda:13.4.3-devel-ubuntu22.04
FROM ${CUDA_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends ninja-build python3 python3-pip git && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir cmake==3.31.10
WORKDIR /workspace
COPY CMakeLists.txt ./
COPY src/ src/
COPY benchmark-manifest.json ./
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
ENTRYPOINT ["/workspace/build/vector_add"]
