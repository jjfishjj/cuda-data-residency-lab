ARG CUDA_IMAGE=nvidia/cuda:13.4.3-devel-ubuntu24.04
FROM ${CUDA_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY CMakeLists.txt ./
COPY src/ src/
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
ENTRYPOINT ["/workspace/build/vector_add"]
