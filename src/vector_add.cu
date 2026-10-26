#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while (0)

__global__ void add_kernel(const float* a, const float* b, float* c, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
  const size_t n = argc > 1 ? std::stoull(argv[1]) : (1ULL << 24);
  const int iterations = argc > 2 ? std::stoi(argv[2]) : 20;
  const size_t bytes = n * sizeof(float);
  float *a, *b, *c, *da, *db, *dc;
  CUDA_CHECK(cudaMallocHost(&a, bytes)); CUDA_CHECK(cudaMallocHost(&b, bytes)); CUDA_CHECK(cudaMallocHost(&c, bytes));
  CUDA_CHECK(cudaMalloc(&da, bytes)); CUDA_CHECK(cudaMalloc(&db, bytes)); CUDA_CHECK(cudaMalloc(&dc, bytes));
  for (size_t i = 0; i < n; ++i) { a[i] = 1.0f; b[i] = 2.0f; }
  cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int it = 0; it < iterations; ++it) {
    CUDA_CHECK(cudaMemcpyAsync(da, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(db, b, bytes, cudaMemcpyHostToDevice));
    add_kernel<<<(n + 255) / 256, 256>>>(da, db, dc, n);
    CUDA_CHECK(cudaMemcpyAsync(c, dc, bytes, cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  if (c[n - 1] != 3.0f) throw std::runtime_error("result validation failed");
  std::cout << "{\"elements\":" << n << ",\"iterations\":" << iterations << ",\"elapsed_ms\":" << ms
            << ",\"avg_ms\":" << ms / iterations << "}\n";
  cudaEventDestroy(start); cudaEventDestroy(stop); cudaFree(da); cudaFree(db); cudaFree(dc);
  cudaFreeHost(a); cudaFreeHost(b); cudaFreeHost(c);
}
