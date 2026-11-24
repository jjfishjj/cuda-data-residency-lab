#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/utsname.h>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e)); } while (0)

__global__ void add_kernel(const float* a, const float* b, float* c, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

static std::string json_escape(const std::string& value) {
  std::string out;
  for (char ch : value) {
    if (ch == '"' || ch == '\\') out += '\\';
    if (ch == '\n') out += "\\n"; else if (ch == '\r') out += "\\r"; else out += ch;
  }
  return out;
}

static std::string command_output(const char* command) {
  std::string result;
  FILE* pipe = popen(command, "r");
  if (!pipe) return result;
  char buffer[256];
  while (fgets(buffer, sizeof(buffer), pipe)) result += buffer;
  pclose(pipe);
  while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) result.pop_back();
  return result;
}

struct Stats { double median, mean, stdev, min, max; };
static Stats stats(std::vector<double> values) {
  if (values.empty()) return {0, 0, 0, 0, 0};
  std::sort(values.begin(), values.end());
  double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  double variance = 0;
  for (double v : values) variance += (v - mean) * (v - mean);
  variance /= values.size();
  double median = values.size() % 2 ? values[values.size() / 2] : (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2;
  return {median, mean, std::sqrt(variance), values.front(), values.back()};
}

static void print_number(double value) { std::cout << std::setprecision(10) << value; }
static double seconds_since(const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static void print_measurement(const std::string& case_id, const std::string& backend, size_t n,
                              int warmups, int iterations, const std::vector<double>& samples, const std::string& notes) {
  Stats s = stats(samples);
  std::cout << "{\"case_id\":\"" << case_id << "\",\"backend\":\"" << backend
            << "\",\"status\":\"completed\",\"unit\":\"seconds\",\"input_rows\":" << n
            << ",\"output_rows\":" << n << ",\"warmup_iterations\":" << warmups << ",\"iterations_per_sample\":" << iterations << ",\"samples\":[";
  for (size_t i = 0; i < samples.size(); ++i) { if (i) std::cout << ','; print_number(samples[i]); }
  std::cout << "],\"statistics\":{\"median\":"; print_number(s.median);
  std::cout << ",\"mean\":"; print_number(s.mean); std::cout << ",\"stdev\":"; print_number(s.stdev);
  std::cout << ",\"min\":"; print_number(s.min); std::cout << ",\"max\":"; print_number(s.max);
  std::cout << "},\"notes\":\"" << json_escape(notes) << "\"}";
}

int main(int argc, char** argv) {
  try {
    const size_t n = argc > 1 ? std::stoull(argv[1]) : (1ULL << 24);
    const int iterations = argc > 2 ? std::stoi(argv[2]) : 20;
    const int repetitions = argc > 3 ? std::stoi(argv[3]) : 5;
    const int warmups = argc > 4 ? std::stoi(argv[4]) : 1;
    if (n == 0 || iterations <= 0 || repetitions <= 0 || warmups < 0) throw std::invalid_argument("n, iterations, repetitions must be positive; warmups cannot be negative");

    int device = 0, runtime_version = 0, driver_api_version = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_api_version));
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const size_t bytes = n * sizeof(float);
    float *a = nullptr, *b = nullptr, *c = nullptr, *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMallocHost(&a, bytes)); CUDA_CHECK(cudaMallocHost(&b, bytes)); CUDA_CHECK(cudaMallocHost(&c, bytes));
    CUDA_CHECK(cudaMalloc(&da, bytes)); CUDA_CHECK(cudaMalloc(&db, bytes)); CUDA_CHECK(cudaMalloc(&dc, bytes));
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) { a[i] = distribution(rng); b[i] = distribution(rng); }
    const dim3 grid((n + 255) / 256), block(256);
    std::vector<double> baseline_transfer, baseline_kernel, resident_total, resident_kernel_loop_wall;

    auto run_kernel = [&]() { add_kernel<<<grid, block>>>(da, db, dc, n); CUDA_CHECK(cudaGetLastError()); };
    auto run_baseline = [&]() {
      for (int it = 0; it < iterations; ++it) {
        CUDA_CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
        run_kernel();
        CUDA_CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));
      }
    };
    auto run_resident = [&]() { for (int it = 0; it < iterations; ++it) run_kernel(); };
    auto validate_output = [&]() {
      CUDA_CHECK(cudaDeviceSynchronize());
      for (size_t i = 0; i < n; ++i) {
        if (std::fabs(c[i] - (a[i] + b[i])) > 1e-5f) throw std::runtime_error("result validation failed");
      }
    };

    for (int w = 0; w < warmups; ++w) { CUDA_CHECK(cudaDeviceSynchronize()); run_baseline(); CUDA_CHECK(cudaDeviceSynchronize()); }
    for (int rep = 0; rep < repetitions; ++rep) {
      CUDA_CHECK(cudaDeviceSynchronize());
      cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
      CUDA_CHECK(cudaEventRecord(start));
      auto wall_start = std::chrono::steady_clock::now();
      for (int it = 0; it < iterations; ++it) {
        CUDA_CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
        run_kernel(); CUDA_CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));
      }
      CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
      baseline_transfer.push_back(seconds_since(wall_start));
      float ignored = 0; CUDA_CHECK(cudaEventElapsedTime(&ignored, start, stop));
      CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
      // Kernel-only timing uses a separate synchronized event interval with no host copies.
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); CUDA_CHECK(cudaEventRecord(start));
      for (int it = 0; it < iterations; ++it) run_kernel();
      CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      baseline_kernel.push_back(ms / 1000.0); CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    }
    // Validate baseline's final D2H output before the resident path overwrites device buffers.
    validate_output();

    // Resident warmups establish a synchronized device-resident state. They are not samples.
    CUDA_CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int w = 0; w < warmups; ++w) { run_resident(); CUDA_CHECK(cudaDeviceSynchronize()); }
    for (int rep = 0; rep < repetitions; ++rep) {
      // Full resident end-to-end sample: one H2D (both inputs), all kernel iterations, one D2H.
      // The resident kernel loop itself remains free of H2D/D2H operations.
      CUDA_CHECK(cudaDeviceSynchronize());
      auto total_start = std::chrono::steady_clock::now();
      CUDA_CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
      run_resident();
      CUDA_CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());
      resident_total.push_back(seconds_since(total_start));

      // Kernel-loop wall time: synchronized wall-clock interval around kernels only.
      CUDA_CHECK(cudaDeviceSynchronize());
      auto loop_start = std::chrono::steady_clock::now();
      run_resident();
      CUDA_CHECK(cudaDeviceSynchronize());
      resident_kernel_loop_wall.push_back(seconds_since(loop_start));
    }
    // This D2H result was checked before resident setup, and the resident result is checked here too.
    validate_output();

    utsname host{}; uname(&host); const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{}; gmtime_r(&now, &utc); char timestamp[32]; strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &utc);
    std::ostringstream run_id; run_id << timestamp << "-" << std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::string commit = command_output("git rev-parse HEAD 2>/dev/null");
    const std::string dirty = command_output("git status --porcelain 2>/dev/null").empty() ? "false" : "true";
    const std::string python = command_output("python3 --version 2>/dev/null");
    const std::string nvidia_driver = command_output("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1");
    std::cout << "{\"schema_version\":\"1.0.0\",\"benchmark_id\":\"cuda-data-residency-vector-add\",\"run_id\":\"" << json_escape(run_id.str()) << "\",\"timestamp_utc\":\"" << timestamp << "\",\"git\":{\"commit\":" << (commit.empty() ? "null" : "\"" + json_escape(commit) + "\"") << ",\"dirty\":" << dirty << "},\"environment\":{\"os\":\"" << json_escape(host.sysname) << "\",\"os_release\":\"" << json_escape(host.release) << "\",\"architecture\":\"" << json_escape(host.machine) << "\",\"python\":\"" << json_escape(python) << "\",\"cuda_runtime\":\"" << runtime_version / 1000 << "." << (runtime_version % 1000) / 10 << "\",\"cuda_driver_api\":\"" << driver_api_version / 1000 << "." << (driver_api_version % 1000) / 10 << "\",\"cuda_driver\":" << (nvidia_driver.empty() ? "null" : "\"" + json_escape(nvidia_driver) + "\"") << ",\"software\":{\"nvcc\":\"" << json_escape(command_output("nvcc --version 2>/dev/null | tail -1")) << "\"},\"gpus\":[{\"index\":0,\"name\":\"" << json_escape(prop.name) << "\",\"compute_capability\":\"" << prop.major << "." << prop.minor << "\",\"memory_bytes\":" << prop.totalGlobalMem << "}]},\"parameters\":{\"elements\":" << n << ",\"iterations\":" << iterations << ",\"repetitions\":" << repetitions << ",\"warmup_iterations\":" << warmups << ",\"seed\":12345,\"operation\":\"float vector addition\",\"block_size\":256},\"measurements\":[";
    print_measurement("baseline.transfer-inclusive", "cuda-transfer-per-iteration", n, warmups, iterations, baseline_transfer, "Each sample is one synchronized wall-clock run of all iterations; every iteration includes 2 H2D copies, one kernel, and 1 D2H copy."); std::cout << ',';
    print_measurement("baseline.kernel-only", "cuda-transfer-per-iteration", n, warmups, iterations, baseline_kernel, "Each sample times all kernel iterations only; copies are excluded and GPU is synchronized before and after."); std::cout << ',';
    print_measurement("resident.total-inclusive", "cuda-gpu-resident", n, warmups, iterations, resident_total, "Each sample includes exactly one H2D of both inputs, all iterations of the resident kernel loop, and one final D2H; synchronized wall-clock timing."); std::cout << ',';
    print_measurement("resident.kernel-loop-wall", "cuda-gpu-resident", n, warmups, iterations, resident_kernel_loop_wall, "Each sample is a synchronized wall-clock interval for all resident kernel iterations only; the loop contains no H2D/D2H. Resident warmup runs are excluded."); std::cout << "]}\n";
    cudaFree(da); cudaFree(db); cudaFree(dc); cudaFreeHost(a); cudaFreeHost(b); cudaFreeHost(c);
    return 0;
  } catch (const std::exception& error) { std::cerr << "benchmark unavailable or failed: " << error.what() << "\n"; return 2; }
}
