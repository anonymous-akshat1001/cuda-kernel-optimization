#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                  \
  do {                                                                                    \
    cudaError_t error__ = (call);                                                         \
    if (error__ != cudaSuccess) {                                                         \
      std::ostringstream message__;                                                       \
      message__ << #call << " failed at " << __FILE__ << ":" << __LINE__ << ": "        \
                << cudaGetErrorString(error__);                                          \
      throw std::runtime_error(message__.str());                                          \
    }                                                                                     \
  } while (0)

constexpr int kTile = 16;

struct Row {
  std::string workload;
  int size;
  std::string implementation;
  double time_ms;
  double gflops;
  bool correct;
  int repeats;
  std::string device;
};

struct Options {
  std::vector<int> sizes{128, 256, 512};
  int vector_elements = 1 << 24;
  int repeats = 20;
  std::string output = "results/benchmark_results.csv";
  bool skip_cpu = false;
};

static std::vector<int> parse_sizes(const std::string& text) {
  std::vector<int> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const int value = std::stoi(token);
    if (value <= 0) throw std::runtime_error("Matrix dimensions must be positive.");
    values.push_back(value);
  }
  if (values.empty()) throw std::runtime_error("--sizes needs one or more values.");
  return values;
}

static Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto value = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("Missing value after " + arg);
      return argv[i];
    };
    if (arg == "--sizes") options.sizes = parse_sizes(value());
    else if (arg == "--vector-elements") options.vector_elements = std::stoi(value());
    else if (arg == "--repeats") options.repeats = std::stoi(value());
    else if (arg == "--output") options.output = value();
    else if (arg == "--skip-cpu") options.skip_cpu = true;
    else if (arg == "--help") {
      std::cout << "Usage: cuda_bench [--sizes 128,256,512] [--vector-elements N] "
                   "[--repeats N] [--output path] [--skip-cpu]\\n";
      std::exit(0);
    } else throw std::runtime_error("Unknown option: " + arg);
  }
  if (options.vector_elements <= 0 || options.repeats <= 0)
    throw std::runtime_error("--vector-elements and --repeats must be positive.");
  return options;
}

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) c[index] = a[index] + b[index];
}

__global__ void matmul_naive_kernel(const float* a, const float* b, float* c, int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < n && col < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) sum += a[row * n + k] * b[k * n + col];
    c[row * n + col] = sum;
  }
}

// Each block cooperatively caches a kTile-by-kTile patch of both inputs.
// This removes repeated global-memory loads used by nearby output elements.
__global__ void matmul_tiled_kernel(const float* a, const float* b, float* c, int n) {
  __shared__ float tile_a[kTile][kTile];
  __shared__ float tile_b[kTile][kTile];
  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;
  float sum = 0.0f;
  for (int tile = 0; tile < (n + kTile - 1) / kTile; ++tile) {
    const int a_col = tile * kTile + threadIdx.x;
    const int b_row = tile * kTile + threadIdx.y;
    tile_a[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();
    for (int k = 0; k < kTile; ++k) sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
    __syncthreads();
  }
  if (row < n && col < n) c[row * n + col] = sum;
}

static void fill_random(std::vector<float>& values, std::mt19937& generator) {
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  for (float& value : values) value = distribution(generator);
}

static void vector_add_cpu(const std::vector<float>& a, const std::vector<float>& b,
                           std::vector<float>& c) {
  for (size_t i = 0; i < a.size(); ++i) c[i] = a[i] + b[i];
}

static void matmul_cpu(const std::vector<float>& a, const std::vector<float>& b,
                       std::vector<float>& c, int n) {
  std::fill(c.begin(), c.end(), 0.0f);
  // i-k-j order gives the CPU reference reasonable cache behavior without changing math.
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < n; ++k) {
      const float aik = a[i * n + k];
      for (int j = 0; j < n; ++j) c[i * n + j] += aik * b[k * n + j];
    }
}

static bool close_enough(const std::vector<float>& actual, const std::vector<float>& expected,
                         float atol = 1e-3f, float rtol = 1e-3f) {
  if (actual.size() != expected.size()) return false;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float difference = std::fabs(actual[i] - expected[i]);
    if (difference > atol + rtol * std::fabs(expected[i])) {
      std::cerr << "Mismatch at " << i << ": got " << actual[i]
                << ", expected " << expected[i] << "\\n";
      return false;
    }
  }
  return true;
}

template <typename Launch>
static float time_gpu_ms(Launch launch, int repeats) {
  // One warm-up removes lazy setup effects from the timing average.
  launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeats; ++i) launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / repeats;
}

template <typename Work>
static double time_cpu_ms(Work work, int repeats) {
  work(); // warm-up
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeats; ++i) work();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() / repeats;
}

static double gflops_for_matmul(int n, double time_ms) {
  return (2.0 * n * n * n) / (time_ms * 1.0e6);
}

static void write_csv(const std::vector<Row>& rows, const std::string& path) {
  std::ofstream file(path);
  if (!file) throw std::runtime_error("Cannot open output file: " + path);
  file << "workload,size,implementation,time_ms,gflops,correct,repeats,device\\n";
  file << std::fixed << std::setprecision(6);
  for (const Row& row : rows)
    file << row.workload << ',' << row.size << ',' << row.implementation << ','
         << row.time_ms << ',' << row.gflops << ',' << (row.correct ? "PASS" : "FAIL") << ','
         << row.repeats << ',' << '"' << row.device << "\"\\n";
}

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) throw std::runtime_error("No CUDA-capable GPU detected.");
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    const std::string device = properties.name;
    std::cout << "GPU: " << device << " (compute capability " << properties.major << '.'
              << properties.minor << ")\\n";

    std::mt19937 generator(20261013); // fixed seed makes input data reproducible
    std::vector<Row> rows;

    // Vector addition: CPU reference + one CUDA kernel.
    const int vn = options.vector_elements;
    std::vector<float> va(vn), vb(vn), vcpu(vn), vgpu(vn);
    fill_random(va, generator); fill_random(vb, generator);
    float *dva = nullptr, *dvb = nullptr, *dvc = nullptr;
    CUDA_CHECK(cudaMalloc(&dva, vn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dvb, vn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dvc, vn * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dva, va.data(), vn * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvb, vb.data(), vn * sizeof(float), cudaMemcpyHostToDevice));
    const int vector_blocks = (vn + 255) / 256;
    const auto vector_launch = [&] { vector_add_kernel<<<vector_blocks, 256>>>(dva, dvb, dvc, vn); };
    const float vector_gpu_ms = time_gpu_ms(vector_launch, options.repeats);
    CUDA_CHECK(cudaMemcpy(vgpu.data(), dvc, vn * sizeof(float), cudaMemcpyDeviceToHost));
    vector_add_cpu(va, vb, vcpu);
    const bool vector_ok = close_enough(vgpu, vcpu, 1e-6f, 1e-6f);
    rows.push_back({"vector_add", vn, "cuda_vector_add", vector_gpu_ms, 0.0, vector_ok,
                    options.repeats, device});
    if (!options.skip_cpu) {
      const double ms = time_cpu_ms([&] { vector_add_cpu(va, vb, vcpu); }, options.repeats);
      rows.push_back({"vector_add", vn, "cpu_reference", ms, 0.0, true, options.repeats, device});
    }
    CUDA_CHECK(cudaFree(dva)); CUDA_CHECK(cudaFree(dvb)); CUDA_CHECK(cudaFree(dvc));
    std::cout << "vector_add: " << (vector_ok ? "PASS" : "FAIL") << "\\n";

    for (const int n : options.sizes) {
      const size_t elements = static_cast<size_t>(n) * n;
      std::vector<float> a(elements), b(elements), cpu(elements), naive(elements), tiled(elements);
      fill_random(a, generator); fill_random(b, generator);
      float *da = nullptr, *db = nullptr, *dc = nullptr;
      CUDA_CHECK(cudaMalloc(&da, elements * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&db, elements * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&dc, elements * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(da, a.data(), elements * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(db, b.data(), elements * sizeof(float), cudaMemcpyHostToDevice));
      const dim3 block(kTile, kTile);
      const dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
      const auto naive_launch = [&] { matmul_naive_kernel<<<grid, block>>>(da, db, dc, n); };
      const auto tiled_launch = [&] { matmul_tiled_kernel<<<grid, block>>>(da, db, dc, n); };

      // Compute the oracle once, outside GPU timing and correctness comparison.
      matmul_cpu(a, b, cpu, n);
      const float naive_ms = time_gpu_ms(naive_launch, options.repeats);
      CUDA_CHECK(cudaMemcpy(naive.data(), dc, elements * sizeof(float), cudaMemcpyDeviceToHost));
      const bool naive_ok = close_enough(naive, cpu);
      const float tiled_ms = time_gpu_ms(tiled_launch, options.repeats);
      CUDA_CHECK(cudaMemcpy(tiled.data(), dc, elements * sizeof(float), cudaMemcpyDeviceToHost));
      const bool tiled_ok = close_enough(tiled, cpu);
      rows.push_back({"matrix_multiply", n, "cuda_naive", naive_ms,
                      gflops_for_matmul(n, naive_ms), naive_ok, options.repeats, device});
      rows.push_back({"matrix_multiply", n, "cuda_tiled_shared_memory", tiled_ms,
                      gflops_for_matmul(n, tiled_ms), tiled_ok, options.repeats, device});
      if (!options.skip_cpu) {
        const double cpu_ms = time_cpu_ms([&] { matmul_cpu(a, b, cpu, n); }, options.repeats);
        rows.push_back({"matrix_multiply", n, "cpu_reference", cpu_ms,
                        gflops_for_matmul(n, cpu_ms), true, options.repeats, device});
      }
      CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
      std::cout << "matrix " << n << 'x' << n << ": naive=" << (naive_ok ? "PASS" : "FAIL")
                << ", tiled=" << (tiled_ok ? "PASS" : "FAIL") << "\\n";
    }
    write_csv(rows, options.output);
    const bool all_correct = std::all_of(rows.begin(), rows.end(),
                                         [](const Row& row) { return row.correct; });
    std::cout << "Wrote " << options.output << "\\n";
    return all_correct ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << "\\n";
    return 1;
  }
}
