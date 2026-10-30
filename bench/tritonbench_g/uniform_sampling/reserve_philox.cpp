#include <ATen/core/Generator.h>
#include <torch/csrc/utils/pybind.h>
#include <limits>
#include <mutex>
#include <tuple>

// Use the same mutex as ATen random operations. No tensor allocation, CUDA
// launch, private generator layout, or guessed get_state() byte ABI is used.
static std::tuple<uint64_t, uint64_t> reserve_philox(
    at::Generator generator, int device_index, uint64_t block,
    uint64_t counter_count) {
  TORCH_CHECK(generator.defined(), "undefined generator");
  TORCH_CHECK(generator.device() == c10::Device(c10::kCUDA, device_index),
              "generator must belong to the output CUDA device");
  TORCH_CHECK(block == 512 || block == 1024, "invalid Philox block");
  TORCH_CHECK(counter_count > 0 && counter_count % block == 0,
              "counter reservation must contain complete blocks");
  std::lock_guard<std::mutex> guard(generator.mutex());
  const uint64_t seed = generator.current_seed();
  const uint64_t offset = generator.get_offset();
  const uint64_t skip = (block - offset % block) % block;
  constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
  TORCH_CHECK(skip <= maximum - offset, "Philox offset exhausted");
  const uint64_t start = offset + skip;
  TORCH_CHECK(counter_count <= maximum - start, "Philox offset exhausted");
  generator.set_offset(start + counter_count);
  return std::make_tuple(seed, start);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("reserve", &reserve_philox,
             pybind11::call_guard<pybind11::gil_scoped_release>());
}
