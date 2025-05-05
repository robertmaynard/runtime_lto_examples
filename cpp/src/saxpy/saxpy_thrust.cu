/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <random>
#include <vector>

#include <thrust/transform.h>

#include "saxpy_setup.h"

int main(int, char**) {
  rmm::cuda_stream stream{};
  saxpy_memory saxpy{stream};

  const auto n = static_cast<std::int64_t>(saxpy.x->size());

  std::cout << "Launch thrust::transform fast saxpy with " << n << " elements\n";
  thrust::transform(rmm::exec_policy(stream), saxpy.x->begin(), saxpy.x->end(),
                    saxpy.y->begin(), saxpy.y->begin(),
                    [] __device__(float x, float y) { return 2.0f * x + y; });


  std::cout << "Launch thrust::transform slow saxpy pass 1 with " << n << " elements\n";
  thrust::transform(rmm::exec_policy(stream), saxpy.x->begin(), saxpy.x->end(),
                    saxpy.x->begin(),
                    [] __device__(float x) { return 2.0f * x; });
  std::cout << "Launch thrust::transform slow saxpy pass 2 with " << n << " elements\n";
  thrust::transform(rmm::exec_policy(stream), saxpy.x->begin(), saxpy.x->end(),
                    saxpy.y->begin(), saxpy.y->begin(),
                    [] __device__(float x, float y) { return x + y; });

  std::vector<float> host_y;
  host_y.resize(saxpy.y->size());

  cudaMemcpyAsync(host_y.data(), saxpy.y->begin(),
                  saxpy.y->size() * sizeof(float), cudaMemcpyDefault,
                  stream.value());
  cudaStreamSynchronize(stream.value());
  return 0;
}
