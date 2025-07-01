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

#pragma once

#include "AlgorithmPlanner.h"
#include "MakeFragmentKey.h"

// https://github.com/NVIDIA/cccl/issues/5032
struct SumPlanner : AlgorithmPlanner {
  SumPlanner() : AlgorithmPlanner("sum") {}

  template <typename... Args>
  void iteration(LaunchType launch) {
    auto key = make_fragment_key<Args...>();
    return this->save_iteration(launch, key);
  }

  template <typename... Args>
  void compute() {
    auto key = make_fragment_key<Args...>();
    return this->save_compute(key);
  }

};
