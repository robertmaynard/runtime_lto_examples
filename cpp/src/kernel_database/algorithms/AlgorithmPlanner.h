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

#include <vector>
#include <string>

#include "AlgorithmLauncher.h"

struct FragmentEntry;

struct AlgorithmPlanner {

  AlgorithmPlanner(std::string const& n) : name(n){}

  void save_iteration(LaunchType launch, std::vector<std::string> const& params);
  void save_compute(std::vector<std::string> const& params);

  AlgorithmLauncher build();

  std::string name;
  LaunchType launch_type;
  std::vector<FragmentEntry*> fragments;
};
