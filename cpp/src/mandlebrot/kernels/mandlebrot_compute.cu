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

template <class T>
__device__ inline int CalcMandelbrot(const T xPos, const T yPos, int crunch) {
  T x, y, xx, yy;
  y = 0;
  x = 0;
  yy = 0;
  xx = 0;

  while (--crunch && (xx + yy < T(4.0))) {
    y = x * y * T(2.0) + yPos;
    x = xx - yy + xPos;
    yy = y * y;
    xx = x * x;
  }

  return crunch;
}

template __device__ int CalcMandelbrot<float>(float, float, int);
