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

// OpenGL Graphics includes
#include <kernels/mandlebrot_grid_stride.cu>

inline int iDivUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}
void RunMandelbrot(uchar4      *dst,
                    const int    imageW,
                    const int    imageH,
                    const int    crunch,
                    const double xOff,
                    const double yOff,
                    const double scale,
                    const uchar4 colors,
                    const int    frame,
                    const int    numSMs)
{
  constexpr int BLOCKDIM_X = 16;
  constexpr int BLOCKDIM_Y = 16;
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
  dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));
  Mandelbrot<float><<<numSMs, threads>>>(
      dst, imageW, imageH, crunch, (float)xOff, (float)yOff, (float)scale,
      colors, frame, grid.x, grid.x * grid.y);
}
