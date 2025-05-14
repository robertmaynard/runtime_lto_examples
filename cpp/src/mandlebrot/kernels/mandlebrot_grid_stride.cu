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

template<typename T>
extern int __device__ CalcMandelbrot(T x, T y, int);

template <class T>
__global__ void Mandelbrot(uchar4 *dst,
                           const int imageW,
                           const int imageH,
                           const int crunch,
                           const T xOff,
                           const T yOff,
                           const T scale,
                           const uchar4 colors,
                           const int frame,
                           const int gridWidth,
                           const int numBlocks) {
  // loop until all blocks completed
  for (unsigned int blockIndex = blockIdx.x; blockIndex < numBlocks;
       blockIndex += gridDim.x) {
    unsigned int blockX = blockIndex % gridWidth;
    unsigned int blockY = blockIndex / gridWidth;

    // process this block
    const int ix = blockDim.x * blockX + threadIdx.x;
    const int iy = blockDim.y * blockY + threadIdx.y;

    if ((ix < imageW) && (iy < imageH)) {
      // Calculate the location
      const T xPos = (T)ix * scale + xOff;
      const T yPos = (T)iy * scale + yOff;

      // Calculate the Mandelbrot index for the current location
      int m = CalcMandelbrot<T>(xPos, yPos, crunch);
      m = m > 0 ? crunch - m : 0;

      // Convert the Mandelbrot index into a color
      uchar4 color;

      if (m) {
        color.x = m * colors.x;
        color.y = m * colors.y;
        color.z = m * colors.z;
      } else {
        color.x = 0;
        color.y = 0;
        color.z = 0;
      }

      // Output the pixel
      int pixel = imageW * iy + ix;

      int frame1 = frame + 1;
      int frame2 = frame1 / 2;
      dst[pixel].x = (dst[pixel].x * frame + color.x + frame2) / frame1;
      dst[pixel].y = (dst[pixel].y * frame + color.y + frame2) / frame1;
      dst[pixel].z = (dst[pixel].z * frame + color.z + frame2) / frame1;
    }
  }
}

template __global__ void Mandelbrot<float>(uchar4 *, int, int, int, float,
                                           float, float, uchar4, int, int, int);
