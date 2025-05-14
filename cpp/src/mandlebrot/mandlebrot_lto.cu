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
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/helper.h>
#include <GL/freeglut.h>

// CUDA utilities and system includes
#include <cuda_gl_interop.h>
#include <cuda.h>

#include <memory>
#include <vector>

#include "cuda_wrapper.hpp"

#include <cub/detail/launcher/cuda_driver.cuh>

CUlibrary load_fatbins(CUdevice, std::vector<std::string>);


// OpenGL PBO and texture "names"
GLuint                       gl_PBO, gl_Tex, gl_Shader;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange

// Source image on the host side
std::vector<uchar4> h_Src;

// Destination image on the GPU side
uchar4 *d_dst = NULL;

// Original image width and height
int imageW = 4096, imageH = 2160;

// Starting iteration limit
int crunch = 4194304;

// Starting position and scale
double xOff  = -1.27427518;
double yOff  = 0.21999864;
double scale = 3.458208e-01;
int pass = 0;

// Starting color multipliers and random seed
int colorSeed = 0;
uchar4 colors;

int numSMs = 0;  // number of multiprocessors

#define BUFFER_DATA(i) ((char *)0 + i)


inline int iDivUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

CUlibrary cuda_lib;
CUkernel kernel;

void load_mandlebrot_fatbins(CUdevice& cuda_device) {
    std::cout << "Start loading Mandlebrot LTO FATBINS \n";
    cuda_lib = load_fatbins(
        cuda_device, std::vector<std::string>{"mandlebrot_compute.fatbin",
                                              "mandlebrot_grid_stride.fatbin"});
    std::cout << "Finished loading \n";

    // The trick way is to use cuLibraryEnumerateKernel and cuKernelGetName
    // Since our modules only have 1 kernel we can fetch the kernel without
    // ever knowing the mangled name
    //
    //  There should be a better way, but I need to do some research
    unsigned int count = 0;
    cuLibraryGetKernelCount(&count, cuda_lib);
    std::cout << " count " << count << std::endl;

    std::unique_ptr<CUkernel[]> kernels_{new CUkernel[count]};
    cuLibraryEnumerateKernels(kernels_.get(), count, cuda_lib);
    for (unsigned int i = 0; i < count; ++i) {
      const char *result;
      cuKernelGetName(&result, kernels_[i]);
      std::cout << result << std::endl;
    }

    kernel =  kernels_[0];
}

void RunMandelbrot( uchar4      *dst,
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


  //Build up a launcher for kernels with the same grid, block, etc
  const dim3 launch_grid = { (unsigned int) numSMs, 1, 1};
  constexpr dim3 threads = {BLOCKDIM_X, BLOCKDIM_Y, 1};
  constexpr size_t shared_mem = 0;
  CUstream stream;
  DEMO_CUDA_TRY(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  cub::detail::CudaDriverLauncher launcher{launch_grid, threads, shared_mem, stream};

  dim3 kernel_iteration_space(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));
  launcher.doit(kernel, dst, imageW, imageH, crunch, (float) xOff, (float)yOff, (float)scale,
      colors, frame, kernel_iteration_space.x, kernel_iteration_space.x * kernel_iteration_space.y);

  DEMO_CUDA_TRY(cuStreamSynchronize(stream));
  DEMO_CUDA_TRY(cuStreamDestroy(stream));
}

// render Mandelbrot image using CUDA or CPU
void renderImage()
{
  static const unsigned char samplePoints[128][2] = {
      {64, 64},  {0, 0},     {1, 63},    {63, 1},    {96, 32},  {97, 95},
      {36, 96},  {30, 31},   {95, 127},  {4, 97},    {33, 62},  {62, 33},
      {31, 126}, {67, 99},   {99, 65},   {2, 34},    {81, 49},  {19, 80},
      {113, 17}, {112, 112}, {80, 16},   {115, 81},  {46, 15},  {82, 79},
      {48, 78},  {16, 14},   {49, 113},  {114, 48},  {45, 45},  {18, 47},
      {20, 109}, {79, 115},  {65, 82},   {52, 94},   {15, 124}, {94, 111},
      {61, 18},  {47, 30},   {83, 100},  {98, 50},   {110, 2},  {117, 98},
      {50, 59},  {77, 35},   {3, 114},   {5, 77},    {17, 66},  {32, 13},
      {127, 20}, {34, 76},   {35, 110},  {100, 12},  {116, 67}, {66, 46},
      {14, 28},  {23, 93},   {102, 83},  {86, 61},   {44, 125}, {76, 3},
      {109, 36}, {6, 51},    {75, 89},   {91, 21},   {60, 117}, {29, 43},
      {119, 29}, {74, 70},   {126, 87},  {93, 75},   {71, 24},  {106, 102},
      {108, 58}, {89, 9},    {103, 23},  {72, 56},   {120, 8},  {88, 40},
      {11, 88},  {104, 120}, {57, 105},  {118, 122}, {53, 6},   {125, 44},
      {43, 68},  {58, 73},   {24, 22},   {22, 5},    {40, 86},  {122, 108},
      {87, 90},  {56, 42},   {70, 121},  {8, 7},     {37, 52},  {25, 55},
      {69, 11},  {10, 106},  {12, 38},   {26, 69},   {27, 116}, {38, 25},
      {59, 54},  {107, 72},  {121, 57},  {39, 37},   {73, 107}, {85, 123},
      {28, 103}, {123, 74},  {55, 85},   {101, 41},  {42, 104}, {84, 27},
      {111, 91}, {9, 19},    {21, 39},   {90, 53},   {41, 60},  {54, 26},
      {92, 119}, {51, 71},   {124, 101}, {68, 92},   {78, 10},  {13, 118},
      {7, 84},   {105, 4}};
  size_t num_bytes;
  cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
  cudaGraphicsResourceGetMappedPointer((void **)&d_dst, &num_bytes,
                                       cuda_pbo_resource);

  auto sampleIndex = pass & 127;
  float xs = (1.0f / 128.0f) * (0.5f + (float)samplePoints[sampleIndex][0]);
  float ys = (1.0f / 128.0f) * (0.5f + (float)samplePoints[sampleIndex][1]);

  // Get the pixel scale and offset
  double s = scale / (float)imageW;
  double x = (xs - (double)imageW * 0.5f) * s + xOff;
  double y = (ys - (double)imageH * 0.5f) * s + yOff;

  RunMandelbrot(d_dst, imageW, imageH, crunch, x, y, s, colors, pass++, numSMs);
  cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

// OpenGL display function
void displayFunc(void)
{
  // render the Mandelbrot image
  renderImage();

  // load texture from PBO
  glBindTexture(GL_TEXTURE_2D, gl_Tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA,
                  GL_UNSIGNED_BYTE, BUFFER_DATA(0));

  // fragment program is required to display floating point texture
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
  glEnable(GL_FRAGMENT_PROGRAM_ARB);
  glDisable(GL_DEPTH_TEST);

  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f);
  glVertex2f(0.0f, 0.0f);
  glTexCoord2f(1.0f, 0.0f);
  glVertex2f(1.0f, 0.0f);
  glTexCoord2f(1.0f, 1.0f);
  glVertex2f(1.0f, 1.0f);
  glTexCoord2f(0.0f, 1.0f);
  glVertex2f(0.0f, 1.0f);
  glEnd();

  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_FRAGMENT_PROGRAM_ARB);

  glutSwapBuffers();

}

void cleanup()
{
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glDeleteBuffers(1, &gl_PBO);
    glDeleteTextures(1, &gl_Tex);
    glDeleteProgramsARB(1, &gl_Shader);
}

void timerEvent(int value)
{
    if (glutGetWindow()) {
        glutPostRedisplay();
        glutTimerFunc(10, timerEvent, 0);
    }
}

void initOpenGLBuffers(int w, int h)
{
    // delete old buffers
    if (gl_Tex) {
        glDeleteTextures(1, &gl_Tex);
        gl_Tex = 0;
    }

    if (gl_PBO) {
        // DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &gl_PBO);
        gl_PBO = 0;
    }

    // allocate new buffers
    h_Src.resize(w*h);

    std::printf("Creating GL texture...\n");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src.data());
    std::printf("Texture created.\n");

    std::printf("Creating PBO...\n");
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Src.data(), GL_STREAM_COPY);
    // While a PBO is registered to CUDA, it can't be used
    // as the destination for OpenGL drawing calls.
    // But in our particular case OpenGL is only used
    // to display the content of the PBO, specified by CUDA kernels,
    // so we need to register/unregister it only once.

    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, cudaGraphicsMapFlagsWriteDiscard);

    // load shader program
    // gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB);

    static const std::string shader_code{
        "!!ARBfp1.0\n"
        "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
        "END"};

    glGenProgramsARB(1, &gl_Shader);
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
    glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB,
                       (GLsizei)shader_code.size(),
                       (GLubyte *)shader_code.c_str());
}

void reshapeFunc(int w, int h)
{
    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    if (w != 0 && h != 0)  // Do not call when window is minimized that is when
                           // width && height == 0
    {
        initOpenGLBuffers(w, h);
    }

    imageW = w;
    imageH = h;
    pass   = 0;

    glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    CUdevice cuda_device;
    CUcontext cuda_context;
    cuInit(0);
    cuDeviceGet(&cuda_device, 0);
    cuCtxCreate(&cuda_context, 0, cuda_device);

    // Initialize all the information needed to render
    colors.w = 0;
    colors.x = 3;
    colors.y = 5;
    colors.z = 7;

    cuDeviceGetAttribute(
        &numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuda_device);
    std::printf("Data initialization done.\n");

    // Initialize OpenGL context first before the CUDA context is created.  This
    // is needed
    // to achieve optimal performance with OpenGL/CUDA interop.
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(imageW, imageH);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("Mandlebrot byhand");

    glutDisplayFunc(displayFunc);  // bind the callback function for rendering
    glutReshapeFunc(reshapeFunc);
    glutTimerFunc(10, timerEvent, 0);

    initOpenGLBuffers(imageW, imageH);

    std::printf("Compiling Mandlebrot kernel...\n");
    load_mandlebrot_fatbins(cuda_device);

    bool interactive = true;
    if (argc >= 2) {
        interactive = false;
    }

    if (interactive) {
        std::printf("Starting GLUT main loop...\n");
        crunch = 4096;
        glutCloseFunc(cleanup);
        glutMainLoop();
    } else {
        std::printf("Render 64 times for nvprof. Cranking the crunch level up. This will be slow\n");
        crunch = 1048576;
        for(int i=0; i < 64; ++i) {
            renderImage();
        }
    }

  return 0;
}
