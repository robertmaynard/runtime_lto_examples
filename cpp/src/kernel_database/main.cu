#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "algorithms/SaxpyPlanner.h"
#include "algorithms/SumPlanner.h"

int main(int, char**) {

    //Tasks:
    /*
    [x] Unify ComputeFragmentEntry and LaunchKernelEntry
    [x] Refactor the Launchkerneldatabase to be
        part of AlgorithmFragmentDatabase
    [x] Stream Execution shouldn't be bound at build time
    [ ] Validate NVRTC support for algorithms
    [ ] Support PASSTHROUGH launch type
    [ ] Better caching of the AlgorithmPlanner compile arguments
    */

    std::int64_t array_size = 0;
    std::int64_t shared_mem = 0;
    cudaStream_t stream;
    cudaStreamCreate( &stream );

    SaxpyPlanner saxpy;
    saxpy.setup<float*, float*, float*>(LaunchType::GRID_1D);
    auto saxpy_launcher = saxpy.build();

    SumPlanner sum;
    sum.setup<float*, float*>(LaunchType::GRID_1D);
    auto sum_launcher = sum.build();

    //Should actual generate some input data
    float* x = nullptr;
    float* y = nullptr;
    float* output = nullptr;
    saxpy_launcher.exec_info(stream, shared_mem);
    saxpy_launcher(array_size, x, y, output);


    //Should actual generate some input data
    sum_launcher.exec_info(stream, shared_mem);
    sum_launcher(array_size, x, output);

    return 0;
}
