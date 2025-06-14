#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "LaunchKernelDatabase.h"
#include "KernelPlanner.h"

int main(int, char**) {

    auto& kernelDB = build_launch_kernel_database();
    //what we need
    //
    // [x] fatbin to cpp file build-infra
    // [x] registration from fatbin to LKD
    // [x] nvrtc hookup to LKD
    //
    if (!kernelDB.has<float*, float, int>()) {
      kernelDB.add<float*, float, int>();
    }

    // auto planner = KernelPlanner{kernelDB};
    // planner.iteration_pattern<float*, float, int>();
    // planner.add_algorithm_fragment(saxpy...);
    // planner.add_algorithm_fragment(saxpy...);
    // planner.add_algorithm_fragment(saxpy...);
    // auto launcher = planner.build();

    // launcher.run(float*, int, float*)

    return 0;
}
