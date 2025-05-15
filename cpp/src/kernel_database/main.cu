#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include <LaunchKernelDatabase.h>
#include <KernelPlanner.h>

int main(int, char**) {

    auto kernelDB = LaunchKernelDatabase::build();

    if(! kernelDB.has<float*, float, int>) {
        kernelDB.add<float*,float,int>();
    }

    // auto planner = KernelPlanner{kernelDB};
    // planner.iteration_pattern<float*, int, float>();


    return 0;
}
