
####################
##### Summary ######
####################

Compile Times:
| Implementation    | Compile Time |  IncComp Time |
| =========-------- | ------------ |  ------------ |
|       LTO         |   14.75sec   |     0.90sec   |
|       Thrust      |   34.72sec   |    34.72sec   |
|       ByHand      |   31.25sec   |    31.25sec   |


Binary Sizes:
| Implementation    | Sizes |
| =========-------- | ------- |
|       LTO         |  8672KB |
|       Thrust      | 63744KB |
|       ByHand      | 30968KB |


Kernel Runtimes:
| Implementation    | Runtime |
| =========-------- | ------- |
|       LTO         | 7.546ms |
|       Thrust      | 7.439ms |
|       ByHand      | 7.5233  |



# Clean compile times

## LTO
```
  Command being timed: "ninja -j1 saxpy_kernels saxpy"
  User time (seconds): 12.04
  System time (seconds): 2.71
  Percent of CPU this job got: 100%
  Elapsed (wall clock) time (h:mm:ss or m:ss): 0:14.75
```

## Thrust
```
  Command being timed: "ninja -j1 saxpy_thrust"
  User time (seconds): 29.06
  System time (seconds): 5.64
  Percent of CPU this job got: 99%
  Elapsed (wall clock) time (h:mm:ss or m:ss): 0:34.72
```

## By Hand

```
  Command being timed: "ninja -j1 saxpy_byhand"
  User time (seconds): 25.62
  System time (seconds): 5.62
  Percent of CPU this job got: 99%
  Elapsed (wall clock) time (h:mm:ss or m:ss): 0:31.25
```

# Incremental compile times for kernel change

## LTO
```
  Command being timed: "ninja -j1 saxpy_kernels saxpy"
  User time (seconds): 0.72
  System time (seconds): 0.17
  Percent of CPU this job got: 99%
  Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.90
```

## Thrust
```
  Command being timed: "ninja -j1 saxpy_thrust"
  User time (seconds): 29.06
  System time (seconds): 5.64
  Percent of CPU this job got: 99%
  Elapsed (wall clock) time (h:mm:ss or m:ss): 0:34.72

```

## By Hand
```
  Command being timed: "ninja -j1 saxpy_byhand"
  User time (seconds): 25.62
  System time (seconds): 5.62
  Percent of CPU this job got: 99%
  Elapsed (wall clock) time (h:mm:ss or m:ss): 0:31.25
```

# Binary Size

Binary size measurements are done by only looking at the CUBIN/LTO-IR size as
we are only interested in how much device code savings are possible.

Note: This presumes that the auto generated code to setup CUDART kernel launches
is noise

## LTO
```
  1920 May  5 08:21 saxpy_compute.fatbin
  1920 May  5 08:21 saxpy_compute_slow_1.fatbin
  1920 May  5 08:21 saxpy_compute_slow_2.fatbin
  2912 May  5 08:21 saxpy_grid_stride.fatbin
```

## Thrust

```
cuobjdump --extract-ptx all  ./saxpy_thrust
cuobjdump --extract-elf all  ./saxpy_thrust
fatbinary --compress-all --compress-mode=size \
  --image3=kind=elf,sm=70,file=./saxpy_thrust.1.sm_70.cubin \
  --image3=kind=elf,sm=75,file=./saxpy_thrust.2.sm_75.cubin \
  --image3=kind=elf,sm=80,file=./saxpy_thrust.3.sm_80.cubin \
  --image3=kind=elf,sm=86,file=./saxpy_thrust.4.sm_86.cubin \
  --image3=kind=elf,sm=90,file=./saxpy_thrust.5.sm_90.cubin \
  --image3=kind=elf,sm=100,file=./saxpy_thrust.6.sm_100.cubin \
  --image3=kind=elf,sm=120,file=./saxpy_thrust.7.sm_120.cubin \
  --image3=kind=ptx,sm=120,file=./saxpy_thrust.1.sm_120.ptx \
  --create="thrust.fatbin"
63744 May  5 09:09 thrust.fatbin
```

## By Hand

```
cuobjdump --extract-ptx all  ./saxpy_byhand
cuobjdump --extract-elf all  ./saxpy_byhand
fatbinary --compress-all --compress-mode=size \
  --image3=kind=elf,sm=70,file=./saxpy_byhand.1.sm_70.cubin \
  --image3=kind=elf,sm=75,file=./saxpy_byhand.2.sm_75.cubin \
  --image3=kind=elf,sm=80,file=./saxpy_byhand.3.sm_80.cubin \
  --image3=kind=elf,sm=86,file=./saxpy_byhand.4.sm_86.cubin \
  --image3=kind=elf,sm=90,file=./saxpy_byhand.5.sm_90.cubin \
  --image3=kind=elf,sm=100,file=./saxpy_byhand.6.sm_100.cubin \
  --image3=kind=elf,sm=120,file=./saxpy_byhand.7.sm_120.cubin \
  --image3=kind=ptx,sm=120,file=./saxpy_byhand.1.sm_120.ptx \
  --create="byhand.fatbin"
30968 May  5 09:32 byhand.fatbin
```

So total sizes are:
  LTO:     8672
  Thrust: 63744 ( 7.3x increase )
  ByHand: 30968 ( 3.5x increase )


# Kernel Runtime

## LTO

```
4.07%  7.5463ms         3  2.5154ms  1.9938ms  2.7765ms  saxpy
```

## Thrust

```
1.54%  2.7730ms         1  2.7730ms  2.7730ms  2.7730ms  _ZN3cub6detail9transform16transform_kernelINS1_10policy_hubILb1EN4cuda3std3__45tupleIJPfS8_EEEE10policy1000EiZ4mainEUlffE0_S8_JS8_S8_EEEvT0_iT1_T2_DpNS1_10kernel_argIT3_EE
1.54%  2.7723ms         1  2.7723ms  2.7723ms  2.7723ms  _ZN3cub6detail9transform16transform_kernelINS1_10policy_hubILb1EN4cuda3std3__45tupleIJPfS8_EEEE10policy1000EiZ4mainEUlffE_S8_JS8_S8_EEEvT0_iT1_T2_DpNS1_10kernel_argIT3_EE
1.05%  1.8937ms         1  1.8937ms  1.8937ms  1.8937ms  _ZN3cub6detail9transform16transform_kernelINS1_10policy_hubILb1EN4cuda3std3__45tupleIJPfEEEE10policy1000EiZ4mainEUlfE_S8_JS8_EEEvT0_iT1_T2_DpNS1_10kernel_argIT3_EE
```

## By Hand
```
1.38%  2.7767ms         1  2.7767ms  2.7767ms  2.7767ms  saxpy_pass_2()
1.38%  2.7744ms         1  2.7744ms  2.7744ms  2.7744ms  saxpy_fast()
0.98%  1.9722ms         1  1.9722ms  1.9722ms  1.9722ms  saxpy_pass_1()
```

Totals:
  LTO: 7.5463ms
  Thrust: 7.439ms
  ByHand: 7.5233ms


# Complete Runtime

## LTO

```
  Command being timed: "./saxpy"
  User time (seconds): 1.36
  System time (seconds): 2.12
  Percent of CPU this job got: 98%
  Elapsed (wall clock) time (h:mm:ss or m:ss): 0:03.54
```

## Thrust

```
  Command being timed: "./saxpy_thrust"
  User time (seconds): 1.35
  System time (seconds): 2.12
  Percent of CPU this job got: 98%
  Elapsed (wall clock) time (h:mm:ss or m:ss): 0:03.53
```

## By Hand
```
  Command being timed: "./bin/saxpy_byhand"
  User time (seconds): 1.38
  System time (seconds): 2.11
  Percent of CPU this job got: 98%
  Elapsed (wall clock) time (h:mm:ss or m:ss): 0:03.55
```
