
When we tune LTO compute kernels we see performance at the same level as whole compilation

==189151== Profiling application: ./mandlebrot profile
==189151== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  161.080s        64  2.51688s  2.17734s  2.66619s  void Mandelbrot<float>(uchar4*, int, int, int, float, float, float, uchar4, int, int, int)


==189317== Profiling application: ./mandlebrot_byhand profile
==189317== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  163.530s        64  2.55516s  2.20814s  2.65715s  void Mandelbrot<float>(uchar4*, int, int, int, float, float, float, uchar4, int, int, int)

==189215== Profiling application: ./mandlebrot_rdc profile
==189215== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  167.631s        64  2.61923s  2.27619s  2.70183s  void Mandelbrot<float>(uchar4*, int, int, int, float, float, float, uchar4, int, int, int)


When we don't tune LTO compute kernels we see performance between that of whole compilation
and rdc=true

=190028== Profiling application: ./mandlebrot profile
==190028== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  164.188s        64  2.56543s  2.19639s  2.66142s  void Mandelbrot<float>(uchar4*, int, int, int, float, float, float, uchar4, int, int, int)
