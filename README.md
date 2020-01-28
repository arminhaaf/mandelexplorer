# Mandelbrot Explorer

A GPU/AVX accelerated Mandelbrot explorer implemented in Java 9.

GPU implementation using aparapi (http://aparapi.com/)

Start nimra.mandelexplorer.MandelExplore from your IDE

With maven just start mvn (using exec plugin to start mandelExplorer)


Use mouse to explore:

- scroll wheel to zoom
- left click to move
- [Control] double click to to [un-]zoom

## Algorithm






## Performance

Some performance data for a Ryzen 5 3600 with a GeForce RTX2070S on a WHQD (2.560 x 1.440) display.

All calculations without distance calc. QuadFloat implementation is broken for more than double precision.. 

####Julia Island  
  
> (-0.743643887037151 + 0.131825904205330i scale 0.000000000051299) 

|Implementation  | Duration Millis |
| --- | --- |
| GPU Cuda FloatFloat | 150 |
| GPU OpenCL FloatFloat | 160 |
| CPU Native AVX2 Double | 400 |
| GPU Cuda Double | 450 |
| GPU OpenCL Double | 460 |
| CPU Java Double | 1400 |
| GPU Aparapi Double | 1100 |
| CPU Native Double | 1100 |
| GPU Cuda DoubleDouble | 3150 |
| CPU Native AVX2 DoubleDouble | 3150 |
| GPU OpenCL DoubleDouble | 3400 |
| CPU Java DoubleDouble | 9800 |
| CPU Native DoubleDouble | hangs |



####Annabelle1 
  
> (-1.1193802146697243 -0.27181088621256605i scale 1.1146992856978543E-6) 

|Implementation  | Duration Millis |
| --- | --- |
| GPU Cuda FloatFloat | 280 |
| GPU OpenCL FloatFloat | 360 |
| CPU Native AVX2 Double | 850 |
| GPU Cuda Double | 1150 |
| GPU OpenCL Double | 1250 |
| CPU Java Double | 1400 |
| GPU Aparapi Double | 1100 |
| CPU Native Double | 2250 |
| GPU Cuda DoubleDouble | 8750 |
| CPU Native AVX2 DoubleDouble | 7650 |
| GPU OpenCL DoubleDouble | 9900 |
| CPU Java DoubleDouble | 21550 |
| CPU Native DoubleDouble | hangs |


## Some Examples

![Mandel Home](examples/Home.png?raw=true "Mandelbrot")

![Distance Lighting](examples/DistLightExample.png?raw=true "Distance Lighting")

![Histo Palette](examples/HistoExample2.png?raw=true "Histo Palette")

![Histo Palette](examples/HistoExample3.png?raw=true "Histo Palette")