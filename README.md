# Mandelbrot Explorer

A GPU accerelated Mandelbrot explorer implemented in Java 8 using aparapi (https://gitter.im/Syncleus/aparapi)

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

|Implementation  | Duration Millis|
| --- | --- :|
GPU Double | 700  
GPU FloatFloat | 320  
GPU QuadFloat | 2200  
CPU Double | 1400
CPU DoubleDouble | 38000



####Annabelle1 
  
> (-1.1193802146697243 -0.27181088621256605i scale 1.1146992856978543E-6) 

|Implementation  | Duration Millis|
| --- | --- :|
GPU Double | 1200  
GPU FloatFloat | 470  
GPU QuadFloat | 2900  
CPU Double | 2350
CPU DoubleDouble | 41000


## Some Examples

![Mandel Home](examples/Home.png?raw=true "Mandelbrot")

![Distance Lighting](examples/DistLightExample.png?raw=true "Distance Lighting")

![Histo Palette](examples/HistoExample2.png?raw=true "Histo Palette")

![Histo Palette](examples/HistoExample3.png?raw=true "Histo Palette")