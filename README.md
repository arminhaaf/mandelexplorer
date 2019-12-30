# Mandelbrot Explorer

A GPU accerelated Mandelbrot explorer implemented in Java 8 using aparapi (https://gitter.im/Syncleus/aparapi)

Start nimra.mandelexplorer.MandelExplore from your IDE

With maven just start mvn (using exec plugin to start mandelExplorer)


Use mouse to explore:

- scroll wheel to zoom
- left click to move
- [Control] double click to to [un-]zoom

## Performance

Some performance data for a Ryzen 5 3600 with a GeForce RTX2070S on a WHQD (2.560 x 1.440) Display

Float calculation on a GPU is blazing fast: WHQD (2.560 x 1.440) inside only rendering with max iterations 10000 
about 130ms (with distance calculation). Wheras JTP (Java Threads) float calculation takes about 10s.
However float rendering is limited -> for  my WHQD display to a scale of 0.0001.

For calculation with double the performance advantage of a GPU is small. GPU needs 3.7s , JTP same as float about 10s


####Julia Island  
  
> (-0.743643887037151 + 0.131825904205330i scale 0.000000000051299) 

GPU Double: 700ms  
CPU Double: 1400ms


####Annabelle1 
  
> (-1.1193802146697243 -0.27181088621256605i scale 1.1146992856978543E-6) 

GPU Double: 1900ms  
CPU Double: 3000ms



## Some Examples

![Mandel Home](examples/Home.png?raw=true "Mandelbrot")

![Distance Lighting](examples/DistLightExample.png?raw=true "Distance Lighting")

![Histo Palette](examples/HistoExample2.png?raw=true "Histo Palette")

![Histo Palette](examples/HistoExample3.png?raw=true "Histo Palette")