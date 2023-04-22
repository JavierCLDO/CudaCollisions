# CudaCollisions
Using cuda, calculates the collisions of particles in a 2D space

I've created this project to **practice** and **learn** how to use CUDA. Thus, the functions I made have probably quite poor performance compared to someone who knows what they are doing. 

In order to create the algorithm I based my code on Nvidia [GPU GEMS 3: Chapter 32](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)

The performance is quite good, performing using: CudaCollisions.exe 10 18 100 512 8 0.5 1.0 0.0 0.0 1024.0 1024.0, I get the next results: 

```
Args used: ITERATIONS: 10; NUM_OBJECTS: 262144; NUM_BLOCKS: 100; NUM_THREADS: 512; SUB_STEPS: 8; MIN_RAD: 0.50; MAX_RAD: 1.00; MIN_X: 0.00; MIN_Y: 0.00; MAX_X: 1024.00; MAX_Y: 1024.00

col:235812
col:181231
col:167582
col:160547
col:157298
col:155929
col:155553
col:155799
col:156220
col:156688

avg time for Cells_Init: 0.6555 ms
avg time for Sort: 7.63372 ms
avg time for Cols_Init: 1.54637 ms
avg time for Cols_Resolve: 5.20316 ms
avg time for Move: 0.51054 ms
total avg time: 15.5493 ms
```

From the results we can see that the sorting is the bottleneck. If we run using a greater area, then the performance of Cols_X would be greater because of fewer collisions, but the sorting would remain the same:

```
Args used: ITERATIONS: 10; NUM_OBJECTS: 262144; NUM_BLOCKS: 100; NUM_THREADS: 512; SUB_STEPS: 8; MIN_RAD: 0.50; MAX_RAD: 1.00; MIN_X: -2048.00; MIN_Y: -2048.00; MAX_X: 2048.00; MAX_Y: 2048.00
col:14851
col:7635
col:7508
col:7507
col:7505
col:7505
col:7505
col:7505
col:7505
col:7505

avg time for Cells_Init: 0.65918 ms
avg time for Sort: 8.50939 ms
avg time for Cols_Init: 1.39715 ms
avg time for Cols_Resolve: 0.89879 ms
avg time for Move: 0.31606 ms
total avg time: 11.7806 ms
```

Take into account the the total avg time is multiplied by SUB_STEPS (in the examples, by 8)