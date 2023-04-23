# CudaCollisions
Using cuda, calculates the collisions of particles in a 2D space

I've created this project to **practice** and **learn** how to use CUDA. Thus, the functions I made are probably quite poorly written compared to someone who knows what they are doing. 

In order to create the algorithm I based my code on Nvidia [GPU GEMS 3: Chapter 32](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)

The performance is quite good, performing using: CudaCollisions.exe 10 18 100 512 8 0.5 1.0 0.0 0.0 1 1024.0 1024.0, I get the next results: 

```
Args used: ITERATIONS: 10; NUM_OBJECTS: 262144; NUM_BLOCKS: 100; NUM_THREADS: 512; SUB_STEPS: 8;
        MIN_RAD: 0.50; MAX_RAD: 1.00; CALCULATE_RAD: false; MIN_X: 0.00; MIN_Y: 1.00; MAX_X: 1024.00; MAX_Y: 1024.00
Seed used: 1682242813
col:236381
col:181571
col:168507
col:161447
col:158105
col:156723
col:156383
col:156560
col:156853
col:157345

avg time for Cells_Init: 0.6594 ms
avg time for Sort: 8.4458 ms
avg time for Cols_Init: 1.91456 ms
avg time for Cols_Resolve: 6.87591 ms
avg time for Move: 0.4354 ms
total avg time: 18.3311 ms
```

From the results we can see that the sorting is the bottleneck. If we run using a greater area, then the performance of Cols_X would be greater because of fewer collisions, but the sorting would remain roughly the same:

```
Args used: ITERATIONS: 10; NUM_OBJECTS: 262144; NUM_BLOCKS: 100; NUM_THREADS: 512; SUB_STEPS: 8;
        MIN_RAD: 0.50; MAX_RAD: 1.00; CALCULATE_RAD: false; MIN_X: -2048.00; MIN_Y: -2048.00; MAX_X: 2048.00; MAX_Y: 2048.00
Seed used: 1682242757
col:14584
col:7434
col:7352
col:7357
col:7357
col:7357
col:7357
col:7357
col:7357
col:7357

avg time for Cells_Init: 0.70112 ms
avg time for Sort: 8.19718 ms
avg time for Cols_Init: 1.40867 ms
avg time for Cols_Resolve: 1.15457 ms
avg time for Move: 0.25412 ms
total avg time: 11.7157 ms
```

Take into account that the total avg time is multiplied by SUB_STEPS (in the examples, by 8)