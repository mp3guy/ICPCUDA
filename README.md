# ICPCUDA
Super fast implementation of ICP in CUDA for compute capable devices 2.0 or higher. On an nVidia GeForce GTX 780 Ti it runs at over __540Hz__ (using projective data assocation). To compile all architectures you'll need CUDA 7.0 I think, (or 6.5 with the special release for 9xx cards). You can compile for older cards by removing the unsupported architectures from the CMakeLists.txt file. 

Requires CUDA, Boost, Eigen and OpenCV. I've built it to take in raw TUM RGB-D datasets to do frame-to-frame dense ICP as an example application.

The code is a mishmash of my own stuff written from scratch, plus a bunch of random classes/types taken from [PCL](https://github.com/PointCloudLibrary/pcl/tree/master/gpu/kinfu/src/cuda) (on which the code does not depend :D). The slower version of ICP I compare to is the exact same version in PCL. In my benchmarks I have also found it to be faster than the [SLAMBench](http://apt.cs.manchester.ac.uk/projects/PAMELA/tools/SLAMBench/) implementation and hence the [KFusion](https://github.com/GerhardR/kfusion) implementation. I have not tested against [InfiniTAM](https://github.com/victorprad/InfiniTAM).

The particular version of ICP implemented is the one introduced by [KinectFusion](http://homes.cs.washington.edu/~newcombe/papers/newcombe_etal_ismar2011.pdf). This means a three level coarse-to-fine registration pyramid, from 160x120 to 320x240 and finally 640x480 image sizes, with 4, 5 and 10 iterations per level respectively. 

The fast ICP implementation, which is my own, essentially exploits the shlf instruction added to compute capable 3.0 devices that removes the need for warp level synchronisation when exchanging values, see more [here](http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/). In the case that your card is less than compute capable 3.0, I've included a fake shfl method which accomplishes the same thing at the cost of performance. 

Run like;

```bash
./ICP ~/Desktop/rgbd_dataset_freiburg1_desk/
```

Where ~/Desktop/rgbd\_dataset\_freiburg1\_desk/ contains the association.txt file with rgb first and depth second, for more information see [here](http://vision.in.tum.de/data/datasets/rgbd-dataset).

The main idea to getting the best performance is determining the best thread/block sizes to use. I have provided an exhaustive search function to do this, since it varies between GPUs. Simply pass the "-v" switch to the program to activate the search. The code will then first do a search for the best thread/block sizes and then run both methods for ICP and output something like this on an nVidia GeForce GTX 780 Ti;

```bash
GeForce GTX 780 Ti
Searching for the best thread/block configuration for your GPU...
Best: 128 threads, 112 blocks (1.825ms), 100%    
Fast ICP: 1.8486ms, Slow ICP: 6.0648ms
3.2807 times faster. Fast ICP speed: 540Hz
```

And something like this on an nVidia GeForce GTX 880M;

```bash
GeForce GTX 880M
Searching for the best thread/block configuration for your GPU...
Best: 512 threads, 16 blocks (2.8558ms), 100%    
Fast ICP: 2.8119ms, Slow ICP: 11.0008ms
3.9122 times faster. Fast ICP speed: 355Hz
```

The code will output two files, fast.poses and slow.poses. You can evaluate them on the TUM benchmark by using their tools. I get something like this;

```bash
python ~/stuff/Kinect_Logs/Freiburg/evaluate_ate.py ~/Desktop/rgbd_dataset_freiburg1_desk/groundtruth.txt fast.poses 
0.147173
python ~/stuff/Kinect_Logs/Freiburg/evaluate_ate.py ~/Desktop/rgbd_dataset_freiburg1_desk/groundtruth.txt slow.poses 
0.147113
```

The difference in values comes down to the fact that each method uses a different reduction scheme and floating point operations are [not associative](https://halshs.archives-ouvertes.fr/hal-00949355v1/document).

Also, if you're using this code in academic work and it would be suitable to do so, please consider referencing some of my possibly relevant [research](http://www.thomaswhelan.ie/#publications) in your literature review/related work section. 
