# ICPCUDA
Super fast implementation of ICP in CUDA for compute capable devices 3.0 or higher

Requires CUDA, Boost, Eigen and OpenCV. I've built it to take in raw TUM RGB-D datasets to do frame-to-frame dense ICP as an example application.

The code is a mishmash of my own stuff written from scratch, plus a bunch of random classes/types taken from [PCL](https://github.com/PointCloudLibrary/pcl/tree/master/gpu/kinfu/src/cuda) (on which the code does not depend :D). The slower version of ICP I compare to is the exact same version in PCL. In my benchmarks I have also found it to be faster than the [SLAMBench](http://apt.cs.manchester.ac.uk/projects/PAMELA/tools/SLAMBench/) implementation and hence the [KFusion](https://github.com/GerhardR/kfusion) implementation. I have not tested against [InfiniTAM](https://github.com/victorprad/InfiniTAM).

The fast ICP implementation, which is my own, essentially exploits the shlf instruction added to compute capable 3.0 devices that removes the need for warp level synchronisation when exchanging values, see more [here](http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/).

Run like;

```bash
./ICP ~/Desktop/rgbd_dataset_freiburg1_desk/
```

Where ~/Desktop/rgbd\_dataset\_freiburg1\_desk/ contains the association.txt file with rgb first and depth second, for more information see [here](http://vision.in.tum.de/data/datasets/rgbd-dataset).

The code will run both methods for ICP and output something like this on an nVidia GeForce GTX 780 Ti;

```bash
Fast ICP: 3.8693, Slow ICP: 6.1334
1.5852 times faster.
```

And something like this on an nVidia GeForce GTX 880M;

```bash
Fast ICP: 8.0522, Slow ICP: 11.3533
1.4100 times faster.
```

The main part to mess with is the thread/block sizes used, around line 339 of src/Cuda/icp.cu. Try what's best for you! 

The code will output two files, fast.poses and slow.poses. You can evaluate them on the TUM benchmark by using their tools. I get something like this;

```bash
python ~/stuff/Kinect_Logs/Freiburg/evaluate_ate.py ~/Desktop/rgbd_dataset_freiburg1_desk/groundtruth.txt fast.poses 
0.147061
python ~/stuff/Kinect_Logs/Freiburg/evaluate_ate.py ~/Desktop/rgbd_dataset_freiburg1_desk/groundtruth.txt slow.poses 
0.147113
```

The difference in values comes down to the fact that each method uses a different reduction scheme and floating point operations are [not associative](https://halshs.archives-ouvertes.fr/hal-00949355v1/document).

Also, if you're using this code in academic work and it would be suitable to do so, please consider referencing some of my possibly relevant [research](http://www.thomaswhelan.ie/#publications) in your literature review/related work section. 
