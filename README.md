# ICPCUDA
Super fast implementation of ICP in CUDA for compute capable devices 2.0 or higher. On an nVidia GeForce GTX TITAN X it runs at over __750Hz__ (using projective data assocation). To compile all architectures you'll need CUDA 7.0 I think, (or 6.5 with the special release for 9xx cards). You can compile for older cards by removing the unsupported architectures from the CMakeLists.txt file. 

Requires CUDA, [Pangolin](https://github.com/stevenlovegrove/Pangolin), [Eigen](https://github.com/stevenlovegrove/eigen) and [Sophus](https://github.com/stevenlovegrove/Sophus). I've built it to take in raw TUM RGB-D datasets to do frame-to-frame dense ICP as an example application.

The particular version of ICP implemented is the one introduced by [KinectFusion](http://homes.cs.washington.edu/~newcombe/papers/newcombe_etal_ismar2011.pdf). This means a three level coarse-to-fine registration pyramid, from 160x120 to 320x240 and finally 640x480 image sizes, with 4, 5 and 10 iterations per level respectively. 

Run like;

```bash
./ICP ~/Desktop/rgbd_dataset_freiburg1_desk/ -v
```

Where ~/Desktop/rgbd\_dataset\_freiburg1\_desk/ contains the depth.txt file, for more information see [here](http://vision.in.tum.de/data/datasets/rgbd-dataset).

The main idea to getting the best performance is determining the best thread/block sizes to use. I have provided an exhaustive search function to do this, since it varies between GPUs. Simply pass the "-v" switch to the program to activate the search. The code will then first do a search for the best thread/block sizes and then run ICP and output something like this on an nVidia GeForce GTX TITAN X;

```bash
GeForce GTX TITAN X
Searching for the best thread/block configuration for your GPU...
Best: 256 threads, 96 blocks (1.3306ms), 100%
ICP: 1.3236ms
ICP speed: 755Hz
```

The code will output one file; output.poses. You can evaluate it on the TUM benchmark by using their tools. I get something like this;

```bash
python ~/stuff/Kinect_Logs/Freiburg/evaluate_ate.py ~/Desktop/rgbd_dataset_freiburg1_desk/groundtruth.txt output.poses 
0.144041
```

The difference in values comes down to the fact that each method uses a different reduction scheme and floating point operations are [not associative](https://halshs.archives-ouvertes.fr/hal-00949355v1/document).

Also, if you're using this code in academic work and it would be suitable to do so, please consider referencing some of my possibly relevant [research](http://www.thomaswhelan.ie/#publications) in your literature review/related work section. 
