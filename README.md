# Elf: Accelerate High-resolution Mobile Deep Vision with Content-aware Parallel Offloading

## Abstract
As mobile devices continuously generate streams of images and videos, a new class of mobile deep vision applications are rapidly emerging, 
which usually involve running deep neural networks on these multimedia data in real-time. To support such applications, having mobile devices offload the computation, especially the neural network inference, to edge clouds has proved effective. 
Existing solutions often assume there exists a dedicated and powerful server, to which the entire inference can be offloaded. In reality, however, we may not be able to find such a server but need to make do with less powerful ones. To address these more practical situations, we propose to partition the video frame and offload the partial inference tasks to multiple servers for parallel processing. This paper presents the design of Elf, a framework to accelerate the mobile deep vision applications with any server provisioning through the parallel offloading.
Elf employs a recurrent region proposal prediction algorithm, a region proposal centric frame partitioning, and a resource-aware multi-offloading scheme. We implement and evaluate Elf upon Linux and Android platforms using four commercial mobile devices and three deep vision applications with ten state-of-the-art models.
The comprehensive experiments show that Elf can speed up the applications by 4.85X with saving bandwidth usage by 52.6%,  while with <1% application accuracy sacrifice.

## Summary
This repo is the implementation of Elf in both Python and C++. Also, it wraps the C++ library with Java Native Interface (JNI) to support Android applications. Our implementation is developed on Ubuntu16.04 and Android10.
We integrate ZeroMQ4.3.2, an asynchronous messaging library that is widely adopted in distributed and concurrent systems, for high-performance multi-server offloading. We use NVIDIA docker to run offloading tasks on edge servers.
We also wrap nvJPEG with Pybind11 for efficient hardware-based image/video encoding on mobile devices.
