---
title: compute_ptx_gpu
date: 2025-11-08 18:35:59
tags:
    - GPU
    - CUDA
    - PTX
    - Compile
categories:
    - GPU
---

### Tally：非侵入式的GPU资源隔离框架

分享一篇最近的论文，我觉得其中最让我眼前一亮的是他的技术手段。文章的动机比较常见，是要解决GPU上运行多应用时，出现的相互干扰的问题。

围绕这类问题，最近几年出现了很多文章，主要场景有云环境中GPU服务多租户、智能驾驶领域提高GPU使用率等。当一个GPU上运行多个不同应用，可能会造成如下问题：

1，相互干扰导致应用尾延迟增大。

2，关键型应用无法立刻执行。

3，GPU资源利用率降低。

为了解决这些问题，常见的方案有切分和抢占两种措施。众所周知，GPU上的执行并不能用户精细操控。当kernel从cpu端发射到GPU端后，自动切分为block并且随机分配给SM执行。所以当一个kernel包含的block数量过多时，就会导致停留在GPU上等待调度队列的block数量过多，从而阻塞了新的kernel的发射和执行。因此，切分的方法就是将大的kernel切分成小kernel，将一次发射变为多次发射。从而可以在发射阶段来控制是否要继续执行。

抢占的实现有多种做法，本文是从代码级别实现的。由于nvidia GPU不开源，同时GPU本身不提供抢占原语，所以无法从硬件层面实现抢占的效果。本文从代码层面，将源代码转换为了持久化线程风格的代码，从而实现了手动抢占的实现。
持久化线程（PTB），指的是让kernel不再执行一次任务后就结束，而是可以不断地主动获取任务，从而实现代码层面控制block的执行进度。控制block的执行进度这一点使用一般的编程风格是做不到的，因此block在SM上的执行是无序随机的，不能确定哪个block先执行，哪个block后执行。
将原本kernel转换成持久化线程的方案也不难理解。首先是将原本的kernel作为最外层循环中执行的任务，之后添加一些控制变量，比如任务的ID，抢占信号等。

原本的kernel：
```
__global__ void kernel(float* input, float* output, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    output[idx] = input[idx] / 255.0f; // 具体任务
    return;
}
```
转换后的kernel：
```
__global__ int global_idx;
__global__ bool *preempt_flag;
__global__ void kernel(
    float* input, float* output, int total,
    volatile int* global_idx, volatile bool* preempt_flag
) {
    while (true) {
        if (*preempt_flag) return;
        int idx = atomicAdd(global_idx, 1);
        if (idx >= total) continue;
        output[idx] = input[idx] / 255.0f; // 具体任务
    }
}
```
上述的两种方案是并行的，适合不同类型的kernel。有的kernel发射开销比较大，那么可能就更倾向于选择抢占方式的kernel转换。作者在衡量kernel具体适合什么样的转换方式时，使用了一个叫做周转时间的衡量指标。这个指标的含义是当前kernel在执行时切换上下文大概要花费的时间。对于切分的方案来说，这个时间就等同于小kernel本身的执行时间，因为切分必须要等待小kernel完成后才可以切换上下文。

实际上，本文开头是和TGS做对比的，强调的是对用户透明同时又可以进行更加细粒度的调度。切分和抢占都是具体的block级别的调度方法，而对用户透明这一点，则是整体的框架实现。上面提到的转换方案，都是Tally框架在kernel运行前，通过实时的ptx级的kernel修改得到的结果，从而实现了无侵入式的kernel调度框架。这个技术方案是最近发现的唯一一篇可以在PTX修改做到这个程度并且开源的文章，大家可以去学习借鉴~

![alt text](/source/images/image.png)

![alt text](/source/images/image-1.png)