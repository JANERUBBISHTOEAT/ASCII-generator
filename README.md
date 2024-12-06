# [PYTHON] ASCII generator

## Fork message

可播彩六

## Introduction

Original readme could be found at <https://github.com/vietnh1009/ASCII-generator>.

This repository is a fork of the original repository, featuring stream-to-stream conversion to ASCII art.

## Changes

- Implemented stream-to-stream conversion.
- Still supports file-to-file conversion.
- Optimized performance by splitting functions into separate threads.  
- Enhanced screen capture speed using `mss`.  
- Replaced nested loops with `np.mean` for improved efficiency.

## Rationale

### Single-threaded performance

![single-thread-fps](./demo/single-thread-fps.png)
> Before threading, the typical fps is 7-8
> (4K, stream-to-stream, file-to-file not supported yet)

### Multi-threaded performance

![multi-thread-fps](./demo/multi-thread-fps.png)
> After splitting the functions into separate threads, the typical fps is 9-10
> (4K, file-to-file)

![before-np](./demo/before-np.png)
> The nested loop version (`main`) takes approximately 0.09 seconds, serving as the program's bottleneck.
> Note that the `screenshot` time here is inaccurately measured, as it includes the wait time for putting the image into the queue while the `main` function continues to process the queue.

### Performance optimization

### v3.0 - sliding window + `np.mean` + threading

![multi-thread-fps](./demo/multi-thread-fps-np.png)
> After replacing the nested loops with `np.mean`, the typical fps is 17-20
> (4K, file-to-file)

### v3.1 - Split `screenshot` and `np` + minor optimizations

![after-np-split](./demo/after-np-split.png)
![after-np-stacked](./demo/after-np-stacked.png)
> The time distribution is now more balanced across each component.

### v3.2 - Split `mean` and `text` + minor optimizations

![split-maen-text](./demo/split-mean-text.png)
> Splitting `mean` and `text` decreases bottleneck time to 0.06 seconds, yielding a theoretical throughput of 16.67 fps and an actual performance of 17-20 fps.
> (4K, file-to-file)

### Next steps

Move time-consuming functions to separate threads to further improve performance.

- [x] Split `mean` and `text` will increase little delay but improve performance.
