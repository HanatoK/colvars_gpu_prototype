# CV on GPU prototype

## Dependencies

- [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python with numpy and pytorch
- [Pytorch C++ library (libtorch)](https://pytorch.org/get-started/locally/)
- CMake and Ninja
- [fmt](https://github.com/fmtlib/fmt)

## Building

1. Download [libtorch](https://pytorch.org/get-started/locally/) and unzip the tarball;
2. Make a `build` directory and enter it: `mkdir -p build && cd build`;
3. Run `cmake -DCMAKE_PREFIX_PATH=<libtorch-dir> ../`. If your GCC version is too new to use with CUDA, then you could install an old one and add a cmake parameter `-DCMAKE_CUDA_HOST_COMPILER='<your-old-g++-binary>'`.

## Running

The RMSD CV implementation is in `test_data/rmsd.py`, which generates a compiled pytorch model `RMSD.pt` after running. The `cuda_torch_example` can be used to read an atomic trajectory file into CUDA device memory and load `RMSD.pt` to compute the RMSD. After compilation, you can run the following command under the `build` directory:
```
./cuda_torch_example ../test_data/RMSD.pt ../test_data/example_rmsd_input.txt
```
The RMSD results are in `test.out`.
