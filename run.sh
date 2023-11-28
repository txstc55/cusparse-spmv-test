nvcc -o spmv spmv.cu -O3 -gencode arch=compute_86,code=sm_86 -use_fast_math -lcusparse
./spmv
