#include <cusparse.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#define NUM_EXECUTION 1000
#define BLOCKSIZE 3
// Error checking macro for CUDA API calls
#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Error checking macro for cuSPARSE API calls
#define CHECK_CUSPARSE(call) \
    do { \
        cusparseStatus_t status = call; \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSPARSE Error: " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)



template <class T>
void readValues(std::vector<T>& values, std::string filename) {
    std::ifstream file(filename);
    for (int i = 0; i < values.size(); i++) {
        file >> values[i];
    }
    file.close();
}

__device__ __forceinline__ void blockMultiply(const double* blockValues, const double* x, double* y) {
    #pragma unroll
    for (int i = 0; i < BLOCKSIZE; i++) {
        #pragma unroll
        for (int j = 0; j < BLOCKSIZE; j++) {
            y[i] += blockValues[i * BLOCKSIZE + j] * x[j];
        }
    }
}

__device__ __forceinline__ void blockMultiplyTranspose(const double* blockValues, const double* x, double* y){
    double temp[BLOCKSIZE] = {0.0, 0.0, 0.0};
    #pragma unroll
    for (int i = 0; i < BLOCKSIZE; i++) {
        #pragma unroll
        for (int j = 0; j < BLOCKSIZE; j++) {
            temp[i] += blockValues[j * BLOCKSIZE + i] * x[j];
        }
    }

    #pragma unroll
    for (int i = 0; i < BLOCKSIZE; i++) {
        // atomic add
        atomicAdd(y + i, temp[i]);
        // y[i] += temp[i];
    }

}

__global__ void blockSymmetricSpMV(const double* diagonalBlockValues, const double* offDiagonalBlockValues, const int* blockOuterIndices, const int* blockInnerIndices, const double* x, double* y, double* yt, int nRows, int outerSize) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < outerSize - 1) {
        int start = blockOuterIndices[row];
        int end = blockOuterIndices[row + 1];
        double rowResult[BLOCKSIZE] = {0.0, 0.0, 0.0};
        const double x_slice[BLOCKSIZE] = {*(x + row * BLOCKSIZE), *(x + row * BLOCKSIZE + 1), *(x + row * BLOCKSIZE + 2)};
        // blockMultiply(diagonalBlockValues + row * BLOCKSIZE * BLOCKSIZE, x + row * BLOCKSIZE, rowResult);
        for (int i = start; i < end; i++) {
            int col = blockInnerIndices[i];
            const double block[9] = {*(offDiagonalBlockValues + i * BLOCKSIZE * BLOCKSIZE), *(offDiagonalBlockValues + i * BLOCKSIZE * BLOCKSIZE + 1), *(offDiagonalBlockValues + i * BLOCKSIZE * BLOCKSIZE + 2),
                               *(offDiagonalBlockValues + i * BLOCKSIZE * BLOCKSIZE + 3), *(offDiagonalBlockValues + i * BLOCKSIZE * BLOCKSIZE + 4), *(offDiagonalBlockValues + i * BLOCKSIZE * BLOCKSIZE + 5),
                               *(offDiagonalBlockValues + i * BLOCKSIZE * BLOCKSIZE + 6), *(offDiagonalBlockValues + i * BLOCKSIZE * BLOCKSIZE + 7), *(offDiagonalBlockValues + i * BLOCKSIZE * BLOCKSIZE + 8)};
            blockMultiply(block, x + col * BLOCKSIZE, rowResult);  
            blockMultiplyTranspose(block, x_slice, y + col * BLOCKSIZE); 
        }

        
        #pragma unroll
        for (int i = 0; i < BLOCKSIZE; i++) {
            atomicAdd(y + row * BLOCKSIZE + i, rowResult[i]);
            // y[row * BLOCKSIZE + i] += rowResult[i];
        }
        

    }
    // else if (row < nRows){
    //     // only do the diagonals
    //     double rowResult[BLOCKSIZE] = {0.0, 0.0, 0.0};
    //     blockMultiply(diagonalBlockValues + row * BLOCKSIZE * BLOCKSIZE, x + row * BLOCKSIZE, rowResult);
    //     #pragma unroll
    //     for (int i = 0; i < BLOCKSIZE; i++) {
    //         atomicAdd(y + row * BLOCKSIZE + i, rowResult[i]);
    //         // y[row * BLOCKSIZE + i] += rowResult[i];
    //     }
    // }
}

__global__ void addDaigonal(const double* diagonalBlockValues, const double* x, double* y, const double* yt, int nRows){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nRows){
        double rowResult[BLOCKSIZE] = {0, 0, 0};
        blockMultiply(diagonalBlockValues + row * BLOCKSIZE * BLOCKSIZE, x + row * BLOCKSIZE, rowResult);
        for (int i = 0; i < BLOCKSIZE; i++){
            y[row * BLOCKSIZE + i] += rowResult[i];
        }
    }
}

__global__ void setZero(double* y, int nRows){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nRows){
        for (int i = 0; i < BLOCKSIZE; i++){
            y[row * BLOCKSIZE + i] = 0.0;
        }
    }
}

__global__ void blockSymmetricSpMVCOO(const double* diagonalBlockValues, const double* offDiagonalBlockValues, const int* blockOuterIndices, const int* blockInnerIndices, const double* x, double* y, double* yt, int nRows, int outerSize){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < outerSize){
        int row = blockOuterIndices[id];
        int col = blockInnerIndices[id];
        double rowResult[BLOCKSIZE] = {0.0, 0.0, 0.0};
        const double block[9] = {*(offDiagonalBlockValues + id * BLOCKSIZE * BLOCKSIZE), *(offDiagonalBlockValues + id * BLOCKSIZE * BLOCKSIZE + 1), *(offDiagonalBlockValues + id * BLOCKSIZE * BLOCKSIZE + 2),
            *(offDiagonalBlockValues + id * BLOCKSIZE * BLOCKSIZE + 3), *(offDiagonalBlockValues + id * BLOCKSIZE * BLOCKSIZE + 4), *(offDiagonalBlockValues + id * BLOCKSIZE * BLOCKSIZE + 5),
            *(offDiagonalBlockValues + id * BLOCKSIZE * BLOCKSIZE + 6), *(offDiagonalBlockValues + id * BLOCKSIZE * BLOCKSIZE + 7), *(offDiagonalBlockValues + id * BLOCKSIZE * BLOCKSIZE + 8)};
        blockMultiply(block, x + col * BLOCKSIZE, rowResult);
        #pragma unroll
        for (int i = 0; i < BLOCKSIZE; i++) {
            atomicAdd(y + row * BLOCKSIZE + i, rowResult[i]);
        }
        const double x_slice[BLOCKSIZE] = {*(x + row * BLOCKSIZE), *(x + row * BLOCKSIZE + 1), *(x + row * BLOCKSIZE + 2)};
        blockMultiplyTranspose(block, x_slice, y + col * BLOCKSIZE); 
    }
}




int main() {
    // first we read the dimensions
    int nRows, blockSize, blockOuterSize, blockInnerSize, expandedOuterSize, expandedInnerSize, diagonalBlockSize, offDiagonalBlockSize;
    std::ifstream file("dimensions.txt");
    file >> nRows >> blockSize >> blockOuterSize >> blockInnerSize >> expandedOuterSize >> expandedInnerSize >> diagonalBlockSize >> offDiagonalBlockSize;
    file.close();
    printf("nRows: %d, blockSize: %d\n", nRows, blockSize);
    printf("blockOuterSize: %d, blockInnerSize: %d\n", blockOuterSize, blockInnerSize);
    printf("offDiagonalBlockSize: %d, diagonalBlockSize: %d\n", offDiagonalBlockSize, diagonalBlockSize);
    printf("expandedOuterSize: %d, expandedInnerSize: %d\n", expandedOuterSize, expandedInnerSize);
    std::vector<int> blockOuterIndices(blockOuterSize);
    std::vector<int> blockInnerIndices(blockInnerSize);
    std::vector<int> expandedOuterIndices(expandedOuterSize);
    std::vector<int> expandedInnerIndices(expandedInnerSize);
    std::vector<double> diagonalBlockValues(diagonalBlockSize);
    std::vector<double> offDiagonalBlockValues(offDiagonalBlockSize);
    std::vector<double> expandedData(expandedInnerSize);
    std::vector<double> x(nRows * blockSize);
    std::vector<double> result(nRows * blockSize);
    std::vector<double> computedResult(nRows * blockSize);
    
    std::vector<int> blockFullOuterIndices;
    blockFullOuterIndices.reserve(blockInnerSize);

    std::vector<int> expandedFullOuterIndices;
    expandedFullOuterIndices.reserve(expandedInnerSize);

    // now read all the values
    readValues<int>(blockOuterIndices, "block_csr_outer.txt");
    readValues<int>(blockInnerIndices, "block_csr_inner.txt");
    readValues<int>(expandedOuterIndices, "expanded_csr_outer.txt");
    readValues<int>(expandedInnerIndices, "expanded_csr_inner.txt");
    readValues<double>(diagonalBlockValues, "diagonal_blocks.txt");
    readValues<double>(offDiagonalBlockValues, "off_diagonal_blocks.txt");
    readValues<double>(expandedData, "expanded_csr_values.txt");
    readValues<double>(x, "dense_vector.txt");
    readValues<double>(result, "result.txt");

    for (int i = 0; i < blockOuterIndices.size() - 1; i++) {
        for (int j = blockOuterIndices[i]; j < blockOuterIndices[i + 1]; j++) {
            blockFullOuterIndices.push_back(i);
        }
    }
    for (int i = 0; i < expandedOuterIndices.size() - 1; i++) {
        for (int j = expandedOuterIndices[i]; j < expandedOuterIndices[i + 1]; j++) {
            expandedFullOuterIndices.push_back(i);
        }
    }

    printf("blockFullOuterIndices size: %lu\n", blockFullOuterIndices.size());
    printf("expandedFullOuterIndices size: %lu\n", expandedFullOuterIndices.size());





    // create the cusparse handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaError_t cuda_error;
    double error = 0.0;
    double sum2 = 0.0;
    float milliseconds = 0;

    double *d_expanded_csr_vals, *d_x, *d_y, *d_yt;
    int *d_expanded_csr_outer_indices, *d_expanded_csr_inner_indices, *d_expanded_full_outer_indices;

    // allocate the memory on the device
    cudaMalloc((void**)&d_x, x.size() * sizeof(double));
    cudaMalloc((void**)&d_y, result.size() * sizeof(double));
    cudaMalloc((void**)&d_yt, result.size() * sizeof(double));
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice);


    cudaMalloc((void**)&d_expanded_csr_vals, expandedData.size() * sizeof(double));
    cudaMalloc((void**)&d_expanded_csr_inner_indices, expandedInnerIndices.size() * sizeof(int));
    cudaMalloc((void**)&d_expanded_csr_outer_indices, expandedOuterIndices.size() * sizeof(int));
    


    // Copy data to device
    cudaMemcpy(d_expanded_csr_vals, expandedData.data(), expandedData.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expanded_csr_inner_indices, expandedInnerIndices.data(), expandedInnerIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expanded_csr_outer_indices, expandedOuterIndices.data(), expandedOuterIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    

    // Prepare the cusparseSpMV operation
    double alpha = 1.0;
    double beta = 0.0;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    
    cusparseCreateCsr(&matA, nRows * blockSize, nRows * blockSize, expandedData.size(), d_expanded_csr_outer_indices, d_expanded_csr_inner_indices, d_expanded_csr_vals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cudaEventRecord(start);
    cusparseCreateDnVec(&vecX, nRows * blockSize, d_x, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, nRows * blockSize, d_y, CUDA_R_64F);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUSPARSE allocation time: %f ms\n", milliseconds);

    size_t bufferSize;
    void* dBuffer;
    
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // do the actual SpMV operation
    cudaEventRecord(start);
    for (int i = 0; i < NUM_EXECUTION; i++){
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
                    &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUSPARSE SpMV time for %d executions: %f ms\n", NUM_EXECUTION, milliseconds);

    // Copy result back to host
    cudaMemcpy(computedResult.data(), d_y, computedResult.size() * sizeof(double), cudaMemcpyDeviceToHost);

    // Compute the error
    for (int i = 0; i < result.size(); i++) {
        error += (result[i] - computedResult[i]) * (result[i] - computedResult[i]);
    }
    printf("Error: %lf\n", error);

    cusparseDestroySpMat(matA);

    // do the coo method
    cudaMalloc((void**)&d_expanded_full_outer_indices, expandedFullOuterIndices.size() * sizeof(int));
    cudaMemcpy(d_expanded_full_outer_indices, expandedFullOuterIndices.data(), expandedFullOuterIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cusparseSpMatDescr_t matA2;
    cusparseCreateCoo(&matA2, nRows * blockSize, nRows * blockSize, expandedData.size(), d_expanded_full_outer_indices, d_expanded_csr_inner_indices, d_expanded_csr_vals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    

    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                            &alpha, matA2, vecX, &beta, vecY, CUDA_R_64F,
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    // do the actual SpMV operation
    cudaEventRecord(start);
    for (int i = 0; i < NUM_EXECUTION; i++){
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA2, vecX,
                    &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUSPARSE SpMV COO time for %d executions: %f ms\n", NUM_EXECUTION, milliseconds);
    // Copy result back to host
    cudaMemcpy(computedResult.data(), d_y, computedResult.size() * sizeof(double), cudaMemcpyDeviceToHost);
    
    error = 0.0;
    // Compute the error
    for (int i = 0; i < result.size(); i++) {
        error += (result[i] - computedResult[i]) * (result[i] - computedResult[i]);
    }
    sum2 = 0.0;
    for (int i = 0; i < result.size(); i++) {
        sum2 += computedResult[i] * computedResult[i];
    }
    printf("Squared sum: %lf\n", sum2);
    printf("Error: %lf\n", error);

    // Clean up
    cudaFree(d_expanded_csr_vals);
    cudaFree(d_expanded_csr_outer_indices);
    cudaFree(d_expanded_csr_inner_indices);
    cudaFree(d_expanded_full_outer_indices);

    cudaFree(dBuffer);
    cusparseDestroySpMat(matA2);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);


    // use our method
    double* d_diagonalBlockValues, *d_offDiagonalBlockValues;
    int* d_blockOuterIndices, *d_blockInnerIndices;
    CHECK_CUDA(cudaMalloc((void**)&d_diagonalBlockValues, diagonalBlockValues.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_offDiagonalBlockValues, offDiagonalBlockValues.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_blockOuterIndices, blockOuterIndices.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_blockInnerIndices, blockInnerIndices.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_diagonalBlockValues, diagonalBlockValues.data(), diagonalBlockValues.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_offDiagonalBlockValues, offDiagonalBlockValues.data(), offDiagonalBlockValues.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_blockOuterIndices, blockOuterIndices.data(), blockOuterIndices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_blockInnerIndices, blockInnerIndices.data(), blockInnerIndices.size() * sizeof(int), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    // set y to zero
    setZero<<<(nRows + 32) / 32, 32>>>(d_y, nRows);
    setZero<<<(nRows + 32) / 32, 32>>>(d_yt, nRows);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < NUM_EXECUTION; i++){
    // for(int i = 0; i < 1; i++){
        setZero<<<(nRows + 32) / 32, 32>>>(d_y, nRows);
        // setZero<<<(nRows + 32) / 32, 32>>>(d_yt, nRows);
        blockSymmetricSpMV<<<(nRows + 32) / 32, 32>>>(d_diagonalBlockValues, d_offDiagonalBlockValues, d_blockOuterIndices, d_blockInnerIndices, d_x, d_y, d_yt, nRows, blockOuterIndices.size());
        addDaigonal<<<(nRows + 32) / 32, 32>>>(d_diagonalBlockValues, d_x, d_y, d_yt, nRows);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
        return -1; // or handle the error as needed
    }
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Block SpMV time for %d executions: %f ms\n", NUM_EXECUTION, milliseconds);
    // Copy result back to host
    cudaMemcpy(computedResult.data(), d_y, computedResult.size() * sizeof(double), cudaMemcpyDeviceToHost);

    // Compute the error
    error = 0.0;
    for (int i = 0; i < result.size(); i++) {
        error += (result[i] - computedResult[i]) * (result[i] - computedResult[i]);
    }
    printf("Error: %lf\n", error);


    // use coo method
    int* d_blockFullOuterIndices;
    CHECK_CUDA(cudaMalloc((void**)&d_blockFullOuterIndices, blockFullOuterIndices.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_blockFullOuterIndices, blockFullOuterIndices.data(), blockFullOuterIndices.size() * sizeof(int), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    // set y to zero
    setZero<<<(nRows + 32) / 32, 32>>>(d_y, nRows);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < NUM_EXECUTION; i++){
        setZero<<<(nRows + 32) / 32, 32>>>(d_y, nRows);
        blockSymmetricSpMVCOO<<<(blockFullOuterIndices.size() + 32) / 32, 32>>>(d_diagonalBlockValues, d_offDiagonalBlockValues, d_blockFullOuterIndices, d_blockInnerIndices, d_x, d_y, d_yt, nRows, blockFullOuterIndices.size());
        addDaigonal<<<(nRows + 32) / 32, 32>>>(d_diagonalBlockValues, d_x, d_y, d_yt, nRows);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
        return -1; // or handle the error as needed
    }
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Block SpMV COO time for %d executions: %f ms\n", NUM_EXECUTION, milliseconds);
    // Copy result back to host
    cudaMemcpy(computedResult.data(), d_y, computedResult.size() * sizeof(double), cudaMemcpyDeviceToHost);

    // Compute the error
    error = 0.0;
    for (int i = 0; i < result.size(); i++) {
        error += (result[i] - computedResult[i]) * (result[i] - computedResult[i]);
    }
    printf("Error: %lf\n", error);

    sum2 = 0.0;
    for (int i = 0; i < result.size(); i++) {
        sum2 += computedResult[i] * computedResult[i];
    }
    printf("Squared sum: %lf\n", sum2);


    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_diagonalBlockValues);
    cudaFree(d_offDiagonalBlockValues);
    cudaFree(d_blockOuterIndices);
    cudaFree(d_blockInnerIndices);
    cudaFree(d_yt);
    cudaFree(d_blockFullOuterIndices);


    return 0;
}