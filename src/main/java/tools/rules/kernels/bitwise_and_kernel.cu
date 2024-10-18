// bitwise_and_kernel.cu
extern "C"
__global__ void computeAndOperation(int* cover1, int* cover2, int* result, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = cover1[i] & cover2[i];
    }
}
