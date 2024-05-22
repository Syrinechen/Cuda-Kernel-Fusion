#include <math.h>

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

extern "C"
__global__ void elementwise_multiply_sigmoid(float* A, float* C, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        float val = A[idx];
        C[idx] = val * sigmoid(val);
    }
}
