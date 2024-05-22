# kernel_wrapper.pyx

cdef extern from "cuda_runtime.h":
    int cudaMalloc(void** devPtr, size_t size)
    int cudaMemcpy(void* dst, const void* src, size_t count, int kind)
    int cudaFree(void* devPtr)
    enum cudaMemcpyKind:
        cudaMemcpyHostToDevice
        cudaMemcpyDeviceToHost

cdef extern from "path/to/your/cuda_kernel.cu":
    void elementwise_multiply_sigmoid(float* A, float* C, int width, int height)

def multiply_sigmoid(np.ndarray[np.float32_t, ndim=2] A):
    cdef int width = A.shape[1]
    cdef int height = A.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] C = np.empty_like(A)

    cdef float* d_A
    cdef float* d_C

    cdef int size = width * height * sizeof(float)
    
    # Allocate device memory
    cudaMalloc(&d_A, size)
    cudaMalloc(&d_C, size)
    
    # Copy data from host to device
    cudaMemcpy(d_A, &A[0, 0], size, cudaMemcpyHostToDevice)
    
    # Define block and grid dimensions
    cdef int block_dim = 16
    cdef dim3 block(block_dim, block_dim)
    cdef dim3 grid((width + block_dim - 1) // block_dim, (height + block_dim - 1) // block_dim)
    
    # Launch kernel
    elementwise_multiply_sigmoid<<<grid, block>>>(d_A, d_C, width, height)
    
    # Copy result from device to host
    cudaMemcpy(&C[0, 0], d_C, size, cudaMemcpyDeviceToHost)
    
    # Free device memory
    cudaFree(d_A)
    cudaFree(d_C)
    
    return C
