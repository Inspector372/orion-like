/*
    kernel_example.cu
    
*/

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t err = (expr);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                        \
                    cudaGetErrorString(err), __FILE__, __LINE__);               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublasLt.h>
#include "kernel_example.h"
#include <stdio.h>
#include <vector>
#include <cmath>

static void fillSequential(std::vector<float>& v, float scale = 0.1f) {
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = static_cast<float>(i % 10) * scale;
}

__global__ void dotKernel(float* a, float* b, float* result, int N) {
    __shared__ float cache[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float temp = 0.0f;
    if (idx < N) temp = a[idx] * b[idx];
    cache[tid] = temp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) cache[tid] += cache[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, cache[0]);
}

void dotCheck(float* h_A, float* h_B, float h_result, int N) {
    double expected = 0.0;
    for (int i = 0; i < N; i++) expected += (double)h_A[i] * h_B[i];
    double diff = fabs(expected - (double)h_result);
    if (diff > 1e-2 * fabs(expected) + 1e-3) {
        fprintf(stderr, "mismatch: expected=%f, got=%f, diff=%f\n", expected, h_result, diff);
        return;
    }
    fprintf(stderr, "dot product OK: %f\n", h_result);
}

extern "C" void* dotKernel_wrap(void* arg) {
    (void)arg;
    int N = (1 << 16) + 10;
    float *h_A, *h_B, h_result = 0.0f;
    float *d_A, *d_B, *d_result;

    h_A = (float*)malloc(N * sizeof(float));
    h_B = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
        h_B[i] = (float)(rand() % 100) / 10.0f;
    }

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    dotKernel<<<blocks, threads>>>(d_A, d_B, d_result, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    dotCheck(h_A, h_B, h_result, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);
    free(h_A);
    free(h_B);

    cudaError_t err = cudaGetLastError();
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));

    return nullptr;
}

// *this matmul kernel does not work with current design because it only supports 1D atomization.
__global__ void matMulKernel(float* A, float* B, float* C, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < M) {
        float sum = 0.0f;
        for (int k = 0; k < M; k++) {
            sum += A[row * M + k] * B[k * M + col];
        }
        C[row * M + col] = sum;
    }
}

void matMulCheck(float* h_A, float* h_B, float* h_C, int M) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < M; col++) {
            float expected = 0.0f;
            for (int k = 0; k < M; k++) {
                expected += h_A[row * M + k] * h_B[k * M + col];
            }
            float got = h_C[row * M + col];
            if (fabs(expected - got) > 1e-2f * fabs(expected) + 1e-2f) {
                fprintf(stderr, "mismatch at (%d,%d): expected=%f, got=%f\n", row, col, expected, got);
                return;
            }
        }
    }
    fprintf(stderr, "matmul OK\n");
}

extern "C" void* matMulKernel_wrap(void* arg) {
    (void)arg;
    int M = 256;
    size_t bytes = (size_t)M * M * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);
    for (int i = 0; i < M * M; i++) {
        h_A[i] = (float)(rand() % 10);
        h_B[i] = (float)(rand() % 10);
    }

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((M + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    matMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, M);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    matMulCheck(h_A, h_B, h_C, M);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    cudaError_t err = cudaGetLastError();
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));

    return nullptr;
}

__global__ void histKernel(int* data, int* hist, int N, int numBins, int maxVal) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int bin = (data[idx] * numBins) / (maxVal + 1);
        if (bin >= numBins) bin = numBins - 1;
        atomicAdd(&hist[bin], 1);
    }
}

void histCheck(int* h_data, int* h_hist, int N, int numBins, int maxVal) {
    int* expected = (int*)calloc(numBins, sizeof(int));
    for (int i = 0; i < N; i++) {
        int bin = (h_data[i] * numBins) / (maxVal + 1);
        if (bin >= numBins) bin = numBins - 1;
        expected[bin]++;
    }
    for (int b = 0; b < numBins; b++) {
        if (expected[b] != h_hist[b]) {
            fprintf(stderr, "mismatch at bin %d: expected=%d, got=%d\n", b, expected[b], h_hist[b]);
            free(expected);
            return;
        }
    }
    fprintf(stderr, "hist OK\n");
    free(expected);
}

extern "C" void* histKernel_wrap(void* arg) {
    (void)arg;
    int N = 1 << 15;
    int numBins = 16;
    int maxVal = 999;
    int *h_data, *h_hist;
    int *d_data, *d_hist;

    h_data = (int*)malloc(N * sizeof(int));
    h_hist = (int*)malloc(numBins * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % (maxVal + 1);
    }

    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_hist, numBins * sizeof(int));

    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, numBins * sizeof(int));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    histKernel<<<blocks, threads>>>(d_data, d_hist, N, numBins, maxVal);
    cudaDeviceSynchronize();

    cudaMemcpy(h_hist, d_hist, numBins * sizeof(int), cudaMemcpyDeviceToHost);

    histCheck(h_data, h_hist, N, numBins, maxVal);

    cudaFree(d_data);
    cudaFree(d_hist);
    free(h_data);
    free(h_hist);

    cudaError_t err = cudaGetLastError();
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));

    return nullptr;
}


__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// one block per row
__global__ void softmaxKernel(float* in, float* out, int cols) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float* rowData = in + (size_t)row * cols;
    float* rowOut  = out + (size_t)row * cols;

    // 1) row max
    float localMax = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) localMax = fmaxf(localMax, rowData[i]);
    localMax = warpReduceMax(localMax);
    if ((tid % warpSize) == 0) shared[tid / warpSize] = localMax;
    __syncthreads();

    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    float rowMax = (tid < numWarps) ? shared[tid] : -INFINITY;
    if (tid < warpSize) rowMax = warpReduceMax(rowMax);
    if (tid == 0) shared[0] = rowMax;
    __syncthreads();
    rowMax = shared[0];

    // 2) sum of exp
    float localSum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) localSum += expf(rowData[i] - rowMax);
    localSum = warpReduceSum(localSum);
    if ((tid % warpSize) == 0) shared[tid / warpSize] = localSum;
    __syncthreads();

    float rowSum = (tid < numWarps) ? shared[tid] : 0.0f;
    if (tid < warpSize) rowSum = warpReduceSum(rowSum);
    if (tid == 0) shared[0] = rowSum;
    __syncthreads();
    rowSum = shared[0];

    // 3) normalize
    for (int i = tid; i < cols; i += blockDim.x) {
        rowOut[i] = expf(rowData[i] - rowMax) / rowSum;
    }
}

void softmaxCheck(float* h_in, float* h_out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float* rowIn = h_in + (size_t)r * cols;
        float* rowOut = h_out + (size_t)r * cols;
        float maxVal = -INFINITY;
        for (int c = 0; c < cols; c++) maxVal = fmaxf(maxVal, rowIn[c]);
        double sum = 0.0;
        for (int c = 0; c < cols; c++) sum += exp(rowIn[c] - maxVal);
        for (int c = 0; c < cols; c++) {
            float expected = (float)(exp(rowIn[c] - maxVal) / sum);
            if (fabs(expected - rowOut[c]) > 1e-4f) {
                fprintf(stderr, "mismatch at row %d col %d: expected=%f, got=%f\n", r, c, expected, rowOut[c]);
                return;
            }
        }
    }
    fprintf(stderr, "softmax OK\n");
}

extern "C" void* softmaxKernel_wrap(void* arg) {
    (void)arg;
    int rows = 256, cols = 256;
    size_t bytes = (size_t)rows * cols * sizeof(float);
    float *h_in, *h_out;
    float *d_in, *d_out;

    h_in = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);
    for (int i = 0; i < rows * cols; i++) h_in[i] = ((float)(rand() % 2000) - 1000.0f) / 100.0f;

    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int numWarps = (threads + 31) / 32;
    size_t shmem = numWarps * sizeof(float);

    softmaxKernel<<<rows, threads, shmem>>>(d_in, d_out, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    softmaxCheck(h_in, h_out, rows, cols);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    cudaError_t err = cudaGetLastError();
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));

    return nullptr;
}

extern "C" void* test_cublas(void* arg) {
    printf("=== Example 1: Basic FP32 MatMul (C = alpha*A*B + beta*C) ===\n");
 
    const int M = 4, K = 6, N = 5;
    const float alpha = 1.f, beta = 0.f;
 
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.f);
    fillSequential(h_A);
    fillSequential(h_B, 0.2f);
 
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, h_C.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, h_C.size() * sizeof(float)));
 
    // --- cublasLt handle ---
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);
 
    // --- Matrix layout descriptors ---
    // cublasLt is column-major; for a row-major MxK matrix the leading
    // dimension equals K (number of columns) and we set the row/col counts
    // to match the transposed view.
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
 
    // Passing B^T (N x K col-major) and A^T (K x M col-major) gives C^T
    // which lands as row-major C (M x N) in memory.
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, K, M, K); // A^T: K rows, M cols, ld=K
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F, N, K, N); // B^T: N rows, K cols, ld=N
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, N, M, N); // C^T: N rows, M cols, ld=N
 
    // --- MatMul descriptor ---
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
 
    // We pass B^T * A^T; both are already transposed by the layout trick,
    // so set op to CUBLAS_OP_N (no additional transpose).
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
 
    // --- Algorithm selection (let cuBLAS pick automatically) ---
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    const size_t workspaceSize = 32 * 1024; // 32 KB
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
 
    void* d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));
 
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult{};
    cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc,
        layoutB, layoutA, layoutC, layoutC,   // B^T, A^T, C^T, C^T
        pref, 1, &heuristicResult, &returnedResults);
 
    if (returnedResults == 0) {
        fprintf(stderr, "No algorithm found!\n");
        exit(EXIT_FAILURE);
    }
    printf("  Algorithm found (wavesCount=%f)\n", heuristicResult.wavesCount);
 
    // --- Execute ---
    cublasLtMatmul(
        ltHandle, matmulDesc,
        &alpha,
        d_B, layoutB,   // B^T  (first  arg = "A" in col-major GEMM)
        d_A, layoutA,   // A^T  (second arg = "B" in col-major GEMM)
        &beta,
        d_C, layoutC,   // C^T  (in)
        d_C, layoutC,   // C^T  (out)
        &heuristicResult.algo,
        d_workspace, workspaceSize,
        0);             // default stream
 
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));
 
    printf("  C[%dx%d] first row:", M, N);
    for (int j = 0; j < N; ++j) printf(" %f", h_C[j]);
    printf("\n\n");
    printf("Cublas OK...?\n");
 
    // Cleanup
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtDestroy(ltHandle);
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A));
    printf("Cublas OK!\n");
    return nullptr;
}



// 2D Row-Major Matrix Initializer
void initMatrix2D(std::vector<float>& mat, int rows, int cols, float scale) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            mat[r * cols + c] = static_cast<float>(r * cols + c) * scale;
        }
    }
}

// Host verification helper
void verifyResults(const std::vector<float>& h_A, const std::vector<float>& h_B, 
                   const std::vector<float>& h_C, int M, int K, int N) {
    bool passed = true;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k) {
                expected += h_A[r * K + k] * h_B[k * N + c];
            }
            float got = h_C[r * N + c];
            if (std::fabs(expected - got) > 1e-3f) {
                fprintf(stderr, "Mismatch at (%d,%d): expected=%f, got=%f\n", r, c, expected, got);
                passed = false;
                break;
            }
        }
        if (!passed) break;
    }
    if (passed) {
        printf("Verification OK: Host calculations match cuBLASLt output!\n");
    }
}

extern "C" void* test_cublas_prime(void* arg) {
    (void)arg;
    pthread_mutex_t* smutex = ((addKernel_arg*)arg)->smutex;
    pthread_mutex_lock(smutex);
    pthread_mutex_unlock(smutex);

    // Rescaled dimensions
    const int M = 32, K = 32, N = 32;
    const float alpha = 1.0f, beta = 0.0f;

    printf("=== FP32 cuBLASLt MatMul (M=%d, K=%d, N=%d) ===\n", M, K, N);
    printf("Total compute footprint: %d FLOPs\n", 2 * M * N * K);

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.0f);
    initMatrix2D(h_A, M, K, 0.01f);
    initMatrix2D(h_B, K, N, 0.02f);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, h_C.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, h_C.size() * sizeof(float)));

    // --- cuBLASLt Handle ---
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    // --- Layout Descriptors ---
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, K, M, K); // A^T
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F, N, K, N); // B^T
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, N, M, N); // C^T

    // --- MatMul Descriptor ---
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    // --- Algorithm Selection ---
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    const size_t workspaceSize = 32 * 1024; // 32 KB workspace
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

    void* d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult{};
    cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc,
        layoutB, layoutA, layoutC, layoutC,
        pref, 1, &heuristicResult, &returnedResults);

    if (returnedResults == 0) {
        fprintf(stderr, "No algorithm found!\n");
        exit(EXIT_FAILURE);
    }

    // --- Execute ---
    cublasLtMatmul(
        ltHandle, matmulDesc,
        &alpha,
        d_B, layoutB,
        d_A, layoutA,
        &beta,
        d_C, layoutC,
        d_C, layoutC,
        &heuristicResult.algo,
        d_workspace, workspaceSize,
        0);

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Print sample output
    printf("C[%dx%d] Row 0 (first 5 elements):", M, N);
    for (int j = 0; j < 5; ++j) printf(" %f", h_C[j]);
    printf("\n");

    // Verify GPU results against CPU implementation
    verifyResults(h_A, h_B, h_C, M, K, N);

    // Cleanup
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtDestroy(ltHandle);
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A));

    return nullptr;
}

// Kernel 1: Vector Addition C[i] = A[i] + B[i]
__global__ void vecAddKernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel 2: Vector Scaling C[i] = C[i] * alpha (In-place operation)
__global__ void vecScaleKernel(float* C, float alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] *= alpha;
    }
}

// Kernel 3: Does nothing
__global__ void emptyKernel() {
    return;
}

// Host checker: Verifies final result C[i] == (A[i] + B[i]) * alpha
void chainedKernelsCheck(const float* h_A, const float* h_B, const float* h_C, float alpha, int N) {
    for (int i = 0; i < N; i++) {
        float expected = (h_A[i] + h_B[i]) * alpha;
        float got = h_C[i];
        if (fabs(expected - got) > 1e-4f * fabs(expected) + 1e-4f) {
            fprintf(stderr, "Mismatch at index %d: expected=%f, got=%f\n", i, expected, got);
            return;
        }
    }
    fprintf(stderr, "Chained Kernels execution OK\n");
}

// C-compatible wrapper launching multiple kernels in sequence
extern "C" void* chainedKernels_wrap(void* arg) {
    pthread_mutex_t* smutex = ((addKernel_arg*)arg)->smutex;
    pthread_mutex_lock(smutex);
    pthread_mutex_unlock(smutex);

    (void)arg;
    int N = 2232;
    float alpha = 2.5f;  // Scaling factor
    size_t bytes = N * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Allocate host memory
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
        h_B[i] = (float)(rand() % 100) / 10.0f;
    }
    cudaStream_t mystream;
    cudaStreamCreate(&mystream);

    // Allocate device memory
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Host-to-Device Transfer
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Grid Configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch Kernel 1: C = A + B
    fprintf(stderr, "Kernel - Launch First\n");
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Launch Kernel 2: C = C * alpha (uses the intermediate output from Kernel 1)
    fprintf(stderr, "Kernel - Launch Second\n");
    emptyKernel<<<blocksPerGrid, threadsPerBlock>>>();
    vecScaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, alpha, N);

    // Wait for stream completion before checking host memory
    fprintf(stderr, "Synchronize - In\n");
    cudaDeviceSynchronize();
    fprintf(stderr, "Synchronize Pass\n");

    // Device-to-Host Transfer
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    fprintf(stderr, "Memcpy Pass\n");

    // Verify Results
    chainedKernelsCheck(h_A, h_B, h_C, alpha, N);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    cudaStreamDestroy(mystream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    }

    return nullptr;
}