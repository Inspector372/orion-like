
// Example kernels.

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

__global__ void addKernel(int* a, int* b, int* out, int N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) 
        out[idx] = a[idx] + b[idx];
}

void addCheck(int* h_A, int* h_B, int* h_out, int N) {
    for(int i = 0; i < N; i++) {
        if(h_A[i] + h_B[i] != h_out[i]) {
            fprintf(stderr, "mismatch at %d, h_A[%d]=%d, h_b[%d]=%d, h_out[%d]=%d\n", i, i, h_A[i], i, h_B[i], i, h_out[i]);
            return;
        }
    }
}

extern "C" void* addKernel_wrap(void* arg) {
    int N;
    int* h_A, *h_B, *h_out;
    int* d_A, *d_B, *d_out;
    N = ((addKernel_arg*)arg)->N;
    h_A = ((addKernel_arg*)arg)->h_A;
    h_B = ((addKernel_arg*)arg)->h_B;
    h_out = ((addKernel_arg*)arg)->h_out;
    pthread_mutex_t* smutex = ((addKernel_arg*)arg)->smutex;
    
    // block before setting everything.
    pthread_mutex_lock(smutex);
    pthread_mutex_unlock(smutex);

    
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaError_t err = cudaGetLastError();
    fprintf(stderr, "PRE: Error: %s\n", cudaGetErrorString(err));

    addKernel<<<blocks, threads>>>(d_A, d_B, d_out, N);

    // can this be a problem? this is not a wrapped method.
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    addCheck(h_A, h_B, h_out, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);

    err = cudaGetLastError();
    fprintf(stderr, "POST: Error: %s\n", cudaGetErrorString(err));

    return nullptr;
}

static void fillSequential(std::vector<float>& v, float scale = 0.1f) {
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = static_cast<float>(i % 10) * scale;
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
    const size_t workspaceSize = 32 * 1024 * 1024; // 32 MB
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
}