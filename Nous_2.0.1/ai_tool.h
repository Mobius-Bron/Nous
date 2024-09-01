#include <stdio.h>
#include <cuda_runtime.h>

cudaError_t ErrorCheck(cudaError_t errorCode, const char *flieName, int lineNumber){
    if(errorCode != cudaSuccess){
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line=%d",
        errorCode, cudaGetErrorName(errorCode), cudaGetErrorString(errorCode), flieName, lineNumber);
    }
    return errorCode;
}

// CUDA 内核函数定义
__global__ void spiteFromGPU(float *v, float *thr, float *spite, const int N) {
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= N) return;
    if (v[gid] > thr[gid]) {
        spite[gid] = 1.0f;
        v[gid] = 0.0f;
    } else {
        spite[gid] = 0.0f;
    }
}

__global__ void countFromGPU(float *A, float *B, float *C, const int N, char op) {
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= N) return;
    switch (op) {
        case '+': C[gid] = A[gid] + B[gid]; break;
        case '-': C[gid] = A[gid] - B[gid]; break;
        case '*': C[gid] = A[gid] * B[gid]; break;
        case '/': C[gid] = A[gid] / B[gid]; break;
    }
}

__global__ void matrixMultiply(float *A, float *B, float *C, int M, int N, int K) {
    // 计算当前线程对应的 C 矩阵的位置
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    C[Row * K + Col] = 0.0f;
    // 检查线程是否在 C 矩阵的有效范围内
    if (Row < M && Col < K) {
        for (int k = 0; k < N; ++k) {
            C[Row * K + Col] += A[Row * N + k] * B[k * K + Col];
        }
    }
}

__global__ void addBias(float *output, float *bias, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        output[idx] += bias[idx];
    }
}

// 初始化随机浮点数数据
void initRandomFloatData(float* addr, int elemCount) {
    for (int i = 0; i < elemCount; i++) {
        addr[i] = static_cast<float>(rand() & 0xFFF);
    }
}

// 主机端内存分配（使用 cudaMallocHost）
float* memsetHost(size_t bytesCount) {
    float *fpHost;
    cudaMallocHost(&fpHost, bytesCount);
    if (fpHost == nullptr) {
        printf("Fail to allocate memory.\n");
        exit(-1);
    }
    memset(fpHost, 0, bytesCount);
    return fpHost;
}

// 设备端内存分配
float* memsetDevice(size_t bytesCount) {
    float *fpDevice;
    cudaMalloc(&fpDevice, bytesCount);
    if (fpDevice == nullptr) {
        printf("Fail to allocate memory.\n");
        exit(-1);
    }
    cudaMemset(fpDevice, 0, bytesCount);
    return fpDevice;
}


class tensor{
public:
    tensor(int shape_size, int* shape);
    ~tensor();
private:
    int shape_size_;
    int data_size_;
    int* shape_;
    float* h_data_;
};

tensor::tensor(int shape_size, int* shape):shape_size_(shape_size), shape_(shape){ // 类初始化
    data_size_ = 0;
    for(int i=0; i<shape_size_; i++){
        data_size_ += shape_[i];
    }
    size_t byteCount = data_size_ * sizeof(float);
    h_data_ = memsetHost(byteCount);
}

tensor::~tensor(){ // 变量周期结束
    cudaFreeHost(h_data_);
}

class Linear{
public:
    Linear(int input_features_, int output_features_);
    ~Linear();

    void forward(float* input, float* output);
    void sync_weights_to_host();
private:
    int input_features_;
    int output_features_;

    float* h_weight_;
    float* h_bias_;

    float* d_weight_;
    float* d_bias_;

    void initialize_weights_and_bias();
};

Linear::Linear(int input_features, int output_features)
:input_features_(input_features), output_features_(output_features){
    size_t byteCount = input_features_ * output_features_ * sizeof(float);
    // 分配主机端内存
    h_weight_ = memsetHost(byteCount);
    h_bias_ = memsetHost(byteCount);

    // 初始化权重和偏置
    initialize_weights_and_bias();

    // 分配设备端内存
    d_weight_= memsetDevice(byteCount);
    d_bias_ = memsetDevice(byteCount);

    // 将权重和偏置从主机复制到设备
    cudaMemcpy(d_weight_, h_weight_, input_features_ * output_features_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_, h_bias_, output_features_ * sizeof(float), cudaMemcpyHostToDevice);
}

Linear::~Linear(){
    cudaFreeHost(h_weight_);
    cudaFreeHost(h_bias_);

    cudaFree(d_weight_);
    cudaFree(d_bias_);
}

void Linear::initialize_weights_and_bias() {
    // 简单地初始化权重和偏置
    for (int i = 0; i < input_features_ * output_features_; ++i) {
        h_weight_[i] = 1.0f; // 或者随机初始化
    }
    for (int i = 0; i < output_features_; ++i) {
        h_bias_[i] = 2.0f; // 或者随机初始化
    }
}

void Linear::forward(float* x, float* output) {
    // 矩阵乘法参数
    int M = 1; // 输出通道数
    int N = input_features_;  // 输入通道数
    int K = output_features_;  // 输入通道数

    // 线程块大小和网格大小
    dim3 threadsPerBlock(20, 20); // 每个块的线程数
    dim3 numBlocks(
        (K + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 矩阵乘法内核调用
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(x, d_weight_, output, M, N, K);

    // 添加偏置的内核调用
    int numThreadsForBias = 256; // 可以根据实际情况调整线程数
    dim3 gridForBias((K + numThreadsForBias - 1) / numThreadsForBias);
    dim3 blockForBias(numThreadsForBias);
    addBias<<<gridForBias, blockForBias>>>(output, d_bias_, K);
}

void Linear::sync_weights_to_host() {
    // 将权重和偏置从设备复制回主机
    cudaMemcpy(h_weight_, d_weight_, input_features_ * output_features_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bias_, d_bias_, output_features_ * sizeof(float), cudaMemcpyDeviceToHost);
}


class Neuron{
public:

private:
    float h_voltage_;          // 主机端的电压
    float h_thr_;              // 主机端的阈值
    float h_spike_;            // 主机端的当前脉冲
    float h_last_spike_time_;  // 主机端的上次脉冲时间
    float h_weight_;           // 主机端的权重

    float* d_voltage_;          // 设备端的电压
    float* d_thr_;              // 设备端的阈值
    float* d_spike_;            // 设备端的当前脉冲
    float* d_last_spike_time_;  // 设备端的上次脉冲时间
    float* d_weight_;           // 设备端的权重
};