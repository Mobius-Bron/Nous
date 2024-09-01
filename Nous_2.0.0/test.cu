#include<stdio.h>
#include<bits/stdc++.h>
#include<nros.h>

int main(void){
    printf("code runnig\n");

    int input_features = 100;
    int output_features = 50;
    Linear linear(input_features, output_features);

    // 示例输入数据
    float* h_input_data = memsetHost(input_features*sizeof(float));
    float* d_input_data = memsetDevice(input_features*sizeof(float));
    // 输出数据
    float* h_output_data = memsetHost(output_features*sizeof(float));
    float* d_output_data = memsetDevice(output_features*sizeof(float));
    float* d_relu_output = memsetDevice(output_features*sizeof(float));
    for (int i = 0; i < input_features; ++i) {
        h_input_data[i] += 1.0f;
    }

    for (int i = 0; i < input_features; ++i) {
        std::cout << "\t" << h_input_data[i];
    }printf("\n");

    dim3 threadsPerBlock(256);
    dim3 numBlocks((output_features + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // 将输入数据从主机复制到设备
    cudaMemcpy(d_input_data, h_input_data, input_features * sizeof(float), cudaMemcpyHostToDevice);

    // 前向传播
    linear.forward(d_input_data, d_output_data);

    // 调用 ReLU 激活内核
    actvation_fun::ReLU<<<numBlocks, threadsPerBlock>>>(d_output_data, d_relu_output, output_features);

    // 将激活后的输出从设备复制回主机
    cudaMemcpy(h_output_data, d_output_data, output_features * sizeof(float), cudaMemcpyDeviceToHost);

    // 同步权重和偏置
    linear.sync_weights_to_host();

    // 打印结果
    for (int i = 0; i < output_features; ++i) {
        std::cout << "Output " << i << ": " << h_output_data[i] << std::endl;
    }

    // 释放内存
    cudaFreeHost(h_input_data);
    cudaFree(d_input_data);
    cudaFreeHost(h_output_data);
    cudaFree(d_output_data);
    cudaFree(d_relu_output);

    printf("code end\n");
    return 0;
}