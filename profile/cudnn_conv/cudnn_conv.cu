#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include <iostream>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }




int main(int argc, const char* argv[]) {

	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);

	cudnnTensorDescriptor_t input_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
				/*format=*/CUDNN_TENSOR_NHWC,
				/*dataType=*/CUDNN_DATA_FLOAT,
				/*batch_size=*/S_batch_size,
				/*channels=*/S_channels,
				/*image_height=*/S_input_height,
				/*image_width=*/S_input_width));

	cudnnFilterDescriptor_t kernel_descriptor;
	checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
	checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
				/*dataType=*/CUDNN_DATA_FLOAT,
				/*format=*/CUDNN_TENSOR_NCHW,
				/*out_channels=*/S_kernels,
				/*in_channels=*/S_channels,
				/*kernel_height=*/S_kernel_height,
				/*kernel_width=*/S_kernel_width));

	cudnnConvolutionDescriptor_t convolution_descriptor;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
	checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
				/*pad_height=*/S_padding_height,
				/*pad_width=*/S_padding_width,
				/*vertical_stride=*/S_vertical_stride,
				/*horizontal_stride=*/S_horizontal_stride,
				/*dilation_height=*/S_dilation_height,
				/*dilation_width=*/S_dilation_width,
				/*mode=*/CUDNN_CROSS_CORRELATION,
				/*computeType=*/CUDNN_DATA_FLOAT));

	int batch_size_output{0}, channels{0}, height{0}, width{0};
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
				input_descriptor,
				kernel_descriptor,
				&batch_size_output,
				&channels,
				&height,
				&width));

	assert(S_batch_size == batch_size_output && channels == S_kernels);

	cudnnTensorDescriptor_t output_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
				/*format=*/CUDNN_TENSOR_NHWC,
				/*dataType=*/CUDNN_DATA_FLOAT,
				/*batch_size=*/S_batch_size,
				/*channels=*/channels,
				/*image_height=*/height,
				/*image_width=*/width));

	cudnnConvolutionFwdAlgo_t convolution_algorithm;
	checkCUDNN(
			cudnnGetConvolutionForwardAlgorithm(cudnn,
				input_descriptor,
				kernel_descriptor,
				convolution_descriptor,
				output_descriptor,
				//CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
				CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
				/*memoryLimitInBytes=*/0,
				&convolution_algorithm));

	//std::cout << convolution_algorithm << "   ";

	float* d_input{nullptr};
	cudaMalloc(&d_input, S_batch_size * S_channels * S_input_height * S_input_width * sizeof(float));


	float* d_output{nullptr};
	cudaMalloc(&d_output, S_batch_size * channels * height * width * sizeof(float));

	float* d_kernel{nullptr};
	cudaMalloc(&d_kernel, S_kernels * S_channels * S_kernel_height * S_kernel_width);

	const float alpha = 1.0f, beta = 0.0f;

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	checkCUDNN(cudnnConvolutionForward(cudnn,
				&alpha,
				input_descriptor,
				d_input,
				kernel_descriptor,
				d_kernel,
				convolution_descriptor,
				convolution_algorithm,
				NULL,
				0,
				&beta,
				output_descriptor,
				d_output));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	//std::cerr << "time to conv: " << time << " ms" << std::endl;
	std::cout << "execution time: " << time << std::endl;


	cudaFree(d_kernel);
	cudaFree(d_input);
	cudaFree(d_output);
	//  cudaFree(d_workspace);

	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);

	cudnnDestroy(cudnn);
}
