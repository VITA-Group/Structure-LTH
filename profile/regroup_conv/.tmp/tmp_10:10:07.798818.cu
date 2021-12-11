#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#define LENGTH 128

extern "C" void spmm_conv(void *input_data_t, void *output_data_t, void *kernel_ptr_t, void *kernel_map_t, void *kernel_offset_t, void *kernel_data_t); 




void spmm_conv(void *input_data_t, void *output_data_t, void *kernel_ptr_t, void *kernel_map_t, void *kernel_offset_t, void *kernel_data_t) {
	float *input_data = (float *)input_data_t;
	float *output_data = (float *)output_data_t;
	int *kernel_ptr = (int *)kernel_ptr_t;
	int *kernel_map = (int *)kernel_map_t;
	int *kernel_offset = (int *)kernel_offset_t;
	float *kernel_data = (float *)kernel_data_t;

	

	float time;
	cudaEvent_t event1, event2;
	cudaEventCreate(&event1);
	cudaEventCreate(&event2);

	cudaDeviceSynchronize();
	cudaEventRecord(event1, 0);

	



	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event1);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&time, event1, event2);

	printf("execution time: %f\n", time);
}


