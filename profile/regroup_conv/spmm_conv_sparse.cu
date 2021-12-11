__global__
void _spmm_conv_sparse(const float * __restrict__ input_data, float *output_data, const int * __restrict__ kernel_ptr_sparse, const int * __restrict__ kernel_map_sparse, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {

	__shared__ float kernel_value[2][LENGTH];
	__shared__ int kernel_off[2][LENGTH];
	
	int i = threadIdx.y + blockIdx.x * 2;
	int k = threadIdx.x + blockIdx.y * 64;
	int kernel_id = i % _NKERNEL;
	//printf("%d %d %d\n", i, k, kernel_id);
	int start = kernel_ptr_sparse[kernel_id];
	int end = kernel_ptr_sparse[kernel_id+1];
	//printf("%d %d\n", start, end);
	float res = 0.0f;
	float res2 = 0.0f;

	int output_x = i / (_OWIDTH * _NKERNEL);
	int output_y = i /_NKERNEL % _OWIDTH;


	int x = output_x * _STRIDE_HEIGHT * _INPUT_WIDTH * _INPUT_CHANNEL * _BATCH_SIZE + output_y * _STRIDE_WIDTH * _INPUT_CHANNEL * _BATCH_SIZE + k;

	for (int b=start; b<end; b+=LENGTH) {
		int length = LENGTH > end-b? end-b:LENGTH;

		for (int j=threadIdx.x; j<length; j+=blockDim.x) {
			kernel_off[threadIdx.y][j] =  kernel_offset[j+b] / (_INPUT_CHANNEL * _KERNEL_WIDTH)  *_INPUT_WIDTH * _INPUT_CHANNEL * _BATCH_SIZE + kernel_offset[j+b] / _INPUT_CHANNEL % _KERNEL_WIDTH * _INPUT_CHANNEL * _BATCH_SIZE   + kernel_offset[j+b] % _INPUT_CHANNEL * _BATCH_SIZE + x;
			kernel_value[threadIdx.y][j] = kernel_data[j+b];
		}
		__syncthreads();

		for (int j=0; j<((length >> 1) << 1); j+=2) {
			res += kernel_value[threadIdx.y][j] * input_data[kernel_off[threadIdx.y][j]];
			res2 += kernel_value[threadIdx.y][j] * input_data[kernel_off[threadIdx.y][j] + 32];
			res += kernel_value[threadIdx.y][j+1] * input_data[kernel_off[threadIdx.y][j+1]];
			res2 += kernel_value[threadIdx.y][j+1] * input_data[kernel_off[threadIdx.y][j+1] + 32];
		}

		if (length & 1) {
			res += kernel_value[threadIdx.y][length-1] * input_data[kernel_off[threadIdx.y][length-1]];
			res2 += kernel_value[threadIdx.y][length-1] * input_data[kernel_off[threadIdx.y][length-1] + 32];
		}
	}
	int output_idx = (output_x*_OWIDTH*_TOT_KERNEL+output_y*_TOT_KERNEL + kernel_map_sparse[kernel_id])*_BATCH_SIZE + k;
	output_data[output_idx] = res;
	output_data[output_idx+32] = res2;
}
