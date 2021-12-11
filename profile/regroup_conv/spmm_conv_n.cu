__global__
void _spmm_conv_n(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * _NN) + blockIdx.x * (_NN << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % _NKERNEL;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (_OWIDTH * _NKERNEL);
	int output_y = i /_NKERNEL % _OWIDTH;

	int x1 = output_x * _STRIDE_HEIGHT * _INPUT_WIDTH * _INPUT_CHANNEL * _BATCH_SIZE + output_y * _STRIDE_WIDTH * _INPUT_CHANNEL * _BATCH_SIZE + c;

	float res[_NN<<1];
#pragma unroll
	for (int i=0; i<(_NN<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[_NN];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (_INPUT_CHANNEL * _KERNEL_WIDTH)  *_INPUT_WIDTH * _INPUT_CHANNEL * _BATCH_SIZE + kernel_offset[threadIdx.x+b] / _INPUT_CHANNEL % _KERNEL_WIDTH * _INPUT_CHANNEL * _BATCH_SIZE   + kernel_offset[threadIdx.x+b] % _INPUT_CHANNEL * _BATCH_SIZE;
#pragma unroll
				for (int k=0; k<_NN; k++) {
					kernel_value[k] = kernel_data[threadIdx.x+b+length*k];
				}
			}
		}

		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+1);
		int idx3 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+2);
		int idx4 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+3);
		int idx5 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+4);
		int idx6 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+5);
		int idx7 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+6);
		int idx8 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+7);
#pragma unroll
		for (int k=0; k<_NN; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<_NN; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<_NN; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<_NN; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<_NN; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<_NN; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<_NN; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<_NN; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (_INPUT_CHANNEL * _KERNEL_WIDTH)  *_INPUT_WIDTH * _INPUT_CHANNEL * _BATCH_SIZE + kernel_offset[threadIdx.x+interm1] / _INPUT_CHANNEL % _KERNEL_WIDTH * _INPUT_CHANNEL * _BATCH_SIZE   + kernel_offset[threadIdx.x+interm1] % _INPUT_CHANNEL * _BATCH_SIZE;
#pragma unroll
			for (int k=0; k<_NN; k++) {
				kernel_value[k] = kernel_data[threadIdx.x+interm1+length*k];
			}
		}
	}

	if (interm1 < interm2) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin+1);
		int idx3 = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin+2);
		int idx4 = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin+3);

#pragma unroll
		for (int k=0; k<_NN; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<_NN; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<_NN; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<_NN; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<_NN; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<_NN; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<_NN; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*_OWIDTH*_TOT_KERNEL+output_y*_TOT_KERNEL)*_BATCH_SIZE + c;
#pragma unroll
	for (int k=0; k<_NN; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*_BATCH_SIZE] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*_BATCH_SIZE+32] = res[(k<<1)+1];
	}

} 


