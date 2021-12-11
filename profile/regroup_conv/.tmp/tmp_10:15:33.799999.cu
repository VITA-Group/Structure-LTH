#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#define LENGTH 128

extern "C" void spmm_conv(void *input_data_t, void *output_data_t, void *kernel_ptr_t, void *kernel_map_t, void *kernel_offset_t, void *kernel_data_t); 


__global__
void _spmm_conv_0(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 32) + blockIdx.x * (32 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 64;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (65 * 64);
	int output_y = i /64 % 65;

	int x1 = output_x * 1 * 65 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[32<<1];
#pragma unroll
	for (int i=0; i<(32<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[32];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<32; k++) {
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
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<32; k++) {
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
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*65*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<32; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_1(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 21) + blockIdx.x * (21 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 21;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (65 * 21);
	int output_y = i /21 % 65;

	int x1 = output_x * 1 * 65 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[21<<1];
#pragma unroll
	for (int i=0; i<(21<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[21];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<21; k++) {
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
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<21; k++) {
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
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*65*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<21; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_2(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 32) + blockIdx.x * (32 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 64;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (65 * 64);
	int output_y = i /64 % 65;

	int x1 = output_x * 1 * 65 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[32<<1];
#pragma unroll
	for (int i=0; i<(32<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[32];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<32; k++) {
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
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<32; k++) {
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
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*65*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<32; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_3(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 21) + blockIdx.x * (21 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 21;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (65 * 21);
	int output_y = i /21 % 65;

	int x1 = output_x * 1 * 65 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[21<<1];
#pragma unroll
	for (int i=0; i<(21<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[21];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<21; k++) {
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
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<21; k++) {
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
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*65*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<21; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_4(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 32) + blockIdx.x * (32 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 64;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (65 * 64);
	int output_y = i /64 % 65;

	int x1 = output_x * 1 * 65 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[32<<1];
#pragma unroll
	for (int i=0; i<(32<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[32];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<32; k++) {
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
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<32; k++) {
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
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*65*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<32; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_5(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 21) + blockIdx.x * (21 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 21;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (65 * 21);
	int output_y = i /21 % 65;

	int x1 = output_x * 1 * 65 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[21<<1];
#pragma unroll
	for (int i=0; i<(21<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[21];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<21; k++) {
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
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<21; k++) {
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
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<21; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*65*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<21; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_6(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 32) + blockIdx.x * (32 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 64;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (65 * 64);
	int output_y = i /64 % 65;

	int x1 = output_x * 1 * 65 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[32<<1];
#pragma unroll
	for (int i=0; i<(32<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[32];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<32; k++) {
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
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<32; k++) {
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
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*65*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<32; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_7(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 1) + blockIdx.x * (1 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 1;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (65 * 1);
	int output_y = i /1 % 65;

	int x1 = output_x * 1 * 65 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[1<<1];
#pragma unroll
	for (int i=0; i<(1<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[1];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<1; k++) {
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
		for (int k=0; k<1; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<1; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<1; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<1; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<1; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<1; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<1; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<1; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<1; k++) {
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
		for (int k=0; k<1; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<1; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<1; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<1; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<1; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<1; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<1; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*65*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<1; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_8(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 32) + blockIdx.x * (32 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 32;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (65 * 32);
	int output_y = i /32 % 65;

	int x1 = output_x * 1 * 65 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[32<<1];
#pragma unroll
	for (int i=0; i<(32<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[32];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<32; k++) {
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
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<32; k++) {
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
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*65*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<32; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_9(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 16) + blockIdx.x * (16 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 16;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (65 * 16);
	int output_y = i /16 % 65;

	int x1 = output_x * 1 * 65 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[16<<1];
#pragma unroll
	for (int i=0; i<(16<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[16];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<16; k++) {
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
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<16; k++) {
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
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*65*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<16; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_10(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 32) + blockIdx.x * (32 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 32;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (65 * 32);
	int output_y = i /32 % 65;

	int x1 = output_x * 1 * 65 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[32<<1];
#pragma unroll
	for (int i=0; i<(32<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[32];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<32; k++) {
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
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<32; k++) {
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
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*65*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<32; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_11(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 16) + blockIdx.x * (16 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 16;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (65 * 16);
	int output_y = i /16 % 65;

	int x1 = output_x * 1 * 65 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[16<<1];
#pragma unroll
	for (int i=0; i<(16<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[16];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<16; k++) {
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
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<16; k++) {
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
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*65*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<16; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_12(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 32) + blockIdx.x * (32 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 32;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (65 * 32);
	int output_y = i /32 % 65;

	int x1 = output_x * 1 * 65 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[32<<1];
#pragma unroll
	for (int i=0; i<(32<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[32];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<32; k++) {
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
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<32; k++) {
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
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*65*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<32; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_13(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 16) + blockIdx.x * (16 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 16;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (65 * 16);
	int output_y = i /16 % 65;

	int x1 = output_x * 1 * 65 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[16<<1];
#pragma unroll
	for (int i=0; i<(16<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[16];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<16; k++) {
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
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<16; k++) {
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
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*65*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<16; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_14(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 32) + blockIdx.x * (32 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 32;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (65 * 32);
	int output_y = i /32 % 65;

	int x1 = output_x * 1 * 65 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[32<<1];
#pragma unroll
	for (int i=0; i<(32<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[32];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<32; k++) {
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
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<32; k++) {
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
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<32; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*65*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<32; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_15(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 16) + blockIdx.x * (16 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 16;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (65 * 16);
	int output_y = i /16 % 65;

	int x1 = output_x * 1 * 65 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[16<<1];
#pragma unroll
	for (int i=0; i<(16<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[16];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<16; k++) {
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
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *65 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<16; k++) {
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
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<16; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*65*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<16; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 




void spmm_conv(void *input_data_t, void *output_data_t, void *kernel_ptr_t, void *kernel_map_t, void *kernel_offset_t, void *kernel_data_t) {
	float *input_data = (float *)input_data_t;
	float *output_data = (float *)output_data_t;
	int *kernel_ptr = (int *)kernel_ptr_t;
	int *kernel_map = (int *)kernel_map_t;
	int *kernel_offset = (int *)kernel_offset_t;
	float *kernel_data = (float *)kernel_data_t;

	cudaStream_t stream_0;
cudaStream_t stream_1;
cudaStream_t stream_2;
cudaStream_t stream_3;
cudaStream_t stream_4;
cudaStream_t stream_5;
cudaStream_t stream_6;
cudaStream_t stream_7;
cudaStream_t stream_8;
cudaStream_t stream_9;
cudaStream_t stream_10;
cudaStream_t stream_11;
cudaStream_t stream_12;
cudaStream_t stream_13;
cudaStream_t stream_14;
cudaStream_t stream_15;


	float time;
	cudaEvent_t event1, event2;
	cudaEventCreate(&event1);
	cudaEventCreate(&event2);

	cudaDeviceSynchronize();
	cudaEventRecord(event1, 0);

	cudaStreamCreate(&stream_0);
dim3 nblocks_0(2112, 1);
dim3 nthreads_0(32, 4);
_spmm_conv_0<<<nblocks_0, nthreads_0, 0, stream_0>>>(input_data, output_data, 0, 65, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_1);
dim3 nblocks_1(1056, 1);
dim3 nthreads_1(32, 4);
_spmm_conv_1<<<nblocks_1, nthreads_1, 0, stream_1>>>(input_data, output_data, 65, 87, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_2);
dim3 nblocks_2(2112, 1);
dim3 nthreads_2(32, 4);
_spmm_conv_2<<<nblocks_2, nthreads_2, 0, stream_2>>>(input_data, output_data, 87, 152, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_3);
dim3 nblocks_3(1056, 1);
dim3 nthreads_3(32, 4);
_spmm_conv_3<<<nblocks_3, nthreads_3, 0, stream_3>>>(input_data, output_data, 152, 174, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_4);
dim3 nblocks_4(2112, 1);
dim3 nthreads_4(32, 4);
_spmm_conv_4<<<nblocks_4, nthreads_4, 0, stream_4>>>(input_data, output_data, 174, 239, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_5);
dim3 nblocks_5(1056, 1);
dim3 nthreads_5(32, 4);
_spmm_conv_5<<<nblocks_5, nthreads_5, 0, stream_5>>>(input_data, output_data, 239, 261, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_6);
dim3 nblocks_6(2112, 1);
dim3 nthreads_6(32, 4);
_spmm_conv_6<<<nblocks_6, nthreads_6, 0, stream_6>>>(input_data, output_data, 261, 326, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_7);
dim3 nblocks_7(1056, 1);
dim3 nthreads_7(32, 4);
_spmm_conv_7<<<nblocks_7, nthreads_7, 0, stream_7>>>(input_data, output_data, 326, 328, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_8);
dim3 nblocks_8(1056, 1);
dim3 nthreads_8(32, 4);
_spmm_conv_8<<<nblocks_8, nthreads_8, 0, stream_8>>>(input_data, output_data, 328, 361, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_9);
dim3 nblocks_9(1056, 1);
dim3 nthreads_9(32, 4);
_spmm_conv_9<<<nblocks_9, nthreads_9, 0, stream_9>>>(input_data, output_data, 361, 378, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_10);
dim3 nblocks_10(1056, 1);
dim3 nthreads_10(32, 4);
_spmm_conv_10<<<nblocks_10, nthreads_10, 0, stream_10>>>(input_data, output_data, 378, 411, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_11);
dim3 nblocks_11(1056, 1);
dim3 nthreads_11(32, 4);
_spmm_conv_11<<<nblocks_11, nthreads_11, 0, stream_11>>>(input_data, output_data, 411, 428, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_12);
dim3 nblocks_12(1056, 1);
dim3 nthreads_12(32, 4);
_spmm_conv_12<<<nblocks_12, nthreads_12, 0, stream_12>>>(input_data, output_data, 428, 461, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_13);
dim3 nblocks_13(1056, 1);
dim3 nthreads_13(32, 4);
_spmm_conv_13<<<nblocks_13, nthreads_13, 0, stream_13>>>(input_data, output_data, 461, 478, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_14);
dim3 nblocks_14(1056, 1);
dim3 nthreads_14(32, 4);
_spmm_conv_14<<<nblocks_14, nthreads_14, 0, stream_14>>>(input_data, output_data, 478, 511, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_15);
dim3 nblocks_15(1056, 1);
dim3 nthreads_15(32, 4);
_spmm_conv_15<<<nblocks_15, nthreads_15, 0, stream_15>>>(input_data, output_data, 511, 528, kernel_ptr, kernel_map, kernel_offset, kernel_data);




	cudaEventRecord(event2, 0);
	cudaEventSynchronize(event1);
	cudaEventSynchronize(event2);
	cudaEventElapsedTime(&time, event1, event2);

	printf("execution time: %f\n", time);
}


