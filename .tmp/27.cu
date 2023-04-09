#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#define LENGTH 128

extern "C" void spmm_conv(void *input_data_t, void *output_data_t, void *kernel_ptr_t, void *kernel_map_t, void *kernel_offset_t, void *kernel_data_t, void *kernel_ptr_sparse_t, void *kernel_map_sparse_t); 



inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

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

	int output_x = i / (4 * 64);
	int output_y = i /64 % 4;

	int x1 = output_x * 1 * 4 * 512 * 64 + output_y * 1 * 512 * 64 + c;

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
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
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
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
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

	int output_idx = (output_x*4*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<32; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}
} 


__global__
void _spmm_conv_1(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 11) + blockIdx.x * (11 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 11;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (4 * 11);
	int output_y = i /11 % 4;

	int x1 = output_x * 1 * 4 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[11<<1];
#pragma unroll
	for (int i=0; i<(11<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[11];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<11; k++) {
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
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<11; k++) {
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
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*4*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<11; k++) {
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

	int output_x = i / (4 * 64);
	int output_y = i /64 % 4;

	int x1 = output_x * 1 * 4 * 512 * 64 + output_y * 1 * 512 * 64 + c;

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
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
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
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
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

	int output_idx = (output_x*4*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<32; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}
} 


__global__
void _spmm_conv_3(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 11) + blockIdx.x * (11 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 11;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (4 * 11);
	int output_y = i /11 % 4;

	int x1 = output_x * 1 * 4 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[11<<1];
#pragma unroll
	for (int i=0; i<(11<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[11];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<11; k++) {
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
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<11; k++) {
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
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*4*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<11; k++) {
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

	int output_x = i / (4 * 64);
	int output_y = i /64 % 4;

	int x1 = output_x * 1 * 4 * 512 * 64 + output_y * 1 * 512 * 64 + c;

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
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
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
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
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

	int output_idx = (output_x*4*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<32; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}
} 


__global__
void _spmm_conv_5(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 11) + blockIdx.x * (11 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 11;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (4 * 11);
	int output_y = i /11 % 4;

	int x1 = output_x * 1 * 4 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[11<<1];
#pragma unroll
	for (int i=0; i<(11<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[11];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<11; k++) {
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
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<11; k++) {
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
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<11; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*4*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<11; k++) {
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

	int output_x = i / (4 * 64);
	int output_y = i /64 % 4;

	int x1 = output_x * 1 * 4 * 512 * 64 + output_y * 1 * 512 * 64 + c;

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
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
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
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
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

	int output_idx = (output_x*4*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<32; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}
} 


__global__
void _spmm_conv_7(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 7) + blockIdx.x * (7 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 7;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (4 * 7);
	int output_y = i /7 % 4;

	int x1 = output_x * 1 * 4 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[7<<1];
#pragma unroll
	for (int i=0; i<(7<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[7];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<7; k++) {
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
		for (int k=0; k<7; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<7; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<7; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<7; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<7; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<7; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<7; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<7; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<7; k++) {
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
		for (int k=0; k<7; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<7; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<7; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<7; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<7; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<7; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<7; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*4*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<7; k++) {
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

	int output_x = i / (4 * 32);
	int output_y = i /32 % 4;

	int x1 = output_x * 1 * 4 * 512 * 64 + output_y * 1 * 512 * 64 + c;

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
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
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
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
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

	int output_idx = (output_x*4*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<32; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}
} 


__global__
void _spmm_conv_9(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 22) + blockIdx.x * (22 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 22;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (4 * 22);
	int output_y = i /22 % 4;

	int x1 = output_x * 1 * 4 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[22<<1];
#pragma unroll
	for (int i=0; i<(22<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[22];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<22; k++) {
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
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<22; k++) {
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
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*4*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<22; k++) {
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

	int output_x = i / (4 * 32);
	int output_y = i /32 % 4;

	int x1 = output_x * 1 * 4 * 512 * 64 + output_y * 1 * 512 * 64 + c;

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
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
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
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
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

	int output_idx = (output_x*4*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<32; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}
} 


__global__
void _spmm_conv_11(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 22) + blockIdx.x * (22 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 22;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (4 * 22);
	int output_y = i /22 % 4;

	int x1 = output_x * 1 * 4 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[22<<1];
#pragma unroll
	for (int i=0; i<(22<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[22];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<22; k++) {
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
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<22; k++) {
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
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*4*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<22; k++) {
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

	int output_x = i / (4 * 32);
	int output_y = i /32 % 4;

	int x1 = output_x * 1 * 4 * 512 * 64 + output_y * 1 * 512 * 64 + c;

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
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
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
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
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

	int output_idx = (output_x*4*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<32; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}
} 


__global__
void _spmm_conv_13(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 22) + blockIdx.x * (22 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 22;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (4 * 22);
	int output_y = i /22 % 4;

	int x1 = output_x * 1 * 4 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[22<<1];
#pragma unroll
	for (int i=0; i<(22<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[22];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<22; k++) {
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
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<22; k++) {
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
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*4*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<22; k++) {
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

	int output_x = i / (4 * 32);
	int output_y = i /32 % 4;

	int x1 = output_x * 1 * 4 * 512 * 64 + output_y * 1 * 512 * 64 + c;

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
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
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
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
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

	int output_idx = (output_x*4*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<32; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}
} 


__global__
void _spmm_conv_15(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 22) + blockIdx.x * (22 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 22;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (4 * 22);
	int output_y = i /22 % 4;

	int x1 = output_x * 1 * 4 * 512 * 64 + output_y * 1 * 512 * 64 + c;

	float res[22<<1];
#pragma unroll
	for (int i=0; i<(22<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[22];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+b] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+b] % 512 * 64;
#pragma unroll
				for (int k=0; k<22; k++) {
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
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (512 * 3)  *4 * 512 * 64 + kernel_offset[threadIdx.x+interm1] / 512 % 3 * 512 * 64   + kernel_offset[threadIdx.x+interm1] % 512 * 64;
#pragma unroll
			for (int k=0; k<22; k++) {
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
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*4*512+output_y*512)*64 + c;
#pragma unroll
	for (int k=0; k<22; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}
} 




void spmm_conv(void *input_data_t, void *output_data_t, void *kernel_ptr_t, void *kernel_map_t, void *kernel_offset_t, void *kernel_data_t, void *kernel_ptr_sparse_t, void *kernel_map_sparse_t) {
	float *input_data = (float *)input_data_t;
	float *output_data = (float *)output_data_t;
	int *kernel_ptr = (int *)kernel_ptr_t;
	int *kernel_map = (int *)kernel_map_t;
	int *kernel_offset = (int *)kernel_offset_t;
	float *kernel_data = (float *)kernel_data_t;
	int *kernel_ptr_sparse = (int *)kernel_ptr_sparse_t;
	int *kernel_map_sparse = (int *)kernel_map_sparse_t;

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
	
	checkCuda(cudaEventCreate(&event1));
	checkCuda(cudaEventCreate(&event2));

	//checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaEventRecord(event1, 0));

	cudaStreamCreate(&stream_0);
dim3 nblocks_0(8, 1);
dim3 nthreads_0(32, 4);
_spmm_conv_0<<<nblocks_0, nthreads_0, 0, stream_0>>>(input_data, output_data, 0, 65, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_1);
dim3 nblocks_1(4, 1);
dim3 nthreads_1(32, 4);
_spmm_conv_1<<<nblocks_1, nthreads_1, 0, stream_1>>>(input_data, output_data, 65, 77, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_2);
dim3 nblocks_2(8, 1);
dim3 nthreads_2(32, 4);
_spmm_conv_2<<<nblocks_2, nthreads_2, 0, stream_2>>>(input_data, output_data, 77, 142, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_3);
dim3 nblocks_3(4, 1);
dim3 nthreads_3(32, 4);
_spmm_conv_3<<<nblocks_3, nthreads_3, 0, stream_3>>>(input_data, output_data, 142, 154, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_4);
dim3 nblocks_4(8, 1);
dim3 nthreads_4(32, 4);
_spmm_conv_4<<<nblocks_4, nthreads_4, 0, stream_4>>>(input_data, output_data, 154, 219, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_5);
dim3 nblocks_5(4, 1);
dim3 nthreads_5(32, 4);
_spmm_conv_5<<<nblocks_5, nthreads_5, 0, stream_5>>>(input_data, output_data, 219, 231, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_6);
dim3 nblocks_6(8, 1);
dim3 nthreads_6(32, 4);
_spmm_conv_6<<<nblocks_6, nthreads_6, 0, stream_6>>>(input_data, output_data, 231, 296, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_7);
dim3 nblocks_7(4, 1);
dim3 nthreads_7(32, 4);
_spmm_conv_7<<<nblocks_7, nthreads_7, 0, stream_7>>>(input_data, output_data, 296, 304, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_8);
dim3 nblocks_8(4, 1);
dim3 nthreads_8(32, 4);
_spmm_conv_8<<<nblocks_8, nthreads_8, 0, stream_8>>>(input_data, output_data, 304, 337, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_9);
dim3 nblocks_9(4, 1);
dim3 nthreads_9(32, 4);
_spmm_conv_9<<<nblocks_9, nthreads_9, 0, stream_9>>>(input_data, output_data, 337, 360, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_10);
dim3 nblocks_10(4, 1);
dim3 nthreads_10(32, 4);
_spmm_conv_10<<<nblocks_10, nthreads_10, 0, stream_10>>>(input_data, output_data, 360, 393, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_11);
dim3 nblocks_11(4, 1);
dim3 nthreads_11(32, 4);
_spmm_conv_11<<<nblocks_11, nthreads_11, 0, stream_11>>>(input_data, output_data, 393, 416, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_12);
dim3 nblocks_12(4, 1);
dim3 nthreads_12(32, 4);
_spmm_conv_12<<<nblocks_12, nthreads_12, 0, stream_12>>>(input_data, output_data, 416, 449, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_13);
dim3 nblocks_13(4, 1);
dim3 nthreads_13(32, 4);
_spmm_conv_13<<<nblocks_13, nthreads_13, 0, stream_13>>>(input_data, output_data, 449, 472, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_14);
dim3 nblocks_14(4, 1);
dim3 nthreads_14(32, 4);
_spmm_conv_14<<<nblocks_14, nthreads_14, 0, stream_14>>>(input_data, output_data, 472, 505, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_15);
dim3 nblocks_15(4, 1);
dim3 nthreads_15(32, 4);
_spmm_conv_15<<<nblocks_15, nthreads_15, 0, stream_15>>>(input_data, output_data, 505, 528, kernel_ptr, kernel_map, kernel_offset, kernel_data);




	checkCuda(cudaEventRecord(event2, 0));
	checkCuda(cudaEventSynchronize(event2));
	checkCuda(cudaEventElapsedTime(&time, event1, event2));

	printf("execution time: %f\n", time);

	cudaStreamDestroy(stream_0);
cudaStreamDestroy(stream_1);
cudaStreamDestroy(stream_2);
cudaStreamDestroy(stream_3);
cudaStreamDestroy(stream_4);
cudaStreamDestroy(stream_5);
cudaStreamDestroy(stream_6);
cudaStreamDestroy(stream_7);
cudaStreamDestroy(stream_8);
cudaStreamDestroy(stream_9);
cudaStreamDestroy(stream_10);
cudaStreamDestroy(stream_11);
cudaStreamDestroy(stream_12);
cudaStreamDestroy(stream_13);
cudaStreamDestroy(stream_14);
cudaStreamDestroy(stream_15);

}