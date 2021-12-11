import argparse
import torch as th
import re
import io
import datetime
import os
import ctypes
import heapq
import sys
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--input_height', type=int, default=127)
parser.add_argument('--input_width', type=int, default=127)
parser.add_argument('--input_channel', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nkernel', type=int, default=64)
parser.add_argument('--kernel_height', type=int, default=3)
parser.add_argument('--kernel_width', type=int, default=3)
parser.add_argument('--vertical_stride', type=int, default=1)
parser.add_argument('--horizontal_stride', type=int, default=1)
parser.add_argument('--vertical_dilation', type=int, default=1)
parser.add_argument('--horizontal_dilation', type=int, default=1)
parser.add_argument('--vertical_padding', type=int, default=1)
parser.add_argument('--horizontal_padding', type=int, default=1)
parser.add_argument('-f', '--kernel_file', default=None)
parser.add_argument('--model_name', default=None)
parser.add_argument('--t1', type=float, default=1.5)
parser.add_argument('--cn', type=int, default=8)

parser.add_argument("--cuda_version", type=str, default="10.1")

args = parser.parse_args()
input_height = args.input_height
input_width = args.input_width
input_channel = args.input_channel
batch_size = args.batch_size
nkernel = args.nkernel
kernel_height = args.kernel_height
kernel_width = args.kernel_width
vertical_stride = args.vertical_stride
horizontal_stride = args.horizontal_stride
vertical_dilation = args.vertical_dilation
horizontal_dilation = args.horizontal_dilation
vertical_padding = args.vertical_padding
horizontal_padding = args.horizontal_padding
model_name = args.model_name

kernel_file = args.kernel_file

t1 = args.t1


nn = 32
B2 = 16

cn = args.cn






th.manual_seed(12345)



def get_sim(a, b):
	s = 0
	for i in range(len(a)):
		if a[i] !=0 and b[i] != 0:
			s += 1
	return s

def extract_dense(sparse_kernel):
	print((sparse_kernel.abs() != 0).sum().float() / sparse_kernel.numel())
	nrows = sparse_kernel.shape[0]
	ncols = sparse_kernel.shape[1]

	nonempty_rows = []
	for i in range(nrows):
		nz = 0
		for j in range(ncols):
			if sparse_kernel[i, j] != 0:
				nonempty_rows.append(i)
				break
	#print (nrows, len(nonempty_rows))

	nonempty_cols = []
	for j in range(ncols):
		nz = 0
		for i in nonempty_rows:
			if sparse_kernel[i, j] != 0:
				nonempty_cols.append(j)
				break
	#print (ncols, len(nonempty_cols))

	f = open('hypergraph.txt', 'w')
	f.write(str(len(nonempty_cols))+' '+str(len(nonempty_rows))+'\n')
	for j in range(len(nonempty_cols)):
		for i in range(len(nonempty_rows)):
			if sparse_kernel[nonempty_rows[i], nonempty_cols[j]] != 0:
				f.write(str(i+1)+' ')
		f.write('\n')

	f.close()
	os.system(f'./shmetis hypergraph.txt {cn} 10')

	f = open(f'hypergraph.txt.part.{cn}', 'r')
	clusters = {}
	s = f.readlines()
	#print (len(s))
	assert (len(s) == len(nonempty_rows))
	for i in range(len(s)):
		t = int(s[i].strip())
		if t not in clusters:
			clusters[t] = []
		clusters[t].append(i)
	f.close()

	os.system('rm hypergraph*')


	clusters = [clusters[c] for c in clusters]
	clusters.sort(key=lambda x:len(x), reverse=True)

	#print (clusters)
	
	

	blocks = []

	for r in clusters:
		nnz_cols = [0] * ncols
		for i in range(ncols):
			s = 0
			for rr in r:
				if sparse_kernel[nonempty_rows[rr],i] != 0:
					s += 1
			nnz_cols[i] = s
		cc = sorted(list(range(ncols)), key=lambda x:nnz_cols[x], reverse=True)

		nnz_rows = [0] * len(r)

		for i in range(len(r)):
			for j in range(ncols):
				if sparse_kernel[nonempty_rows[r[i]], j] != 0:
					nnz_rows[i] += 1


		for i in range(1, ncols):
			dense_cols = cc[:i]

			flag = False
			for j in range(len(r)):
				if sparse_kernel[nonempty_rows[r[j]], i] != 0:
					nnz_rows[j] -= 1
				if i <= t1*nnz_rows[j]:
					flag = True
					break
			
			if flag == False:
				dense_rows = [nonempty_rows[i] for i in r]
				#print (len(dense_rows), len(dense_cols))
				if len(dense_rows) > nn:
					dense_rows_1 = dense_rows[:len(dense_rows)//nn*nn]
					dense_rows_2 = dense_rows[len(dense_rows)//nn*nn:]
					blocks.append((dense_rows_1, dense_cols))
					blocks.append((dense_rows_2, dense_cols))
				elif len(dense_rows) > B2:
					blocks.append((dense_rows, dense_cols))
				break

	return blocks

	
block_ptr = [0]
kernel_ptr = []
kernel_map = []
kernel_offset = []
kernel_value = []

sparse_kernel = th.load(kernel_file)
kernel_shape = sparse_kernel.shape
nkernel = kernel_shape[0]
input_channel = kernel_shape[1]
kernel_height = kernel_shape[2]
kernel_width = kernel_shape[3]
sparse_kernel = sparse_kernel.view(kernel_shape[0], kernel_shape[1] * kernel_shape[2] * kernel_shape[3])
new_kernel = sparse_kernel.clone()




print (f'nkernel: {nkernel}, input_channel: {input_channel}, kernel_height: {kernel_height}, kernel_width: {kernel_width}')
sys.stdout.flush()


nnz = 0
for a in sparse_kernel:
	for b in a:
		if b != 0:
			nnz += 1
	
sparse_nnz = nnz


try:
	blocks = extract_dense(sparse_kernel)
except:
	blocks = []
#if len(blocks) == 0:
	#blocks = [(list(range(sparse_kernel.shape[0])), list(range(sparse_kernel.shape[1])))]
for b in blocks:
	kernel_ptr.append(len(kernel_offset))
	for r in b[0]:
		kernel_offset.extend(b[1])
		kernel_value.extend(sparse_kernel[r,b[1]].tolist())
		kernel_ptr.append(len(kernel_offset))
		kernel_map.append(r)
		for c in b[1]:
			if (sparse_kernel[r,c] != 0):
				sparse_kernel[r, c] = 0
				sparse_nnz -= 1
			else:
				new_kernel[r, c] = np.random.rand() 
					
	kernel_map.append(-1)
	assert (len(kernel_ptr) == len(kernel_map))
	block_ptr.append(len(kernel_ptr))

'''
blocks = extract_dense(sparse_kernel)

for b in blocks:
	kernel_ptr.append(len(kernel_offset))
	for r in b[0]:
		kernel_offset.extend(b[1])
		kernel_value.extend(sparse_kernel[r,b[1]].tolist())
		kernel_ptr.append(len(kernel_offset))
		kernel_map.append(r)
		for c in b[1]:
			if (sparse_kernel[r,c] != 0):
				sparse_kernel[r, c] = 0
				sparse_nnz -= 1
			else:
				new_kernel[r, c] = np.random.rand() 
					
	kernel_map.append(-1)
	assert (len(kernel_ptr) == len(kernel_map))
	block_ptr.append(len(kernel_ptr))
'''
kernel_ptr_sparse = []
kernel_map_sparse = []
nrows = sparse_kernel.shape[0]
ncols = sparse_kernel.shape[1]
kernel_ptr_sparse.append(len(kernel_offset))

for i in range(nrows):
  empty = True
  for j in range(ncols):
    if sparse_kernel[i,j]	!= 0:
      kernel_offset.append(j)
      kernel_value.append(sparse_kernel[i,j])
      empty = False
  if not empty:
    kernel_ptr_sparse.append(len(kernel_offset))
    kernel_map_sparse.append(i)

print (f'remaining_nnz: {sparse_nnz}, original_nnz: {nnz}')
sys.stdout.flush()



block_ptr = th.IntTensor(block_ptr)
kernel_ptr = th.IntTensor(kernel_ptr)
kernel_map = th.IntTensor(kernel_map)
kernel_offset = th.IntTensor(kernel_offset)
kernel_value = th.FloatTensor(kernel_value)
kernel_ptr_sparse = th.IntTensor(kernel_ptr_sparse)
kernel_map_sparse = th.IntTensor(kernel_map_sparse)

'''
if nkernel >= 512 or input_channel >= 512:
	input_height = 65
	input_width = 65

'''

assert (kernel_height % 2 == 1 and kernel_width % 2 == 1)
tmp_kernel_height = kernel_height + (kernel_height - 1) * (vertical_dilation - 1)
tmp_kernel_width = kernel_width + (kernel_width - 1) * (horizontal_dilation - 1)
tmp = input_height - tmp_kernel_height + 2 * vertical_padding
assert (tmp % vertical_stride == 0)
output_height = tmp // vertical_stride + 1
tmp = input_width - tmp_kernel_width + 2 * horizontal_padding
assert (tmp % horizontal_stride == 0)
output_width = tmp // horizontal_stride + 1

output_channels = nkernel



input_tensor = th.rand(input_height+2*vertical_padding, input_width+2*horizontal_padding, input_channel, batch_size)
output_tensor = th.zeros(output_height, output_width, output_channels, batch_size)


f = open('spmm_conv_n.cu', 'r')
code_n = f.read()
f.close()


f = open('test_template.cu', 'r')
code_template = f.read()
f.close()

f = open('spmm_conv_sparse.cu', 'r')
code_s = f.read()
f.close()

code_kernel = ''
call_kernel = ''
code_stream_decl = ''

for i in range(len(block_ptr)-1):
	block_kernel_size = block_ptr[i+1] - block_ptr[i] - 1
	block_kernel_size = block_kernel_size.item()
	if block_kernel_size  < 1:
		continue

	code_stream_decl += f'cudaStream_t stream_{i};\n'

	if block_kernel_size % nn == 0:
		code_kernel += code_n.replace('_OWIDTH', str(output_width)).replace('_OHEIGHT', str(output_height)).replace('_OCHANNEL', str(output_channels)).replace('_STRIDE_HEIGHT', str(vertical_stride)).replace('_STRIDE_WIDTH', str(horizontal_stride)).replace('_PADDING_HEIGHT', str(vertical_padding)).replace('_PADDING_WIDTH', str(horizontal_padding)).replace('_KERNEL_HEIGHT', str(kernel_height)).replace('_KERNEL_WIDTH', str(kernel_width)).replace('_INPUT_HEIGHT', str(input_height)).replace('_INPUT_WIDTH', str(input_width)).replace('_DIALATION_HEIGHT', str(vertical_dilation)).replace('_DIALATION_WIDTH', str(horizontal_dilation)).replace('_INPUT_CHANNEL', str(input_channel)).replace('_BATCH_SIZE', str(batch_size)).replace('_NN', str(nn)).replace('_NKERNEL', str(block_kernel_size)).replace('_TOT_KERNEL', str(output_channels)).replace('_spmm_conv_n', f'_spmm_conv_{i}')
		call_kernel += f'cudaStreamCreate(&stream_{i});'
		call_kernel += f'\ndim3 nblocks_{i}({output_width*output_height*block_kernel_size//(4*nn)}, {batch_size // 64});\ndim3 nthreads_{i}(32, 4);\n_spmm_conv_{i}<<<nblocks_{i}, nthreads_{i}, 0, stream_{i}>>>(input_data, output_data, {block_ptr[i]}, {block_ptr[i+1]}, kernel_ptr, kernel_map, kernel_offset, kernel_data);\n'
	else:
		assert (block_kernel_size < nn)
		code_kernel += code_n.replace('_OWIDTH', str(output_width)).replace('_OHEIGHT', str(output_height)).replace('_OCHANNEL', str(output_channels)).replace('_STRIDE_HEIGHT', str(vertical_stride)).replace('_STRIDE_WIDTH', str(horizontal_stride)).replace('_PADDING_HEIGHT', str(vertical_padding)).replace('_PADDING_WIDTH', str(horizontal_padding)).replace('_KERNEL_HEIGHT', str(kernel_height)).replace('_KERNEL_WIDTH', str(kernel_width)).replace('_INPUT_HEIGHT', str(input_height)).replace('_INPUT_WIDTH', str(input_width)).replace('_DIALATION_HEIGHT', str(vertical_dilation)).replace('_DIALATION_WIDTH', str(horizontal_dilation)).replace('_INPUT_CHANNEL', str(input_channel)).replace('_BATCH_SIZE', str(batch_size)).replace('_NN', str(block_kernel_size)).replace('_NKERNEL', str(block_kernel_size)).replace('_TOT_KERNEL', str(output_channels)).replace('_spmm_conv_n', f'_spmm_conv_{i}')
		call_kernel += f'cudaStreamCreate(&stream_{i});'
		call_kernel += f'\ndim3 nblocks_{i}({output_width*output_height//4}, {batch_size // 64});\ndim3 nthreads_{i}(32, 4);\n_spmm_conv_{i}<<<nblocks_{i}, nthreads_{i}, 0, stream_{i}>>>(input_data, output_data, {block_ptr[i]}, {block_ptr[i+1]}, kernel_ptr, kernel_map, kernel_offset, kernel_data);\n'


if len(kernel_ptr_sparse) > 1 and len(block_ptr) == 1:
    print("INSIDE!!!!!")
    code_stream_decl += 'cudaStream_t stream_sparse;\n'
    sparse_kernel_size = len(kernel_ptr_sparse) - 1
    code_kernel += code_s.replace('_OWIDTH', str(output_width)).replace('_OHEIGHT', str(output_height)).replace('_OCHANNEL', str(output_channels)).replace('_STRIDE_HEIGHT', str(vertical_stride)).replace('_STRIDE_WIDTH', str(horizontal_stride)).replace('_PADDING_HEIGHT', str(vertical_padding)).replace('_PADDING_WIDTH', str(horizontal_padding)).replace('_KERNEL_HEIGHT', str(kernel_height)).replace('_KERNEL_WIDTH', str(kernel_width)).replace('_INPUT_HEIGHT', str(input_height)).replace('_INPUT_WIDTH', str(input_width)).replace('_DIALATION_HEIGHT', str(vertical_dilation)).replace('_DIALATION_WIDTH', str(horizontal_dilation)).replace('_INPUT_CHANNEL', str(input_channel)).replace('_BATCH_SIZE', str(batch_size)).replace('_NKERNEL', str(sparse_kernel_size)).replace('_TOT_KERNEL', str(output_channels))
    call_kernel += f'cudaStreamCreate(&stream_sparse);\ndim3 nblocks_sparse({output_width*output_height*sparse_kernel_size//2}, {batch_size // 64});\ndim3 nthreads_sparse(32, 2);\n_spmm_conv_sparse<<<nblocks_sparse, nthreads_sparse, 0, stream_sparse>>>(input_data, output_data, kernel_ptr_sparse, kernel_map_sparse, kernel_offset, kernel_data);\n'

code = code_template.replace('_CODE_KERNEL', code_kernel).replace('_CALL_KERNEL', call_kernel).replace('_DECL_STREAM', code_stream_decl)

cleanup = ""
print(block_ptr)
for i in range(len(block_ptr)-1):
	block_kernel_size = block_ptr[i+1] - block_ptr[i] - 1
	block_kernel_size = block_kernel_size.item()
	if block_kernel_size  < 1:
		continue

	cleanup += f"cudaStreamDestroy(stream_{i});\n"
if len(kernel_ptr_sparse) > 1 and len(block_ptr) == 1:
	cleanup += "cudaStreamDestroy(stream_sparse);\n"

code = code.replace("_CLEAN_UP", cleanup)
filename = 'compile'

with open(filename+'.cu', 'w') as fw:
	fw.write(code)

os.system(f'/usr/local/cuda-10.2/bin/nvcc -arch=sm_52  -gencode=arch=compute_52,code=sm_52  -gencode=arch=compute_60,code=sm_60  -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70  -gencode=arch=compute_75,code=sm_75 -Xptxas "-v -dlcm=ca" -shared -Xcompiler=\"-fPIC\" -o {filename}.so {filename}.cu')
	
input_tensor = input_tensor.cuda()
output_tensor = output_tensor.cuda()
kernel_ptr = kernel_ptr.cuda()
kernel_map = kernel_map.cuda()
kernel_offset = kernel_offset.cuda()
kernel_value = kernel_value.cuda()
kernel_map_sparse = kernel_map_sparse.cuda()
kernel_ptr_sparse = kernel_ptr_sparse.cuda()


_libdir = os.path.dirname(os.path.realpath(__file__))
_lib = ctypes.CDLL(os.path.join(_libdir, filename+'.so'))
_lib.spmm_conv.restype = None
_lib.spmm_conv.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

_lib.spmm_conv(ctypes.c_void_p(input_tensor.data_ptr()), ctypes.c_void_p(output_tensor.data_ptr()), ctypes.c_void_p(kernel_ptr.data_ptr()), ctypes.c_void_p(kernel_map.data_ptr()),  ctypes.c_void_p(kernel_offset.data_ptr()), ctypes.c_void_p(kernel_value.data_ptr()), ctypes.c_void_p(kernel_ptr_sparse.data_ptr()), ctypes.c_void_p(kernel_map_sparse.data_ptr()))


os.system(f'rm compile*')