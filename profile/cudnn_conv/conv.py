import argparse
import torch as th
import re
import io
import datetime
import os
import ctypes

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
parser.add_argument('--cuda_file', default='cudnn_conv.cu')
parser.add_argument('--kernel_file', default=None)

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

kernel_file = args.kernel_file

sparse_kernel = th.load(kernel_file)
kernel_shape = sparse_kernel.shape
nkernel = kernel_shape[0]
input_channel = kernel_shape[1]
kernel_height = kernel_shape[2]
kernel_width = kernel_shape[3] 


#if nkernel >= 512 or input_channel >= 512:
#	input_height = 65
#	input_width = 65


cuda_file = args.cuda_file


assert (kernel_height % 2 == 1 and kernel_width % 2 == 1)
tmp_kernel_height = kernel_height + (kernel_height - 1) * (vertical_dilation - 1)
tmp_kernel_width = kernel_width + (kernel_width - 1) * (horizontal_dilation - 1)
tmp = input_height - tmp_kernel_height + 2 * vertical_padding
#assert (tmp % vertical_stride == 0)
output_height = tmp // vertical_stride + 1
tmp = input_width - tmp_kernel_width + 2 * horizontal_padding
#assert (tmp % horizontal_stride == 0)
output_width = tmp // horizontal_stride + 1

output_channels = nkernel


with open(cuda_file, 'r') as f:
	code = f.read()
	code = code.replace('S_kernels', str(nkernel)).replace('S_channels', str(input_channel)).replace('S_kernel_height', str(kernel_height)).replace('S_kernel_width', str(kernel_width)).replace('S_vertical_stride', str(vertical_stride)).replace('S_horizontal_stride', str(horizontal_stride)).replace('S_dilation_height', str(vertical_dilation)).replace('S_dilation_width', str(horizontal_dilation)).replace('S_padding_height', str(vertical_padding)).replace('S_padding_width', str(horizontal_padding)).replace('S_batch_size', str(batch_size)).replace('S_input_height', str(input_height)).replace('S_input_width', str(input_width))


timestamp = datetime.datetime.now().time()
filename = f'.tmp/tmp_{timestamp}'
with open(filename+'.cu', 'w') as fw:
	fw.write(code)

os.system(f'cp Makefile .tmp/; cd .tmp; make; CUDA_VISIBLE_DEVICES=2 ./conv; cd ..')
os.system(f'rm .tmp/*')



