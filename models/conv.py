import torch
from torch import Tensor
from typing import Tuple, Union
import datetime
import os
import ctypes
import uuid
from tqdm import tqdm

import time


import pickle
nn = 32

class SparseConv2D(torch.nn.Module):
  
  def extract_dense_old(self, sparse_kernel):
    nrows = sparse_kernel.shape[0]
    ncols = sparse_kernel.shape[1]

    cols = []

    for j in range(ncols):
      count = 0
      for i in range(nrows):
        if sparse_kernel[i,j] != 0:
          count += 1
      if count >= 32:
        cols.append(j)
     
    return cols
  
  def extract_dense(self, sparse_kernel):
    #return self.extract_dense_old(sparse_kernel)
    t1 = 1.5
    B2 = 16
    nrows = sparse_kernel.shape[0]
    cn = 8
    ncols = sparse_kernel.shape[1]

    print((sparse_kernel.abs() > 0).sum().float() / sparse_kernel.numel())
    nonempty_rows = []
    for i in range(nrows):
      for j in range(ncols):
        if sparse_kernel[i, j] != 0:
          nonempty_rows.append(i)
          break

    nonempty_cols = []
    for j in range(ncols):
      for i in nonempty_rows:
        if sparse_kernel[i, j] != 0:
          nonempty_cols.append(j)
          break
    #print (ncols, len(nonempty_cols))
    graph_file = f'{uuid.uuid1()}.txt'
    f = open(graph_file, 'w')
    f.write(str(len(nonempty_cols))+' '+str(len(nonempty_rows))+'\n')
    for j in range(len(nonempty_cols)):
      for i in range(len(nonempty_rows)):
        if sparse_kernel[nonempty_rows[i], nonempty_cols[j]] != 0:
          f.write(str(i+1)+' ')
      f.write('\n')

    f.close()
    os.system(f'./shmetis {graph_file} {cn} 10')
    try:
      f = open(f'{graph_file}.part.{cn}', 'r')
      clusters = {}
      s = f.readlines()
      #print (len(s))
    except:
      return [(list(range(nrows)), list(range(ncols)))]
      #return []
    if len(s) != len(nonempty_rows):
      return [(list(range(nrows)), list(range(ncols)))]
      #return []

    for i in range(len(s)):
      t = int(s[i].strip())
      if t not in clusters:
        clusters[t] = []
      clusters[t].append(i)
    f.close()
    #os.system(f'cat {graph_file}')
    os.system(f'rm {graph_file}*')


    clusters = [clusters[c] for c in clusters]
    clusters.sort(key=lambda x:len(x), reverse=True)
      
    blocks = []

    for r in tqdm(clusters):
      nnz_cols = [0] * ncols
      for i in range(ncols):
        s = 0
        for rr in r:
          if sparse_kernel[nonempty_rows[rr],i] != 0:
            s += 1
        nnz_cols[i] = s
      #print(nnz_cols)
      cc = sorted(list(range(ncols)), key=lambda x:nnz_cols[x], reverse=True)
      nnz_rows = [0] * len(r)

      for i in range(len(r)):
        for j in range(ncols):
          if sparse_kernel[nonempty_rows[r[i]], j] != 0:
            nnz_rows[i] += 1

      #print(r)
      #print(ncols)
      for i in range(ncols):
        #print("------")
        #print(i)
        dense_cols = cc[:(i+1)]
        flag = False
        for j in range(len(r)):
          if sparse_kernel[nonempty_rows[r[j]], cc[i]] != 0:
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
          elif len(dense_rows) > B2 :
            blocks.append((dense_rows, dense_cols))
          break

    if len(blocks) > 0:
      return blocks
    else:
      return [(list(range(nrows)), list(range(ncols)))]
      #return []

  def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, bias: bool = False, identifier=None, reuse=False):
    super(SparseConv2D, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.bias = bias
    self.identifier = uuid.uuid1() if identifier is None else identifier
    self.reuse = reuse
  def load(self, sparse_weight, bias):

    kernel_shape = sparse_weight.shape
    out_channels = kernel_shape[0]
    in_channels = kernel_shape[1]
    kernel_height = kernel_shape[2]
    kernel_width = kernel_shape[3]

    if (isinstance(self.kernel_size, Tuple)):
      self.kernel_height = self.kernel_size[0]
      self.kernel_width = self.kernel_size[1]
    else:
      self.kernel_height = self.kernel_size
      self.kernel_width = self.kernel_size


    #print(out_channels, self.out_channels, in_channels, self.in_channels, kernel_height, self.kernel_height,  kernel_width, self.kernel_width , kernel_height , kernel_width)

    assert(out_channels == self.out_channels and in_channels == self.in_channels and kernel_height == self.kernel_height and kernel_width == self.kernel_width and kernel_height % 2 == 1 and kernel_width % 2 == 1)


    # convert the sparse kernel weight into a sparse matrix and store in CSR format
    block_ptr = [0]
    kernel_ptr = []
    kernel_map = []
    kernel_offset = []
    kernel_value = []

    kernel_ptr_sparse = []
    kernel_map_sparse = []

    sparse_weight = sparse_weight.view(kernel_shape[0], kernel_shape[1] * kernel_shape[2] * kernel_shape[3])

    if not self.reuse:
      blocks = self.extract_dense(sparse_weight)
      pickle.dump(blocks, open(f'.tmp/{self.identifier}.block', 'wb'))
    else:
      blocks = pickle.load(open(f'.tmp/{self.identifier}.block', 'rb'))

    for b in blocks:
      kernel_ptr.append(len(kernel_offset))
      for r in b[0]:
        kernel_offset.extend(b[1])
        kernel_value.extend(sparse_weight[r,b[1]].tolist())
        kernel_ptr.append(len(kernel_offset))
        kernel_map.append(r)
        sparse_weight[r, b[1]] = 0
              
      kernel_map.append(-1)
      assert (len(kernel_ptr) == len(kernel_map))
      block_ptr.append(len(kernel_ptr))
    

    nrows = sparse_weight.shape[0]
    ncols = sparse_weight.shape[1]

    kernel_ptr_sparse.append(len(kernel_offset))
    
    for i in range(nrows):
      empty = True
      for j in range(ncols):
        if sparse_weight[i,j]	!= 0:
          kernel_offset.append(j)
          kernel_value.append(sparse_weight[i,j])
          empty = False
      if not empty:
        kernel_ptr_sparse.append(len(kernel_offset))
        kernel_map_sparse.append(i)


    #print(kernel_ptr_sparse)
    self.block_ptr = torch.IntTensor(block_ptr)
    self.kernel_ptr = torch.IntTensor(kernel_ptr)
    self.kernel_map = torch.IntTensor(kernel_map)
    self.kernel_offset = torch.IntTensor(kernel_offset)
    self.kernel_value = torch.FloatTensor(kernel_value)
    self.kernel_ptr_sparse = torch.IntTensor(kernel_ptr_sparse)
    self.kernel_map_sparse = torch.IntTensor(kernel_map_sparse) 

    self._lib = None
    print(len(self.kernel_ptr_sparse))
    print(len(self.block_ptr))
    return len(blocks)

  def forward(self, input: Tensor) -> Tensor:  # input: HWCN
    input = input.transpose(0, 3).transpose(1, 2).transpose(0, 1)
    if not isinstance(self.dilation, Tuple):
      vertical_dilation = self.dilation
      horizontal_dilation = self.dilation
    else:
      vertical_dilation = self.dilation[0]
      horizontal_dilation = self.dilation[1]

    if not isinstance(self.stride, Tuple):
      vertical_stride = self.stride
      horizontal_stride = self.stride
    else:
      vertical_stride = self.stride[0]
      horizontal_stride = self.stride[1]

    if not isinstance(self.padding, Tuple):
      vertical_padding = self.padding
      horizontal_padding = self.padding
    else:
      vertical_padding = self.padding[0]
      horizontal_padding = self.padding[1]

    tmp_kernel_height = self.kernel_height + (self.kernel_height - 1) * (vertical_dilation -1)
    tmp_kernel_width = self.kernel_width + (self.kernel_width - 1) * (horizontal_dilation - 1)

    # get the input dimension, check if the dimension match with kernel dimension
    input_height = input.shape[0]
    input_width = input.shape[1]
    assert(input.shape[2] == self.in_channels)
    batch_size = input.shape[3]

    tmp = input_height - tmp_kernel_height + 2 * vertical_padding
    #assert(tmp % vertical_stride == 0)
    output_height = tmp // vertical_stride + 1
    tmp = input_width - tmp_kernel_width + 2 * horizontal_padding
    #assert(tmp % horizontal_stride == 0)
    output_width = tmp // horizontal_stride + 1

    output_channels = self.out_channels
    if self._lib == None and not self.reuse:
      f = open('spmm_conv_n.cu', 'r')
      code_n = f.read()
      f.close()

      f = open('spmm_conv_sparse.cu', 'r')
      code_s = f.read()
      f.close()

      f = open('aspt_conv.cu', 'r')
      code_template = f.read()
      f.close()


      code_kernel = ''
      call_kernel = ''
      code_stream_decl = ''

      for i in range(len(self.block_ptr)-1):
        block_kernel_size = self.block_ptr[i+1] - self.block_ptr[i] - 1
        block_kernel_size = block_kernel_size.item()
        if block_kernel_size  < 1:
          continue

        code_stream_decl += f'cudaStream_t stream_{i};\n'


        if block_kernel_size % nn == 0:
          code_kernel += code_n.replace('_OWIDTH', str(output_width)).replace('_OHEIGHT', str(output_height)).replace('_OCHANNEL', str(output_channels)).replace('_STRIDE_HEIGHT', str(vertical_stride)).replace('_STRIDE_WIDTH', str(horizontal_stride)).replace('_PADDING_HEIGHT', str(vertical_padding)).replace('_PADDING_WIDTH', str(horizontal_padding)).replace('_KERNEL_HEIGHT', str(self.kernel_height)).replace('_KERNEL_WIDTH', str(self.kernel_width)).replace('_INPUT_HEIGHT', str(input_height)).replace('_INPUT_WIDTH', str(input_width)).replace('_DIALATION_HEIGHT', str(vertical_dilation)).replace('_DIALATION_WIDTH', str(horizontal_dilation)).replace('_INPUT_CHANNEL', str(self.in_channels)).replace('_BATCH_SIZE', str(batch_size)).replace('_NN', str(nn)).replace('_NKERNEL', str(block_kernel_size)).replace('_TOT_KERNEL', str(output_channels)).replace('_spmm_conv_n', f'_spmm_conv_{i}')
          call_kernel += f'cudaStreamCreate(&stream_{i});'
          call_kernel += f'\ndim3 nblocks_{i}({output_width*output_height*block_kernel_size//(4*nn)}, {batch_size // 64});\ndim3 nthreads_{i}(32, 4);\n_spmm_conv_{i}<<<nblocks_{i}, nthreads_{i}, 0, stream_{i}>>>(input_data, output_data, {self.block_ptr[i]}, {self.block_ptr[i+1]}, kernel_ptr, kernel_map, kernel_offset, kernel_data);\n'
        else:
          assert (block_kernel_size < nn)
          code_kernel += code_n.replace('_OWIDTH', str(output_width)).replace('_OHEIGHT', str(output_height)).replace('_OCHANNEL', str(output_channels)).replace('_STRIDE_HEIGHT', str(vertical_stride)).replace('_STRIDE_WIDTH', str(horizontal_stride)).replace('_PADDING_HEIGHT', str(vertical_padding)).replace('_PADDING_WIDTH', str(horizontal_padding)).replace('_KERNEL_HEIGHT', str(self.kernel_height)).replace('_KERNEL_WIDTH', str(self.kernel_width)).replace('_INPUT_HEIGHT', str(input_height)).replace('_INPUT_WIDTH', str(input_width)).replace('_DIALATION_HEIGHT', str(vertical_dilation)).replace('_DIALATION_WIDTH', str(horizontal_dilation)).replace('_INPUT_CHANNEL', str(self.in_channels)).replace('_BATCH_SIZE', str(batch_size)).replace('_NN', str(block_kernel_size)).replace('_NKERNEL', str(block_kernel_size)).replace('_TOT_KERNEL', str(output_channels)).replace('_spmm_conv_n', f'_spmm_conv_{i}')
          call_kernel += f'cudaStreamCreate(&stream_{i});'
          call_kernel += f'\ndim3 nblocks_{i}({output_width*output_height//4}, {batch_size // 64});\ndim3 nthreads_{i}(32, 4);\n_spmm_conv_{i}<<<nblocks_{i}, nthreads_{i}, 0, stream_{i}>>>(input_data, output_data, {self.block_ptr[i]}, {self.block_ptr[i+1]}, kernel_ptr, kernel_map, kernel_offset, kernel_data);\n'
      
      if len(self.kernel_ptr_sparse) > 1 and len(self.block_ptr) == 1:
        print("INSIDE!!!!!")
        code_stream_decl += 'cudaStream_t stream_sparse;\n'
        sparse_kernel_size = len(self.kernel_ptr_sparse) - 1
        code_kernel += code_s.replace('_OWIDTH', str(output_width)).replace('_OHEIGHT', str(output_height)).replace('_OCHANNEL', str(output_channels)).replace('_STRIDE_HEIGHT', str(vertical_stride)).replace('_STRIDE_WIDTH', str(horizontal_stride)).replace('_PADDING_HEIGHT', str(vertical_padding)).replace('_PADDING_WIDTH', str(horizontal_padding)).replace('_KERNEL_HEIGHT', str(self.kernel_height)).replace('_KERNEL_WIDTH', str(self.kernel_width)).replace('_INPUT_HEIGHT', str(input_height)).replace('_INPUT_WIDTH', str(input_width)).replace('_DIALATION_HEIGHT', str(vertical_dilation)).replace('_DIALATION_WIDTH', str(horizontal_dilation)).replace('_INPUT_CHANNEL', str(self.in_channels)).replace('_BATCH_SIZE', str(batch_size)).replace('_NKERNEL', str(sparse_kernel_size)).replace('_TOT_KERNEL', str(output_channels))
        call_kernel += f'cudaStreamCreate(&stream_sparse);\ndim3 nblocks_sparse({output_width*output_height*sparse_kernel_size//2}, {batch_size // 64});\ndim3 nthreads_sparse(32, 2);\n_spmm_conv_sparse<<<nblocks_sparse, nthreads_sparse, 0, stream_sparse>>>(input_data, output_data, kernel_ptr_sparse, kernel_map_sparse, kernel_offset, kernel_data);\n'
      code = code_template.replace('_CODE_KERNEL', code_kernel).replace('_CODE_N', code_kernel).replace('_CALL_KERNEL', call_kernel).replace('_DECL_STREAM', code_stream_decl)

      cleanup = ""
      for i in range(len(self.block_ptr)-1):
        block_kernel_size = self.block_ptr[i+1] - self.block_ptr[i] - 1
        block_kernel_size = block_kernel_size.item()
        if block_kernel_size  < 1:
          continue
        cleanup += f"cudaStreamDestroy(stream_{i});\n"
      if len(self.kernel_ptr_sparse) > 1 and len(self.block_ptr) == 1:
        cleanup += "cudaStreamDestroy(stream_sparse);\n"

      code = code.replace("_CLEAN_UP", cleanup)
      self.filename = f'.tmp/{self.identifier}'

      with open(self.filename+'.cu', 'w') as fw:
        fw.write(code)
      
      os.system(f'/usr/local/cuda-10.2/bin/nvcc -gencode=arch=compute_60,code=sm_60  -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70  -gencode=arch=compute_75,code=sm_75 -O2 -Xptxas "-v -dlcm=ca" -shared -Xcompiler=\"-fPIC\" -o {self.filename}.so {self.filename}.cu')

      self.kernel_ptr = self.kernel_ptr.cuda()
      self.kernel_map = self.kernel_map.cuda()
      self.kernel_offset = self.kernel_offset.cuda()
      self.kernel_value = self.kernel_value.cuda()
      self.kernel_ptr_sparse = self.kernel_ptr_sparse.cuda()
      self.kernel_map_sparse = self.kernel_map_sparse.cuda()

      self._lib = ctypes.CDLL(self.filename+'.so')
      self._lib.spmm_conv.restype = None
      self._lib.spmm_conv.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    elif self.reuse:
      self.filename = f'.tmp/{self.identifier}'
      self.kernel_ptr = self.kernel_ptr.cuda()
      self.kernel_map = self.kernel_map.cuda()
      self.kernel_offset = self.kernel_offset.cuda()
      self.kernel_value = self.kernel_value.cuda()
      self.kernel_ptr_sparse = self.kernel_ptr_sparse.cuda()
      self.kernel_map_sparse = self.kernel_map_sparse.cuda()

      self._lib = ctypes.CDLL(self.filename+'.so')
      self._lib.spmm_conv.restype = None
      self._lib.spmm_conv.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    output = torch.zeros(output_height, output_width, output_channels, batch_size).cuda()
    self._lib.spmm_conv(ctypes.c_void_p(input.data_ptr()), ctypes.c_void_p(output.data_ptr()), ctypes.c_void_p(self.kernel_ptr.data_ptr()), ctypes.c_void_p(self.kernel_map.data_ptr()),  ctypes.c_void_p(self.kernel_offset.data_ptr()), ctypes.c_void_p(self.kernel_value.data_ptr()), ctypes.c_void_p(self.kernel_ptr_sparse.data_ptr()), ctypes.c_void_p(self.kernel_map_sparse.data_ptr()))
    del input
    a = output.transpose(0, 1).transpose(0, 3).transpose(1, 2).clone()
    del output
    return a
