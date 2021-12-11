#!/bin/bash

for i in 1 2 4 8 16 32 64 128 256
do
	python conv.py --input_channel=64 --input_height=255 --input_width=255 --kernel_height=5 --kernel_width=5 --horizontal_stride=1 --vertical_stride=1  --nkernel=$i 2>/dev/null 
done
