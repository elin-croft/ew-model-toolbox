# !/bin/bash

VGG16="
--kernel_size 3
--channels 3-64,64-64,64-128,128-128,128-256,256-256,256-512
--stride 1
--padding 1
--norm BN
--activations Relu
--pooling max_2-2-2
"
python test.py $VGG16