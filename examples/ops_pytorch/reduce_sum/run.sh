#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e

torch_location=$(python3 -c "import torch; print(torch.__path__[0])")
torch_npu_location=$(python3 -c "import torch_npu; print(torch_npu.__path__[0])")
python_include=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
python_lib=$(python3 -c "import sysconfig; print(sysconfig.get_path('stdlib'))")
lib_path=$(dirname "$python_lib")
export LD_LIBRARY_PATH=${torch_npu_location}/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${torch_location}/lib/:$LD_LIBRARY_PATH
if [ -z "$ATVC_PATH" ]; then
    atvc_path=$(realpath ../../../include)
else
    atvc_path=$ATVC_PATH
fi

rm -rf *.json
rm -rf libascendc_pytorch.so


if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

bisheng -x cce pytorch_ascendc_extension.cpp \
    -D_GLIBCXX_USE_CXX11_ABI=0  \
    -I${torch_location}/include   \
    -I${torch_location}/include/torch/csrc/api/include   \
    -I${python_include}   \
    -I${atvc_path}   \
    -I${torch_npu_location}/include   \
    -L${torch_location}/lib   \
    -L${torch_npu_location}/lib   \
    -L${python_lib}   \
    -L${lib_path}   \
    -L${_ASCEND_INSTALL_PATH}/lib64 \
    -ltorch -ltorch_cpu -lc10 -ltorch_npu -lpython3 -ltorch_python   \
    -shared -cce-enable-plugin --cce-aicore-arch=dav-c220 -fPIC -ltiling_api -lplatform -lm -ldl  \
    -o libascendc_pytorch.so

python3 run_op.py

if [ $? -ne 0 ]; then
    echo "ERROR: verify result failed! the result is wrong!"
    return 1
fi

rm -rf *.json
rm -rf libascendc_pytorch.so