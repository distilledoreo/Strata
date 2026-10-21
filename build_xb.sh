#!/bin/bash
cd /home/distilledoreo/AI/state-projection-strata-expert
export PATH=/usr/local/cuda-12.9/bin:$PATH
cmake -G Ninja -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON   -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DSTRATA_GGML_DIR=/home/distilledoreo/AI/state-projection-strata-expert/third_party/llama.cpp   -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_RESEARCH_ADAPTIVE_PREFILL=ON > configure_xb.log 2>&1 || exit 1
cmake --build build --target strata -j 6 > build_xb.log 2>&1 || cmake --build build --target strata -j 6 >> build_xb.log 2>&1
