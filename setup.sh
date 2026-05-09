#!/bin/bash
# setup.sh - 安装 Python 依赖并编译 dual/polar 中的 C 扩展

set -e  # 遇到错误立即退出

make clean
pip install numpy scipy >/dev/null 
cd dual/polar
make  >/dev/null
cd ../..

echo "Installation successful"
