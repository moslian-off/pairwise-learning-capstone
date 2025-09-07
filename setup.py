import os
import sys
import subprocess
from setuptools import setup, find_packages
from generate_proto import generate_grpc_code

# 如果直接运行此脚本，生成 gRPC 代码
if __name__ == '__main__':
    generate_grpc_code()
    if len(sys.argv) > 1:
        setup(
            name="fedavg",
            version="0.1",
            packages=find_packages(),
            install_requires=[
                "torch",
                "torchvision",
                "grpcio",
                "grpcio-tools",
                "protobuf",
            ],
        )
