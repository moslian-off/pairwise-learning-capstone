import os
import sys
import subprocess

def generate_grpc_code():
    """生成 gRPC 代码并修复导入问题"""
    # 确保目录存在
    os.makedirs(os.path.join('common', 'generated'), exist_ok=True)
    
    # 定义 proto 文件路径
    proto_file = os.path.join('protos', 'federated.proto')
    
    # 生成 Python 代码
    cmd = [
        sys.executable, '-m', 'grpc_tools.protoc',
        '-I', 'protos',
        f'--python_out=common/generated',
        f'--grpc_python_out=common/generated',
        proto_file
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    # 创建 __init__.py 文件
    with open(os.path.join('common', 'generated', '__init__.py'), 'w') as f:
        pass
    
    # 修复导入问题 - 使用 UTF-8 编码
    pb2_grpc_file = os.path.join('common', 'generated', 'federated_pb2_grpc.py')
    try:
        # 明确指定 UTF-8 编码
        with open(pb2_grpc_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换导入语句
        content = content.replace(
            'import federated_pb2 as federated__pb2',
            'from . import federated_pb2 as federated__pb2'
        )
        
        # 使用相同的编码写回文件
        with open(pb2_grpc_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("gRPC 代码生成完成，并已修复导入问题")
    except Exception as e:
        print(f"修复导入问题时出错: {e}")
        # 如果修复失败，不要中断程序执行
        print("继续执行，但可能需要手动修复导入问题")

if __name__ == '__main__':
    generate_grpc_code()
