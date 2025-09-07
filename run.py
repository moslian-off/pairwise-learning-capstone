import os
import sys
import argparse
import subprocess
import time
import signal
import threading
from generate_proto import generate_grpc_code

# 在项目的入口点（如 run.py）中添加
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 输出到标准输出
    ]
)

# 设置所有日志处理器为无缓冲
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        handler.flush = lambda: True  # 强制刷新


def start_server(port=50051, rounds=30, min_clients=3):
    """启动服务器"""
    print(f"启动服务器，端口: {port}, 轮次: {rounds}, 最小客户端数: {min_clients}")
    server_process = subprocess.Popen(
        [
            sys.executable, 
            os.path.join('server', 'server.py'),
            f'--port={port}',
            f'--rounds={rounds}',
            f'--min-clients={min_clients}'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # 创建线程来实时输出服务器日志
    def print_output(process):
        for line in iter(process.stdout.readline, ''):
            print(f"[SERVER] {line.strip()}")
    
    thread = threading.Thread(target=print_output, args=(server_process,))
    thread.daemon = True
    thread.start()
    
    return server_process

def start_client(server_address, client_id=None, data_split=0.7, epochs=100):
    """启动客户端"""
    client_id_str = f"--id={client_id}" if client_id else ""
    print(f"启动客户端，服务器地址: {server_address}, 数据分割: {data_split}, 本地训练轮数: {epochs}")
    
    client_process = subprocess.Popen(
        [
            sys.executable,
            os.path.join('client', 'client.py'),
            f'--server={server_address}',
            client_id_str,
            f'--split={data_split}',  # 修改这里，从 --data-split 改为 --split
            f'--epochs={epochs}'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # 创建线程来实时输出客户端日志
    def print_output(process, client_name):
        for line in iter(process.stdout.readline, ''):
            print(f"[{client_name}] {line.strip()}")
    
    client_name = client_id if client_id else f"CLIENT-{client_process.pid}"
    thread = threading.Thread(target=print_output, args=(client_process, client_name))
    thread.daemon = True
    thread.start()
    
    return client_process

def main():
    
    parser = argparse.ArgumentParser(description='运行联邦学习系统')
    parser.add_argument('--mode', choices=['server', 'client', 'all'], default='all',
                        help='运行模式: server, client, 或 all (默认)')
    parser.add_argument('--port', type=int, default=50054, help='服务器端口')
    parser.add_argument('--rounds', type=int, default=1, help='训练轮次')
    parser.add_argument('--min-clients', type=int, default=3, help='每轮最小客户端数')
    parser.add_argument('--num-clients', type=int, default=3, help='启动的客户端数量')
    parser.add_argument('--data-split', type=float, default=0.5, help='每个客户端使用的数据比例')
    parser.add_argument('--epochs', type=int, default=100, help='每轮本地训练的轮数')
    parser.add_argument('--server-address', type=str, default='localhost', 
                        help='服务器地址 (不包括端口)')
    args = parser.parse_args()
    
    # 生成 gRPC 代码
    generate_grpc_code()
    
    processes = []
    server_address = f"{args.server_address}:{args.port}"
    
    try:
        # 启动服务器
        if args.mode in ['server', 'all']:
            server_process = start_server(args.port, args.rounds, args.min_clients)
            processes.append(server_process)
            # 等待服务器启动
            time.sleep(2)
        
        # 启动客户端
        if args.mode in ['client', 'all']:
            for i in range(args.num_clients):
                client_id = f"client-{i+1}"
                client_process = start_client(
                    server_address, 
                    client_id, 
                    args.data_split, 
                    args.epochs
                )
                processes.append(client_process)
                # 稍微延迟启动下一个客户端
                time.sleep(1)
        
        # 等待所有进程完成
        for process in processes:
            process.wait()
            
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在关闭所有进程...")
        for process in processes:
            try:
                # 在 Windows 上使用 CTRL_BREAK_EVENT，在 Unix 上使用 SIGTERM
                if os.name == 'nt':
                    os.kill(process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    os.kill(process.pid, signal.SIGTERM)
            except:
                # 如果进程已经结束，忽略错误
                pass
        
        # 给进程一些时间来清理
        time.sleep(2)
        
        # 强制终止任何仍在运行的进程
        for process in processes:
            if process.poll() is None:  # 如果进程仍在运行
                process.terminate()
                
        print("所有进程已关闭")

if __name__ == "__main__":
    main()
