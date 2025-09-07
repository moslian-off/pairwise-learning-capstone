"""
联邦学习客户端实现
"""
import os
import time
import uuid
import logging
import pickle
import math
import grpc
import numpy as np
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common.generated import federated_pb2, federated_pb2_grpc
from common.models import GenericAlgorithmModel
from common.utils import load_data
from config import CONFIG

logger = logging.getLogger("FederatedClient")

class FederatedClient:
    def __init__(self, config, client_id=None):
        """
        初始化联邦学习客户端
        
        Args:
            config: 配置对象
            client_id: 客户端ID，如果为None则自动生成
        """
        self.config = config
        self.client_id = client_id if client_id else f"client-{str(uuid.uuid4())[:8]}"
        self.logger = logging.getLogger(f"Client-{self.client_id}")
        
        # 创建gRPC通道
        self.channel = grpc.insecure_channel(
            config['server_address'],
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
                ('grpc.max_receive_message_length', 50 * 1024 * 1024)  # 50MB
            ]
        )
        self.stub = federated_pb2_grpc.FederatedServiceStub(self.channel)
        
        # 初始化模型
        self.model = GenericAlgorithmModel(config['algorithm'], config)
        
        # 加载数据
        self.train_data, self.test_data, self.dataset_size = load_data(
            config['data_path'], 
            config['data_split']
        )
        
        self.logger.info(f"客户端初始化完成，算法: {config['algorithm']}")
        self.logger.info(f"数据集大小: {self.dataset_size} 样本")
    
    def register(self):
        """向服务器注册客户端"""
        try:
            self.logger.info(f"开始注册客户端，ID: {self.client_id}, 样本数: {self.dataset_size}")
            
            request = federated_pb2.ClientInfo(
                client_id=self.client_id,
                num_samples=self.dataset_size
            )
            response = self.stub.Register(request)
            self.logger.info(f"注册成功: {response.message}")
            return True
        except Exception as e:
            self.logger.error(f"注册失败: {e}")
            return False
    
    def get_global_model(self, round_num=0):
        """
        从服务器获取全局模型（分块接收）
        
        Args:
            round_num: 请求的轮次，默认为0表示当前轮次
            
        Returns:
            int: 当前轮次，如果失败则返回None
        """
        request = federated_pb2.ModelRequest(client_id=self.client_id, round=round_num)
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                chunks = []
                total_chunks = None
                current_round = None
                
                # 接收分块模型
                for chunk in self.stub.GetGlobalModel(request):
                    chunks.append((chunk.chunk_id, chunk.model_chunk))
                    if total_chunks is None:
                        total_chunks = chunk.total_chunks
                        current_round = chunk.round
                
                # 确保收到所有块
                if len(chunks) == total_chunks:
                    # 按块ID排序并合并
                    chunks.sort(key=lambda x: x[0])
                    model_bytes = b''.join([chunk[1] for chunk in chunks])
                    
                    # 反序列化模型
                    model_state = pickle.loads(model_bytes)
                    self.model.load_state_dict(model_state)
                    self.logger.info(f"成功获取全局模型，共 {total_chunks} 块，当前轮次: {current_round}")
                    return current_round
                else:
                    self.logger.error(f"接收的块数量不匹配: 收到 {len(chunks)}/{total_chunks}")
                    retry_count += 1
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                    # 服务器正在聚合，需要等待
                    self.logger.info("服务器正在聚合模型，等待...")
                    time.sleep(2)
                    retry_count += 1
                else:
                    self.logger.error(f"获取全局模型失败: {e}")
                    retry_count += 1
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"获取全局模型失败: {e}")
                retry_count += 1
                time.sleep(1)
        
        self.logger.error(f"获取全局模型失败，已达到最大重试次数 {max_retries}")
        return None
    
    def train_local_model(self, round_num=None):
        """
        本地训练
    
        Args:
            round_num: 当前轮次，用于日志记录
        
        Returns:
            tuple: (aucs, times, losses) 训练结果
        """
        self.logger.info(f"开始轮次 {round_num + 1 if round_num is not None else '?'} 的本地训练")
        start_time = time.time()
    
        # 直接在模型训练时传递轮次信息
        round_info = round_num + 1 if round_num is not None else None
        # 执行训练，传递轮次信息作为额外参数
        aucs, times, losses = self.model.train(self.train_data, round_info)
    
        training_time = time.time() - start_time
        self.logger.info(f"轮次 {round_num + 1 if round_num is not None else '?'} 本地训练完成，耗时: {training_time:.6f} 秒")
    
        # 记录训练结果
        if len(aucs) > 0 and len(losses) > 0:
            self.logger.info(f"轮次 {round_num + 1 if round_num is not None else '?'} 训练结果 - 最终AUC: {aucs[-1]:.4f}, 最终Loss: {losses[-1]:.4f}")
    
        return aucs, times, losses


    def submit_update(self, round_num):
        """提交模型更新（分块上传）"""
        try:
            self.logger.info(f"开始准备第 {round_num} 轮模型更新")
            
            model_state = self.model.get_state_dict()
            model_bytes = pickle.dumps(model_state)
            
            self.logger.info(f"模型序列化完成，大小: {len(model_bytes)} 字节")
            
            # 将模型数据分块
            chunk_size = self.config.get('chunk_size', 1024 * 1024)  # 默认1MB
            chunks = [model_bytes[i:i+chunk_size] for i in range(0, len(model_bytes), chunk_size)]
            total_chunks = len(chunks)
            
            self.logger.info(f"模型分块完成，共 {total_chunks} 个分块，每个 {chunk_size} 字节")
            
            # 创建流式请求
            def chunk_generator():
                for chunk_id, chunk_data in enumerate(chunks):
                    if chunk_id % 10 == 0:  # 每10个分块记录一次
                        self.logger.debug(f"发送分块 {chunk_id + 1}/{total_chunks}")
                    yield federated_pb2.ModelChunk(
                        model_chunk=chunk_data,
                        chunk_id=chunk_id,
                        total_chunks=total_chunks,
                        client_id=self.client_id,
                        round=round_num
                    )
            
            # 调用流式RPC
            self.logger.info("开始提交模型更新到服务器")
            response = self.stub.SubmitUpdate(chunk_generator())
            self.logger.info(f"模型更新提交成功: {response.message}")
            return True
        except Exception as e:
            self.logger.error(f"提交失败: {e}")
            return False
    
    def get_training_status(self):
        """获取训练状态"""
        try:
            request = federated_pb2.StatusRequest(client_id=self.client_id)
            response = self.stub.GetTrainingStatus(request)
            return response.current_round, response.training_complete
        except Exception as e:
            self.logger.error(f"获取训练状态失败: {e}")
            return None, None
    
    def wait_for_round_completion(self, current_round):
        """
        等待当前轮次完成
        
        Args:
            current_round: 当前轮次
            
        Returns:
            bool: 如果成功等待轮次完成则返回True，否则返回False
        """
        max_retries = 30  # 最多等待30次
        retry_count = 0
        
        while retry_count < max_retries:
            new_round, training_complete = self.get_training_status()
            
            if training_complete:
                self.logger.info("训练已完成")
                return True
                
            if new_round is None:
                self.logger.error("获取训练状态失败")
                retry_count += 1
                time.sleep(2)
                continue
                
            if new_round > current_round:
                self.logger.info(f"轮次已更新: {current_round} -> {new_round}")
                return True
                
            self.logger.info(f"等待轮次 {current_round} 完成，当前服务器轮次: {new_round}")
            retry_count += 1
            time.sleep(2)
        
        self.logger.error(f"等待轮次 {current_round} 完成超时")
        return False


    def run(self):
        """主循环"""
        self.logger.info("客户端主循环启动")
        
        if not self.register():
            self.logger.error("注册失败，客户端退出")
            return
        
        # 获取训练状态
        current_round, training_complete = self.get_training_status()
        if current_round is None:
            self.logger.error("获取训练状态失败，退出")
            return
        
        if training_complete:
            self.logger.info("训练已完成")
            return
    
        # 获取初始全局模型
        round_num = self.get_global_model()
        if round_num is None:
            self.logger.error("获取初始全局模型失败，退出")
            return
        
        # 评估初始模型
        try:
            initial_metrics = self.model.evaluate(self.test_data)
            self.logger.info(f"初始模型评估 - AUC: {initial_metrics['auc']:.4f}, Loss: {initial_metrics['loss']:.4f}")
        except Exception as e:
            self.logger.error(f"评估初始模型失败: {e}")
            initial_metrics = {'auc': 0.5, 'loss': 0.0}
    
        # 主训练循环
        while True:
            # 获取当前训练状态
            current_round, training_complete = self.get_training_status()
            if current_round is None:
                self.logger.error("获取训练状态失败，重试中...")
                time.sleep(2)
                continue
            
            if training_complete:
                self.logger.info("训练已完成")
                break
            
            # 添加明确的轮次分隔符
            self.logger.info(f"========== 开始联邦学习轮次 {current_round + 1}/{self.config['total_rounds']} ==========")
    
            # 在本地数据上训练模型，传递轮次信息
            aucs, times, losses = self.train_local_model(current_round)
            
            # 提交模型更新
            self.logger.info(f"向服务器提交轮次 {current_round + 1} 的模型更新")
            if not self.submit_update(current_round):
                self.logger.error(f"轮次 {current_round + 1} 提交模型更新失败，重试中...")
                time.sleep(2)
                continue
        
            self.logger.info(f"轮次 {current_round + 1} 模型已提交，等待服务器聚合")
        
            # 等待当前轮次完成
            self.logger.info(f"等待服务器完成轮次 {current_round + 1} 的模型聚合")
            if not self.wait_for_round_completion(current_round):
                self.logger.warning(f"等待轮次 {current_round + 1} 完成超时，继续下一轮")
    
            # 获取新的全局模型
            self.logger.info(f"从服务器获取轮次 {current_round + 2} 的全局模型")
            new_round = self.get_global_model()
            if new_round is None:
                self.logger.error("获取全局模型失败，重试中...")
                time.sleep(2)
                continue
            
            self.logger.info(f"成功获取轮次 {new_round + 1} 的全局模型")
    
        # 添加明确的训练完成分隔符
        self.logger.info("========== 联邦学习训练完成 ==========")
    
        # 获取最终模型并评估
        try:
            final_round = self.get_global_model()
            if final_round is not None:
                final_metrics = self.model.evaluate(self.test_data)
                self.logger.info(f"最终模型评估 - AUC: {final_metrics['auc']:.4f}, Loss: {final_metrics['loss']:.4f}")
            self.logger.info(f"AUC提升: {final_metrics['auc'] - initial_metrics['auc']:.4f}")
        except Exception as e:
            self.logger.error(f"评估最终模型失败: {e}")
    
        self.logger.info("客户端任务完成")


    def close(self):
        """关闭客户端连接"""
        self.channel.close()
        self.logger.info("客户端连接已关闭")

def run_client():
    """启动客户端"""
    parser = argparse.ArgumentParser(description="联邦学习客户端")
    parser.add_argument("--server", type=str, default=None, help="服务器地址")
    parser.add_argument("--id", type=str, default=None, help="客户端ID")
    parser.add_argument("--split", type=float, default=None, help="数据集分割比例")
    parser.add_argument("--epochs", type=int, default=None, help="本地训练轮数")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        from config import load_config
        config = load_config(args.config)
    else:
        from config import CONFIG
        config = CONFIG
    
    # 更新命令行参数
    if args.server:
        config['server_address'] = args.server
    if args.split:
        config['data_split'] = args.split
    if args.epochs:
        config['local_epochs'] = args.epochs
    
    # 创建并运行客户端
    client = FederatedClient(config, args.id)
    
    try:
        client.run()
    except KeyboardInterrupt:
        logger.info("接收到中断信号，客户端退出")
    finally:
        client.close()

if __name__ == "__main__":
    run_client()
