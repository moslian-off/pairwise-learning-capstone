"""
联邦学习服务器实现
"""
import os
import sys
import time
import grpc
import threading
import pickle
import numpy as np
import logging
import argparse
from concurrent import futures
from collections import defaultdict
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common.generated import federated_pb2, federated_pb2_grpc
from common.models import GenericAlgorithmModel
from common.utils import load_data
from config import CONFIG

logger = logging.getLogger("FederatedServer")

class FederatedServer(federated_pb2_grpc.FederatedServiceServicer):
    def __init__(self, config):
        """
        初始化联邦学习服务器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.global_model = GenericAlgorithmModel(config['algorithm'], config)
        self.current_round = 0
        self.total_rounds = config['total_rounds']
        self.min_clients = config['min_clients']
        
        # 客户端管理
        self.clients = {}  # {client_id: {'sample_size': size, 'last_active': timestamp}}
        self.client_lock = threading.Lock()
        
        # 模型上传管理
        self.uploaded_models = defaultdict(dict)  # {round: {client_id: model_data}}
        self.upload_lock = threading.Lock()
        
        # 聚合事件 - 用于通知聚合线程
        self.aggregation_event = threading.Event()
        
        # 聚合线程
        self.aggregation_thread = None
        self.stop_aggregation = False
        
        # 轮次完成事件 - 用于客户端同步
        self.round_completed = threading.Event()
        self.round_completed.set()  # 初始轮次视为已完成
        
        # 加载测试数据（如果有）
        try:
            _, self.test_data, _ = load_data(config['data_path'], 0.2)
            self.has_test_data = True
        except Exception as e:
            logger.warning(f"无法加载测试数据: {e}")
            self.test_data = None
            self.has_test_data = False
        
        logger.info(f"服务器初始化完成，算法: {config['algorithm']}")
        logger.info(f"总轮次: {self.total_rounds}, 最小客户端数: {self.min_clients}")
        
        # 启动聚合线程
        self.start_aggregation_thread()
    
    def start_aggregation_thread(self):
        """启动模型聚合线程"""
        self.aggregation_thread = threading.Thread(target=self.aggregate_models_periodically)
        self.aggregation_thread.daemon = True
        self.aggregation_thread.start()
        logger.info("聚合线程已启动")
    
    def aggregate_models_periodically(self):
        """定期检查是否可以聚合模型"""
        while not self.stop_aggregation:
            # 等待聚合事件或检查是否有足够的模型
            with self.upload_lock:
                can_aggregate = (self.current_round in self.uploaded_models and 
                                len(self.uploaded_models[self.current_round]) >= self.min_clients)
            
            if can_aggregate:
                # 清除轮次完成标志，阻止新客户端进入
                self.round_completed.clear()
                
                logger.info(f"轮次 {self.current_round} 收到足够的模型，开始聚合")
                self.aggregate_models()
                
                # 进入下一轮
                self.current_round += 1
                logger.info(f"进入轮次 {self.current_round}/{self.total_rounds}")
                
                # 设置轮次完成标志，允许客户端进入新轮次
                self.round_completed.set()
                
                # 检查是否完成所有轮次
                if self.current_round >= self.total_rounds:
                    logger.info("所有轮次已完成，保存最终模型")
                    self.save_model("final_model.pth")
                    self.stop_aggregation = True
            else:
                # 如果没有足够的模型，等待一段时间再检查
                time.sleep(1)
    
    def aggregate_models(self):
        """聚合当前轮次的模型"""
        with self.upload_lock:
            if self.current_round not in self.uploaded_models:
                logger.warning(f"轮次 {self.current_round} 没有上传的模型")
                return
            
            client_models = self.uploaded_models[self.current_round]
            if not client_models:
                logger.warning(f"轮次 {self.current_round} 没有上传的模型")
                return
            
            # 计算总样本数
            total_samples = sum(self.clients[client_id]['sample_size'] for client_id in client_models)
            
            # 加权聚合模型参数
            new_params = np.zeros_like(self.global_model.get_parameters())
            
            for client_id, model_data in client_models.items():
                # 反序列化客户端模型
                client_state = pickle.loads(model_data)
                client_params = client_state['parameters']
                
                # 计算权重 (基于样本数量)
                weight = self.clients[client_id]['sample_size'] / total_samples
                
                # 加权累加参数
                new_params += client_params * weight
            
            # 更新全局模型
            self.global_model.set_parameters(new_params)
            
            # 评估全局模型
            if self.has_test_data:
                metrics = self.global_model.evaluate(self.test_data)
                logger.info(f"轮次 {self.current_round} 全局模型评估 - AUC: {metrics['auc']:.4f}, Loss: {metrics['loss']:.4f}")
            
            # 保存当前轮次的模型
            self.save_model(f"model_round_{self.current_round}.pth")
            
            logger.info(f"轮次 {self.current_round} 模型聚合完成，参与客户端: {len(client_models)}")
            
            # 清除当前轮次的上传模型，释放内存
            self.uploaded_models[self.current_round].clear()
    
    def save_model(self, filename):
        """保存模型到文件"""
        try:
            os.makedirs("models", exist_ok=True)
            path = os.path.join("models", filename)
            
            # 保存模型状态
            with open(path, 'wb') as f:
                pickle.dump(self.global_model.get_state_dict(), f)
            
            logger.info(f"模型已保存到 {path}")
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
    
    def Register(self, request, context):
        """客户端注册"""
        client_id = request.client_id
        sample_size = request.num_samples
        
        with self.client_lock:
            self.clients[client_id] = {
                'sample_size': sample_size,
                'last_active': time.time()
            }
            
        logger.info(f"客户端 {client_id} 注册成功，样本大小: {sample_size}")
        logger.info(f"当前注册客户端数: {len(self.clients)}")
        
        return federated_pb2.RegistrationResponse(message=f"客户端 {client_id} 注册成功，样本大小: {sample_size}")
    
    def GetGlobalModel(self, request, context):
        """处理获取全局模型请求（流式传输）"""
        client_id = request.client_id
        round_num = request.round
        
        # 确保客户端请求的是当前轮次或之前的轮次
        if round_num > self.current_round:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(f"请求的轮次 {round_num} 大于当前轮次 {self.current_round}")
            return
        
        # 等待当前轮次完成
        if not self.round_completed.is_set():
            logger.info(f"客户端 {client_id} 等待轮次 {self.current_round} 完成")
            # 在实际的gRPC流中，我们不能直接等待，但可以延迟响应
            # 这里我们简单地等待一段时间
            time.sleep(1)
        
        try:
            logger.info(f"客户端 {client_id} 请求全局模型，当前轮次: {self.current_round}")
            
            # 序列化模型
            model_bytes = pickle.dumps(self.global_model.get_state_dict())
            
            # 计算需要多少块
            chunk_size = self.config.get('chunk_size', 1024 * 1024)  # 默认1MB
            total_chunks = (len(model_bytes) + chunk_size - 1) // chunk_size
            
            # 分块发送
            for i in range(total_chunks):
                start_pos = i * chunk_size
                end_pos = min((i + 1) * chunk_size, len(model_bytes))
                chunk = model_bytes[start_pos:end_pos]
                
                yield federated_pb2.ModelChunk(
                    model_chunk=chunk,
                    chunk_id=i,
                    total_chunks=total_chunks,
                    round=self.current_round
                )
            
            logger.info(f"全局模型已发送给客户端 {client_id}，共 {total_chunks} 块")
        except Exception as e:
            logger.error(f"发送全局模型给客户端 {client_id} 时出错: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"发送模型失败: {str(e)}")
    
    def SubmitUpdate(self, request_iterator, context):
        """接收客户端分块上传的模型更新"""
        chunks = []
        client_id = None
        round_num = None
        total_chunks = None
        
        try:
            for chunk in request_iterator:
                if client_id is None:
                    client_id = chunk.client_id
                    round_num = chunk.round
                    total_chunks = chunk.total_chunks
                
                chunks.append((chunk.chunk_id, chunk.model_chunk))
                
            # 按块ID排序并合并
            chunks.sort(key=lambda x: x[0])
            model_bytes = b''.join([chunk[1] for chunk in chunks])
            
            # 确保客户端上传的是当前轮次的模型
            if round_num != self.current_round:
                message = f"客户端 {client_id} 上传的轮次 {round_num} 与当前轮次 {self.current_round} 不匹配"
                logger.warning(message)
                return federated_pb2.UpdateResponse(message=message)
            
            # 存储上传的模型
            with self.upload_lock:
                self.uploaded_models[round_num][client_id] = model_bytes
                
                message = f"客户端 {client_id} 上传模型成功，轮次: {round_num}"
                logger.info(message)
                logger.info(f"轮次 {round_num} 已收到 {len(self.uploaded_models[round_num])}/{len(self.clients)} 客户端模型")
                
                # 检查是否达到聚合条件
                if len(self.uploaded_models[round_num]) >= self.min_clients:
                    # 触发聚合事件
                    self.aggregation_event.set()
            
            return federated_pb2.UpdateResponse(message=message)
        except Exception as e:
            logger.error(f"处理客户端 {client_id} 的模型更新时出错: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"处理模型更新失败: {str(e)}")
            return federated_pb2.UpdateResponse(message=f"处理模型更新失败: {str(e)}")
    
    def GetTrainingStatus(self, request, context):
        """获取训练状态"""
        client_id = request.client_id
        
        # 更新客户端活跃时间
        with self.client_lock:
            if client_id in self.clients:
                self.clients[client_id]['last_active'] = time.time()
        
        # 返回当前训练状态
        is_complete = self.current_round >= self.total_rounds
        
        logger.debug(f"客户端 {client_id} 请求训练状态: 轮次 {self.current_round}/{self.total_rounds}, 完成: {is_complete}")
        
        return federated_pb2.TrainingStatus(
            current_round=self.current_round,
            training_complete=is_complete
        )

def serve(config=None):
    """启动服务器"""
    if config is None:
        config = CONFIG
    
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)  # 50MB
        ]
    )
    federated_pb2_grpc.add_FederatedServiceServicer_to_server(
        FederatedServer(config), server
    )
    
    # 使用IPv4地址
    server_address = config['server_address']
    server.add_insecure_port(f'0.0.0.0:{server_address.split(":")[-1]}' if ':' in server_address else server_address)
    
    server.start()
    logger.info(f"服务器启动在 {server_address}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("接收到终止信号，服务器关闭")
        server.stop(0)

def run_server():
    """启动服务器的命令行入口"""
    parser = argparse.ArgumentParser(description="联邦学习服务器")
    parser.add_argument("--port", type=int, default=None, help="服务器端口")
    parser.add_argument("--rounds", type=int, default=None, help="训练轮次")
    parser.add_argument("--min-clients", type=int, default=None, help="每轮最小客户端数")
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
    if args.port:
        host = config['server_address'].split(':')[0] if ':' in config['server_address'] else 'localhost'
        config['server_address'] = f"{host}:{args.port}"
    if args.rounds:
        config['total_rounds'] = args.rounds
    if args.min_clients:
        config['min_clients'] = args.min_clients
    
    # 启动服务器
    serve(config)

if __name__ == "__main__":
    run_server()
