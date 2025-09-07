"""
模型定义模块
"""
import numpy as np
import importlib
import logging

logger = logging.getLogger(__name__)

class GenericAlgorithmModel:
    """通用算法模型，支持alg包中的所有算法"""
    
    def __init__(self, algorithm_name, config):
        """
        初始化模型
        
        Args:
            algorithm_name: 算法名称，对应alg包中的模块名
            config: 配置字典或Config实例
        """
        self.algorithm_name = algorithm_name
        self.config = config
        self.parameters = np.zeros(config['feature_dim'])
        
        # 尝试加载算法函数
        try:
            algorithm_module = importlib.import_module(f'alg.{self.algorithm_name}')
            self.algorithm_func = getattr(algorithm_module, self.algorithm_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"加载算法 {self.algorithm_name} 失败: {e}")
            logger.warning("使用占位算法函数")
            self.algorithm_func = self._placeholder_algorithm
        
        logger.info(f"初始化模型: {algorithm_name}, 特征维度: {config['feature_dim']}")
    
    def _placeholder_algorithm(self, x_tr, y_tr, x_te, y_te, options, w_init):
        """占位算法函数，用于测试"""
        logger.warning("使用占位算法函数，仅用于测试")
        w = w_init.copy()
        return w, [0.5], [0.0], [0.0]
    
    def train(self, data, round_info=None):
        """
        调用算法进行训练
        
        Args:
            data: 训练数据元组 (x, y)，其中x为特征，y为标签
            round_info: 轮次信息，用于日志记录
                
        Returns:
            tuple: (aucs, times, losses) 评估指标
        """
        logger.info(f"开始训练，数据大小: {1000}")
    
        round_str = f"轮次 {round_info}" if round_info is not None else "未知轮次"
    
        # 准备训练选项
        options = {
            'n_tr': 1000,
            'dim': self.config['feature_dim'],
            'n_pass': self.config.get('n_pass', 1),
            'eta': self.config.get('learning_rate', 0.01),
            'beta': self.config.get('beta', 0.1),
            'proj_flag': self.config.get('proj_flag', True),
            'etas': self.config.get('learning_rate', 0.01) / np.sqrt(np.arange(1, len(data[1]) + 1)),
            'res_idx': [len(data[1])],
            'round_info': round_info  # 传递轮次信息给算法
        }
    
        # 调用算法，接收4个返回值（包括loss记录）
        try:
            updated_parameters, aucs, times, losses = self.algorithm_func(
                data[0], data[1], data[0], data[1], options, self.parameters
            )
            
            # 更新模型参数
            self.parameters = updated_parameters
            
            logger.info(f"{round_str} 训练完成，最终AUC: {aucs[-1] if len(aucs) > 0 else 'N/A'}")
            return aucs, times, losses
        except Exception as e:
            logger.error(f"{round_str} 训练过程中出错: {e}")
            raise


    def predict(self, data):
        """
        使用模型进行预测
        
        Args:
            data: 特征数据
            
        Returns:
            np.ndarray: 预测结果
        """
        return np.dot(data, self.parameters)
    
    def evaluate(self, data):
        """
        评估模型性能
    
        Args:
            data: 测试数据元组 (x, y)
        
        Returns:
            dict: 包含各种评估指标的字典
        """
        try:
            from sklearn import metrics
        
            x_test, y_test = data
            pred = self.predict(x_test)
        
            # 计算AUC
            try:
                fpr, tpr, _ = metrics.roc_curve(y_test, pred, pos_label=1)
                auc_score = metrics.auc(fpr, tpr)
            except Exception as e:
                logger.warning(f"计算AUC时出错: {e}")
                auc_score = 0.5  # 默认值
        
            # 计算损失（使用hinge loss）
            try:
                loss = np.mean(np.maximum(0, 1 - y_test * pred))
            except Exception as e:
                logger.warning(f"计算损失时出错: {e}")
                loss = float('inf')
        
            return {
                'auc': auc_score,
                'loss': loss
            }
        except ImportError:
            logger.warning("无法导入sklearn，使用简单评估")
            return {
                'auc': 0.5,  # 默认AUC
                'loss': 0.0  # 默认损失
            }
        except Exception as e:
            logger.error(f"评估模型时出错: {e}")
            return {
                'auc': 0.5,
            'loss': 0.0
        }
 
    def get_parameters(self):
        """获取模型参数"""
        return self.parameters
    
    def set_parameters(self, params):
        """设置模型参数"""
        self.parameters = params
    
    def get_state_dict(self):
        """获取模型状态字典"""
        return {'parameters': self.parameters}
    
    def load_state_dict(self, state_dict):
        """加载模型状态字典"""
        if 'parameters' in state_dict:
            self.parameters = state_dict['parameters']
        else:
            # 尝试直接加载参数
            self.parameters = state_dict
