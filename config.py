"""
统一配置管理模块
"""
import os
import logging
from pathlib import Path

# 默认配置
DEFAULT_CONFIG = {
    # 算法配置
    'algorithm': 'auc_fifo_hinge',
    'feature_dim': 8,
    'learning_rate': 0.001,
    'beta': 0.1,
    'proj_flag': True,
    
    # 训练配置
    'n_pass': 1,
    'local_epochs': 100,
    
    # 联邦学习配置
    'server_address': 'localhost:50051',
    'total_rounds': 30,
    'min_clients': 2,
    'data_split': 0.3,
    
    # 数据配置
    'data_path': 'data/diabetes',
    'data_format': 'libsvm',
    
    # 系统配置
    'log_level': 'INFO',
    'chunk_size': 1024 * 1024,  # 1MB
}

# 配置日志级别映射
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

class Config:
    """配置类，负责加载和管理配置"""
    
    def __init__(self, config_file=None):
        """
        初始化配置
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认配置
        """
        self.config = DEFAULT_CONFIG.copy()
    
        
        # 确保数据路径存在
        self._ensure_data_path()
        
        # 设置日志级别
        self._setup_logging()
    
    def _ensure_data_path(self):
        """确保数据路径存在"""
        data_path = self.config['data_path']
        if not os.path.exists(data_path):
            # 尝试在项目根目录下查找
            project_root = Path(__file__).parent
            alt_path = os.path.join(project_root, data_path)
            if os.path.exists(alt_path):
                self.config['data_path'] = alt_path
            else:
                logging.warning(f"数据路径不存在: {data_path}")
    
    def _setup_logging(self):
        """设置日志级别"""
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=LOG_LEVELS.get(log_level, logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def __getitem__(self, key):
        """允许使用字典语法访问配置"""
        return self.config.get(key)
    
    def __setitem__(self, key, value):
        """允许使用字典语法设置配置"""
        self.config[key] = value
    
    def get(self, key, default=None):
        """获取配置值，如果不存在则返回默认值"""
        return self.config.get(key, default)
    
    def update(self, new_config):
        """更新配置"""
        self.config.update(new_config)
        
    def to_dict(self):
        """返回配置字典"""
        return self.config.copy()

# 创建全局配置实例
CONFIG = Config()

def load_config(config_file):
    """加载配置文件"""
    global CONFIG
    CONFIG = Config(config_file)
    return CONFIG
