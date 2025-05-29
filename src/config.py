import torch
import os
from datetime import datetime

class Config:
    def __init__(self):
        # --- 基本配置 ---
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, 'results')
        self.LOG_DIR = os.path.join(self.BASE_DIR, 'log')
        self.PLOTS_DIR = os.path.join(self.RESULTS_DIR, 'plots')

        # --- 数据集配置 ---
        self.DATASET_PATH = os.path.join(self.DATA_DIR, 'national_illness.csv')
        self.DATE_COL = 'date'
        self.TARGET_COLS = ['OT']
        self.TIME_FREQ = 'D'

        # --- 数据处理配置 ---
        self.LOOKBACK_WINDOW = 96
        self.PREDICTION_HORIZON = 192
        self.VAL_SPLIT_RATIO = 0.43
        self.TEST_SPLIT_RATIO = 0.3
        self.BATCH_SIZE = 32 # Increased for better GPU utilization
        self.NUM_WORKERS = 4  # Increased for faster data loading

        # --- 模型配置 ---
        self.TEACHER_MODEL_NAME = 'DLinear'
        self.TEACHER_CONFIG = {
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'n_series': len(self.TARGET_COLS),
            'hidden_size': 256,
            'moving_avg_window': 25,
        }

        self.STUDENT_MODEL_NAME = 'PatchTST'
        self.STUDENT_CONFIG = {
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'n_series': len(self.TARGET_COLS),
            'patch_len': 16,
            'stride': 8,
            'n_layers': 3,
            'n_heads': 4,
            'hidden_size': 128,
            'ff_hidden_size': 256,
            'revin': True,
        }

        # --- 训练配置 ---
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.EPOCHS = 2
        self.LEARNING_RATE = 0.001
        self.OPTIMIZER = 'Adam'
        self.WEIGHT_DECAY = 1e-5
        self.PATIENCE = 50
        self.LOSS_FN = 'MSE'
        self.USE_AMP = True # Enable Automatic Mixed Precision (AMP)

        # --- RDT 配置 ---
        self.ALPHA_START = 0.3
        self.ALPHA_END = 0.7
        self.ALPHA_SCHEDULE = 'linear'
        self.CONSTANT_ALPHA = 0.5
        # 门控策略
        self.USE_ALPHA_GATING = False
        self.GATING_THRESHOLD = 0.01
        self.GATING_PATIENCE = 2

        # --- 评估配置 ---
        self.METRICS = ['mae', 'mse']
        self.ROBUSTNESS_NOISE_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]

        # --- 新增数据处理和评估配置 ---
        self.TRAIN_NOISE_INJECTION_LEVEL = 0.0
        self.VAL_NOISE_INJECTION_LEVEL = 0.0
        self.NOISE_TYPE = 'gaussian'

        self.SMOOTHING_METHOD = 'none'
        self.SMOOTHING_FACTOR = 0.5
        self.SMOOTHING_APPLY_TRAIN = False
        self.SMOOTHING_APPLY_VAL = False
        self.SMOOTHING_APPLY_TEST = False
        self.SMOOTHING_WEIGHT_SMOOTHING = 0.0

        self.SIMILARITY_METRIC = 'cosine_similarity'

        self.STABILITY_RUNS = 1

        # --- 实验管理 ---
        self.SEED = 42
        self.EXPERIMENT_NAME = f"RDT_{self.STUDENT_MODEL_NAME}_vs_{self.TEACHER_MODEL_NAME}_h{self.PREDICTION_HORIZON}"

        # --- 日志配置 ---
        self.LOG_LEVEL = 'INFO'
        self.LOG_FILE_NAME = f"{self.EXPERIMENT_NAME}_{os.path.basename(self.DATASET_PATH).split('.')[0]}.log"
        self.LOG_FILE_PATH = os.path.join(self.LOG_DIR, self.LOG_FILE_NAME)

        # --- 新增模型参数设置 ---
        self.NLINEAR_CONFIG = {
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'n_series': len(self.TARGET_COLS),
        }
        self.MLP_CONFIG = {
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'n_series': len(self.TARGET_COLS),
            'hidden_size': 512,
            'num_layers': 2,
            'activation': 'relu',
            'dropout': 0.1
        }
        self.RNN_CONFIG = {
            'n_series': len(self.TARGET_COLS),
            'lookback': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.1,
            'output_size': len(self.TARGET_COLS)
        }
        self.LSTM_CONFIG = {
            'n_series': len(self.TARGET_COLS),
            'lookback': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.1,
            'output_size': len(self.TARGET_COLS)
        }
        self.AUTOFORMER_CONFIG = {
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'n_series': len(self.TARGET_COLS),
            'hidden_size': 64,
            'n_head': 8,
            'encoder_layers': 2,
            'decoder_layers': 1,
            'd_ff': 256,
            'moving_avg': 25,
            'dropout': 0.05,
            'activation': 'gelu',
        }
        self.INFORMER_CONFIG = {
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'n_series': len(self.TARGET_COLS),
            'hidden_size': 64,
            'n_head': 8,
            'encoder_layers': 2,
            'decoder_layers': 1,
            'd_ff': 256,
            'dropout': 0.05,
            'activation': 'gelu',
        }
        self.FEDFORMER_CONFIG = {
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'n_series': len(self.TARGET_COLS),
            'hidden_size': 64,
            'n_head': 8,
            'encoder_layers': 2,
            'decoder_layers': 1,
            'd_ff': 256,
            'moving_avg': 25,
            'dropout': 0.05,
            'activation': 'gelu',
        }

        # Dynamic updates (now within the class)
        self.TEACHER_CONFIG['n_series'] = len(self.TARGET_COLS)
        self.STUDENT_CONFIG['n_series'] = len(self.TARGET_COLS)
        self.NLINEAR_CONFIG['n_series'] = len(self.TARGET_COLS)
        self.MLP_CONFIG['n_series'] = len(self.TARGET_COLS)
        self.RNN_CONFIG['n_series'] = len(self.TARGET_COLS)
        self.RNN_CONFIG['output_size'] = len(self.TARGET_COLS)
        self.LSTM_CONFIG['n_series'] = len(self.TARGET_COLS)
        self.LSTM_CONFIG['output_size'] = len(self.TARGET_COLS)
        self.AUTOFORMER_CONFIG['n_series'] = len(self.TARGET_COLS)
        self.INFORMER_CONFIG['n_series'] = len(self.TARGET_COLS)
        self.FEDFORMER_CONFIG['n_series'] = len(self.TARGET_COLS)
        self.MLP_CONFIG['input_size'] = self.LOOKBACK_WINDOW
        self.RNN_CONFIG['lookback'] = self.LOOKBACK_WINDOW
        self.LSTM_CONFIG['lookback'] = self.LOOKBACK_WINDOW

    def update_model_configs(self):
        """
        在 LOOKBACK_WINDOW 或 PREDICTION_HORIZON 改变后，
        动态更新所有依赖这些值的模型配置。
        """
        # 更新教师模型配置
        self.TEACHER_CONFIG.update({
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'n_series': len(self.TARGET_COLS),
        })
        # 更新学生模型配置
        self.STUDENT_CONFIG.update({
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'n_series': len(self.TARGET_COLS),
        })
        # 更新其他模型配置
        self.NLINEAR_CONFIG.update({
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'n_series': len(self.TARGET_COLS),
        })
        self.MLP_CONFIG.update({
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'n_series': len(self.TARGET_COLS),
        })
        self.RNN_CONFIG.update({
            'lookback': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'n_series': len(self.TARGET_COLS),
            'output_size': len(self.TARGET_COLS),
        })
        self.LSTM_CONFIG.update({
            'lookback': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'n_series': len(self.TARGET_COLS),
            'output_size': len(self.TARGET_COLS),
        })
        self.AUTOFORMER_CONFIG.update({
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'n_series': len(self.TARGET_COLS),
        })
        self.INFORMER_CONFIG.update({
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'n_series': len(self.TARGET_COLS),
        })
        self.FEDFORMER_CONFIG.update({
            'input_size': self.LOOKBACK_WINDOW,
            'h': self.PREDICTION_HORIZON,
            'n_series': len(self.TARGET_COLS),
        })
        # 确保 EXPERIMENT_NAME 也更新
        self.EXPERIMENT_NAME = f"RDT_{self.STUDENT_MODEL_NAME}_vs_{self.TEACHER_MODEL_NAME}_h{self.PREDICTION_HORIZON}"
        self.LOG_FILE_NAME = f"{self.EXPERIMENT_NAME}_{os.path.basename(self.DATASET_PATH).split('.')[0]}.log"
        self.LOG_FILE_PATH = os.path.join(self.LOG_DIR, self.LOG_FILE_NAME)
