import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from tqdm import tqdm # 进度条库
from src import config, utils
import logging
from src.evaluator import evaluate_model, calculate_similarity_metrics

class EarlyStopping:
    """早停机制，防止过拟合"""
    def __init__(self, patience=5, verbose=True, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta # 最小改进量
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss # 分数越高越好

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} (Best val loss: {self.val_loss_min:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''保存模型当验证损失下降时'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path} ...')
        utils.save_model(model, self.path) # 使用 utils 中的保存函数
        self.val_loss_min = val_loss

def get_optimizer(model, cfg):
    """根据配置获取优化器"""
    if cfg.OPTIMIZER.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPTIMIZER.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPTIMIZER.lower() == 'sgd':
        # SGD 可能需要不同的学习率和 momentum 设置
        return optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=0.9, weight_decay=cfg.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.OPTIMIZER}")

def get_loss_function(cfg):
    """根据配置获取损失函数"""
    if cfg.LOSS_FN.lower() == 'mse':
        return nn.MSELoss()
    elif cfg.LOSS_FN.lower() == 'mae':
        return nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {cfg.LOSS_FN}")


class BaseTrainer:
    """训练器的基类，包含通用训练和验证逻辑"""
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device, epochs, patience, model_save_path, scaler, config_obj, model_name="Model"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        self.model_save_path = model_save_path
        self.model_name = model_name
        self.scaler = scaler # Add scaler to instance attributes
        self.config = config_obj # Add config object to instance attributes
        self.early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_save_path, trace_func=logging.info)
        self.history = {'train_loss': [], 'val_loss': [], 'lr': [], 'grad_norm': []}

    def _train_epoch(self):
        raise NotImplementedError

    def _validate_epoch(self):
        raise NotImplementedError

    def train(self):
        logging.info(f"\n--- Starting Training for {self.model_name} ---")
        start_time = time.time()
        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            # --- 训练 ---
            train_loss, grad_norm = self._train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['grad_norm'].append(grad_norm)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr']) # 记录当前学习率

            # --- 验证 ---
            val_loss, metrics_dict, _, _ = self._validate_epoch() # _validate_epoch 现在返回损失、指标、真实值和预测值
            self.history['val_loss'].append(val_loss)
            # 记录验证指标
            for metric_name, value in metrics_dict.items():
                if f'val_{metric_name}' not in self.history:
                    self.history[f'val_{metric_name}'] = []
                self.history[f'val_{metric_name}'].append(value)

            epoch_duration = time.time() - epoch_start_time
            log_message = f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            for metric_name, value in metrics_dict.items():
                log_message += f" | Val {metric_name.upper()}: {value:.6f}"
            log_message += f" | Time: {epoch_duration:.2f}s"
            logging.info(log_message)

            # --- 早停检查 ---
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                logging.info("Early stopping triggered.")
                break

            if hasattr(self, 'alpha_scheduler') and getattr(self.alpha_scheduler, 'abandon_student', False):
                logging.info("Alpha gating triggered. Abandoning student model.")
                break

        total_time = time.time() - start_time
        logging.info(f"--- Training Finished for {self.model_name} ---")
        logging.info(f"Total Training Time: {total_time:.2f}s")
        logging.info(f"Best validation loss achieved: {self.early_stopping.val_loss_min:.6f}")
        logging.info(f"Model saved to: {self.model_save_path}")

        # 加载性能最好的模型
        self.model = utils.load_model(self.model, self.model_save_path, self.device)

        # 绘制训练指标曲线
        plot_save_dir = os.path.join(self.config.PLOTS_DIR, self.model_name.lower().replace(" ", "_"))
        utils.plot_training_metrics(self.history, plot_save_dir, self.model_name)
        
        # 绘制权重和偏置分布
        utils.plot_weights_biases_distribution(self.model, plot_save_dir, self.model_name)

        return self.model, self.history


class StandardTrainer(BaseTrainer):
    """标准训练流程 (用于教师模型或基线学生模型)"""
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device, epochs, patience, model_save_path, scaler, config_obj, model_name="StandardModel"):
        super().__init__(model, train_loader, val_loader, optimizer, loss_fn, device, epochs, patience, model_save_path, scaler, config_obj, model_name)

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        # 使用 tqdm 创建进度条
        train_iterator = tqdm(self.train_loader, desc=f"Epoch {len(self.history['train_loss']) + 1}/{self.epochs} Training", leave=False)
        for batch_x, batch_y, batch_hist_exog, batch_futr_exog in train_iterator:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()

            input_dict = {'insample_y': batch_x}
            # Always add exog keys, setting to None if not provided by loader
            input_dict['hist_exog'] = batch_hist_exog.to(self.device) if batch_hist_exog is not None else None
            input_dict['futr_exog'] = batch_futr_exog.to(self.device) if batch_futr_exog is not None else None

            outputs = self.model(input_dict) # [batch, horizon, features]

            loss = self.loss_fn(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            train_iterator.set_postfix(loss=loss.item()) # 在进度条上显示当前 batch loss

        # 计算梯度范数
        total_grad_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        return total_loss / len(self.train_loader), total_grad_norm

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_true_labels = []

        val_iterator = tqdm(self.val_loader, desc=f"Epoch {len(self.history['val_loss']) + 1}/{self.epochs} Validation", leave=False)
        with torch.no_grad():
            for batch_x, batch_y, batch_hist_exog, batch_futr_exog in val_iterator:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                input_dict = {'insample_y': batch_x}
                input_dict['hist_exog'] = batch_hist_exog.to(self.device) if batch_hist_exog is not None else None
                input_dict['futr_exog'] = batch_futr_exog.to(self.device) if batch_futr_exog is not None else None

                outputs = self.model(input_dict)

                loss = self.loss_fn(outputs, batch_y)
                total_loss += loss.item()
                val_iterator.set_postfix(loss=loss.item())

                all_preds.append(outputs.cpu())
                all_true_labels.append(batch_y.cpu())

        avg_val_loss = total_loss / len(self.val_loader)
        
        preds_all = torch.cat(all_preds, dim=0) # Keep as tensor for evaluate_model
        true_labels_all = torch.cat(all_true_labels, dim=0) # Keep as tensor for evaluate_model
        
        # evaluate_model 期望原始尺度的预测和真实值，所以需要逆变换
        # 但 evaluate_model 内部会处理逆变换，这里直接传入 scaled 的 tensor
        # evaluate_model 返回 metrics, true_values_original, predictions_original
        val_metrics, true_values_original, predictions_original = evaluate_model(
            self.model, self.val_loader, device=self.device, scaler=self.scaler,
            config_obj=self.config, logger=logging, dataset_type="Validation Set"
        )

        return avg_val_loss, val_metrics, true_values_original, predictions_original


class RDT_Trainer(BaseTrainer):
    """RDT 训练流程"""
    def __init__(self, student_model, teacher_model, train_loader, val_loader, optimizer, task_loss_fn, distill_loss_fn, alpha_scheduler, device, epochs, patience, model_save_path, scaler, config_obj, model_name="RDT_Student"):
        super().__init__(student_model, train_loader, val_loader, optimizer, task_loss_fn, device, epochs, patience, model_save_path, scaler, config_obj, model_name) # loss_fn 作为 task_loss_fn
        self.teacher_model = teacher_model.to(device).eval() # 教师模型设为评估模式且不训练
        self.distill_loss_fn = distill_loss_fn
        self.alpha_scheduler = alpha_scheduler
        self.history['alpha'] = [] # 记录 alpha 变化

    def _train_epoch(self):
        self.model.train() # student model 设置为训练模式
        self.teacher_model.eval() # teacher model 保持评估模式
        total_loss = 0
        total_task_loss = 0
        total_distill_loss = 0
        current_epoch = len(self.history['train_loss']) # 当前 epoch (从 0 开始)
        alpha = self.alpha_scheduler.get_alpha(current_epoch)
        self.history['alpha'].append(alpha) # 记录当前 alpha

        train_iterator = tqdm(self.train_loader, desc=f"Epoch {current_epoch + 1}/{self.epochs} RDT Training (alpha={alpha:.3f})", leave=False)

        for batch_x, batch_y_true, batch_hist_exog, batch_futr_exog in train_iterator:
            batch_x, batch_y_true = batch_x.to(self.device), batch_y_true.to(self.device)

            # Prepare input dicts, always including exog keys
            input_dict_base = {'insample_y': batch_x}
            input_dict_base['hist_exog'] = batch_hist_exog.to(self.device) if batch_hist_exog is not None else None
            input_dict_base['futr_exog'] = batch_futr_exog.to(self.device) if batch_futr_exog is not None else None

            # 1. 教师模型预测 (不计算梯度)
            with torch.no_grad():
                input_dict_teacher = input_dict_base.copy() # Use copy just in case
                batch_y_teacher = self.teacher_model(input_dict_teacher)

            # 2. 学生模型预测
            self.optimizer.zero_grad()
            input_dict_student = input_dict_base.copy() # Use copy just in case
            batch_y_student = self.model(input_dict_student) # self.model is student

            # 3. 计算损失
            loss_task = self.loss_fn(batch_y_student, batch_y_true)
            # detach教师预测，阻止梯度流向教师
            loss_distill = self.distill_loss_fn(batch_y_student, batch_y_teacher.detach())

            # 4. 计算总损失
            loss_total = alpha * loss_task + (1 - alpha) * loss_distill

            # 5. 反向传播和优化 (只更新学生模型)
            loss_total.backward()
            # 计算梯度范数
            total_grad_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            self.optimizer.step()

            total_loss += loss_total.item()
            total_task_loss += loss_task.item()
            total_distill_loss += loss_distill.item()
            train_iterator.set_postfix(loss=loss_total.item(), task_loss=loss_task.item(), dist_loss=loss_distill.item())

        avg_total_loss = total_loss / len(self.train_loader)
        avg_task_loss = total_task_loss / len(self.train_loader)
        avg_distill_loss = total_distill_loss / len(self.train_loader)
        logging.info(f"Epoch {current_epoch+1} Avg Train Losses: Total={avg_total_loss:.6f}, Task={avg_task_loss:.6f}, Distill={avg_distill_loss:.6f}")
        logging.info(f"Current Alpha: {alpha:.6f}") # Log alpha

        return avg_total_loss, total_grad_norm # 返回总损失和梯度范数

    def _validate_epoch(self):
        # RDT 的验证通常只关心学生模型在真实任务上的表现 (Task Loss)
        self.model.eval()
        self.teacher_model.eval() # 确保教师模型也处于评估模式
        total_task_loss = 0
        
        all_student_preds = []
        all_teacher_preds = []
        all_true_labels = []

        val_iterator = tqdm(self.val_loader, desc=f"Epoch {len(self.history['val_loss']) + 1}/{self.epochs} RDT Validation", leave=False)
        with torch.no_grad():
            for batch_x, batch_y_true, batch_hist_exog, batch_futr_exog in val_iterator:
                batch_x, batch_y_true = batch_x.to(self.device), batch_y_true.to(self.device)

                input_dict = {'insample_y': batch_x}
                # Always add exog keys, setting to None if not provided by loader
                input_dict['hist_exog'] = batch_hist_exog.to(self.device) if batch_hist_exog is not None else None
                input_dict['futr_exog'] = batch_futr_exog.to(self.device) if batch_futr_exog is not None else None

                batch_y_student = self.model(input_dict)
                batch_y_teacher = self.teacher_model(input_dict) # 获取教师模型预测

                loss_task = self.loss_fn(batch_y_student, batch_y_true) # 只计算 Task Loss
                total_task_loss += loss_task.item()
                val_iterator.set_postfix(val_task_loss=loss_task.item())

                all_student_preds.append(batch_y_student.cpu())
                all_teacher_preds.append(batch_y_teacher.cpu())
                all_true_labels.append(batch_y_true.cpu())

        avg_val_loss = total_task_loss / len(self.val_loader)
        
        # 将所有批次的预测结果连接起来
        student_preds_all = torch.cat(all_student_preds, dim=0)
        teacher_preds_all = torch.cat(all_teacher_preds, dim=0)
        true_labels_all = torch.cat(all_true_labels, dim=0)

        # 将 scaled 的预测结果逆变换回原始尺度，以便计算相似度
        # evaluate_model 内部会处理 student_preds_all 的逆变换
        # 但 teacher_preds_all 需要在这里逆变换，因为 evaluate_model 不会处理它
        n_samples, horizon, n_features = teacher_preds_all.shape
        teacher_preds_reshaped = teacher_preds_all.view(-1, n_features)
        try:
            teacher_preds_original = self.scaler.inverse_transform(teacher_preds_reshaped)
        except Exception as e:
            logging.error(f"Error during inverse transform of teacher predictions: {e}. Using scaled values for similarity.")
            teacher_preds_original = teacher_preds_reshaped
        teacher_preds_original = teacher_preds_original.reshape(n_samples, horizon, n_features)

        # 计算验证集上的评估指标
        val_metrics, _, _ = evaluate_model(
            self.model, self.val_loader, self.device, self.scaler, self.config, logging,
            model_name=self.model_name, plots_dir=os.path.join(self.config.RESULTS_DIR, "plots"),
            teacher_predictions_original=teacher_preds_original, # 传入原始尺度的教师预测
            dataset_type="Validation Set"
        )
        # 计算学生-教师相似度
        simi_student_teacher = calculate_similarity_metrics(student_preds_all, teacher_preds_all, metric_type=self.config.SIMILARITY_METRIC)
        val_metrics.update(simi_student_teacher) # 将相似度指标直接合并到 val_metrics 字典中

        # 在这里将收集到的预测结果传递给 Alpha 调度器进行更新
        # AlphaScheduler 的 update 方法需要这些信息来动态调整 alpha
        current_epoch = len(self.history['val_loss']) # 验证阶段的当前 epoch
        self.alpha_scheduler.update(current_epoch, avg_val_loss, student_preds_all, teacher_preds_all, true_labels_all)

        # 注意：早停是基于这个验证 Task Loss
        return avg_val_loss, val_metrics, true_labels_all, student_preds_all # 返回所有需要的值
