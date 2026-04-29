import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import device as torch_device  # 别名避免命名冲突
from typing import Union
from nltk.translate.bleu_score import corpus_bleu
import tkinter as tk
from tkinter import filedialog, ttk
import csv
import os
import chardet
import pickle
from datetime import datetime
from konlpy.tag import Okt
import jieba
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import platform
import psutil  # 新增系统监控库

# ---------------------- GUI模块 ----------------------
class TranslationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("韩中神经机器翻译系统")
        self.setup_ui()
        self.train_thread = None
        
    def setup_ui(self):
        # 主框架布局
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding=10)
        control_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # 文件选择组件
        self.setup_file_controls(control_frame)
        
        # 训练进度条
        self.progress = ttk.Progressbar(control_frame, mode='determinate')
        self.progress.pack(pady=5, fill=tk.X)
        
        # 右侧信息面板
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        
        # 词汇表预览
        self.setup_vocab_preview(info_frame)
        
        # 日志组件
        self.setup_logging(info_frame)
        
        # 配置网格权重
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        self.root.geometry("1000x700")
        
    def setup_file_controls(self, parent):
        file_btn_frame = ttk.Frame(parent)
        file_btn_frame.pack(fill=tk.X, pady=5)
        
        self.train_btn = ttk.Button(file_btn_frame, text="选择训练文件", 
                                  command=self.select_train_files)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.test_btn = ttk.Button(file_btn_frame, text="选择测试文件",
                                 command=self.select_test_files)
        self.test_btn.pack(side=tk.LEFT, padx=5)
        
        self.start_btn = ttk.Button(parent, text="开始训练", 
                                  command=self.start_training)
        self.start_btn.pack(pady=5)
        
        self.stop_btn = ttk.Button(parent, text="停止", state=tk.DISABLED,
                                  command=self.stop_training)
        self.stop_btn.pack(pady=5)
        
    def setup_vocab_preview(self, parent):
        vocab_frame = ttk.LabelFrame(parent, text="词汇表预览", padding=10)
        vocab_frame.pack(fill=tk.BOTH, expand=True)
        
        # 韩语词汇
        self.ko_text = self.create_scroll_text(vocab_frame, "韩语词汇表", 0)
        # 中文词汇
        self.zh_text = self.create_scroll_text(vocab_frame, "中文词汇表", 2)
        
    def setup_logging(self, parent):
        log_frame = ttk.LabelFrame(parent, text="系统日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD)
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
    def create_scroll_text(self, parent, title, column):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=column, padx=5, sticky='nsew')
        
        label = ttk.Label(frame, text=title)
        label.pack()
        
        text = tk.Text(frame, height=10, width=25)
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.LEFT, fill=tk.BOTH)
        return text
        
    def select_train_files(self):
        self.train_files = filedialog.askopenfilenames()
        self.log(f"已选择 {len(self.train_files)} 个训练文件")
        
    def select_test_files(self):
        self.test_files = filedialog.askopenfilenames()
        self.log(f"已选择 {len(self.test_files)} 个测试文件")
        
    def start_training(self):
        if not hasattr(self, 'train_files') or not self.train_files:
            self.log("错误：请先选择训练文件！")
            return
            
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # 启动训练线程
        self.train_thread = threading.Thread(
            target=self.run_training, 
            args=(self.train_files, self.test_files)
        )
        self.train_thread.start()
        
    def stop_training(self):
        if self.train_thread and self.train_thread.is_alive():
            # 设置停止标志
            self.should_stop = True
            self.log("正在停止训练...")
            
    def run_training(self, train_files, test_files):
        try:
            trainer = ModelTrainer(self)
            ko_vocab, zh_vocab = trainer.train(train_files, test_files)
            self.update_vocab(ko_vocab, zh_vocab)
        except Exception as e:
            self.log(f"训练出错：{str(e)}")
        finally:
            self.root.after(0, self.reset_ui)
            
    def update_vocab(self, ko_vocab, zh_vocab):
        self.ko_text.delete(1.0, tk.END)
        self.zh_text.delete(1.0, tk.END)
        
        for word, idx in list(ko_vocab.items())[:50]:
            self.ko_text.insert(tk.END, f"{word}: {idx}\n")
            
        for word, idx in list(zh_vocab.items())[:50]:
            self.zh_text.insert(tk.END, f"{word}: {idx}\n")
            
    def log(self, message):
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        
    def reset_ui(self):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress['value'] = 0

# ---------------------- 数据处理模块 ----------------------
class DataProcessor:
    def __init__(self, gui):
        self.gui = gui
        self.okt = Okt()
        
    def load_dataset(self,file_paths):
        okt = Okt()
        korean_sentences = []
        chinese_sentences = []
        error_log = []
        total_rows = 0
    
        for file_path in file_paths:
            # 文件基础验证
            if not os.path.exists(file_path):
                error_log.append(f"文件不存在: {file_path}")
                continue
            if os.path.getsize(file_path) == 0:
                error_log.append(f"空文件: {file_path}")
                continue
            
            # 读取内容
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'  # 添加默认编码
                content = raw_data.decode(encoding)
            except Exception as e:
                error_log.append(f"文件 {file_path} 读取失败: {str(e)}")
                continue
            
            # 检测CSV格式
            try:
                sample_line = content.split('\n')[0]
                dialect = csv.Sniffer().sniff(sample_line)
                has_header = csv.Sniffer().has_header(sample_line)
            except:
                dialect = csv.excel()
                has_header = False
            
            reader = csv.reader(content.splitlines(), delimiter=dialect.delimiter)
        
            # 跳过标题行
            if has_header:
                try:
                    next(reader)
                except StopIteration:
                    error_log.append(f"文件 {file_path} 仅有标题行")
                    continue
                
            # 处理数据行
            valid_rows = 0
            for row_num, row in enumerate(reader, start=2):
                total_rows += 1
                try:
                    # 列数验证
                    if len(row) < 2:
                        raise ValueError("列数不足")
                    
                    # 分词处理
                    ko_sent = okt.morphs(row[0].strip())
                    zh_sent = list(jieba.cut(row[1].strip()))
                
                    # 结果验证
                    if not ko_sent:
                        raise ValueError("韩语句子分词结果为空")
                    if not zh_sent:
                        raise ValueError("中文句子分词结果为空")
                    
                    korean_sentences.append(ko_sent)
                    chinese_sentences.append(zh_sent)
                    valid_rows += 1
                
                except Exception as e:
                    error_log.append(f"文件 {os.path.basename(file_path)} 第 {row_num} 行: {str(e)}")
                
            self.gui.log(f"文件 {file_path} 有效行数: {valid_rows}/{total_rows}")  # 使用gui的log方法
        
        # 最终数据验证
        if not korean_sentences:
            raise ValueError("所有文件均无有效数据，请检查：\n1. 文件格式是否为CSV\n2. 是否包含中韩双语列\n3. 数据是否包含有效句子")
    
        return korean_sentences, chinese_sentences
        
    def read_file_with_encoding(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, 'rb') as f:
                encoding = chardet.detect(f.read())['encoding']
            with open(path, 'r', encoding=encoding) as f:
                return f.read()
                
    def process_content(self, content, path):
        ko_sents, zh_sents = [], []
        errors = []
        reader = csv.reader(content.splitlines(), delimiter='\t')  # 明确使用制表符分隔
        
        try:
            next(reader)  # 跳过标题
        except StopIteration:
            errors.append(f"{path} 是空文件")
            return ko_sents, zh_sents, errors
            
        for row_idx, row in enumerate(reader, start=2):
            if len(row) < 2:
                errors.append(f"行 {row_idx}: 列数不足")
                continue
                
            try:
                ko = self.okt.morphs(row[0].strip())
                zh = list(jieba.cut(row[1].strip()))
                
                if not ko or not zh:
                    raise ValueError("空句子")
                    
                ko_sents.append(ko)
                zh_sents.append(zh)
            except Exception as e:
                errors.append(f"行 {row_idx}: {str(e)}")
                
        return ko_sents, zh_sents, errors
        
    def generate_report(self, errors, total, ko_sents, zh_sents):
        # 生成统计报告和图表
        pass

# ---------------------- 模型模块 ----------------------
class Seq2SeqModel(nn.Module):
    def __init__(self,
                 input_dim, 
                 output_dim, 
                 emb_dim=256, 
                 hid_dim=512, 
                 n_layers=2, 
                 dropout=0.5, 
                 device: Union[str, torch_device] = 'cuda'  # 联合类型注解
                 ):
        super().__init__()
        self.encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout)
        self.decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout)
        # 统一设备类型处理
        if isinstance(device, str):
            self.device = torch_device(device)
        else:
            self.device = device
        
    def forward(self, src, trg, teacher_ratio=0.5):
         # 新增维度对齐逻辑
        if src.size(0) != trg.size(0):
            min_len = min(src.size(0), trg.size(0))
            src = src[:min_len]
            trg = trg[:min_len]
    
        batch_size = src.size(1)
        trg_len = trg.size(0)
        
        # Encoder
        hidden, cell = self.encoder(src)
        
        # 初始化Decoder输入
        inputs = trg[0]
        outputs = torch.zeros(trg_len, batch_size, self.decoder.output_dim).to(self.device)
        
        # 自回归生成
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(inputs, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_ratio
            top1 = output.argmax(1)
            inputs = trg[t] if teacher_force else top1
            
        return outputs.permute(1, 0, 2)  # (batch, seq, vocab)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, 
                          dropout=dropout, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # 统一维度处理 (seq_len, batch)
        if src.dim() == 1:
            src = src.unsqueeze(1)
        elif src.dim() == 3:
            src = src.squeeze(0)
            
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self,
                output_dim: int,  # 明确类型注解
                emb_dim: int,
                hid_dim: int,     # 必须是整数
                n_layers: int,
                dropout: float
                ):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim  # 保存为整数
        # 嵌入层
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # LSTM层（参数必须为整数）
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        # 全连接层
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # 新增维度验证逻辑
        if hidden.size(1) != input.size(0):
            hidden = hidden[:, :input.size(0), :].contiguous()
            cell = cell[:, :input.size(0), :].contiguous()
            
        input = input.unsqueeze(0)  # (1, batch_size)
        embedded = self.dropout(self.embedding(input))  # (1, batch_size, emb_dim)
        
        # LSTM前向传播
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

# ---------------------- 训练模块 ----------------------
class ModelTrainer:
    def __init__(self, gui):
        self.gui = gui
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.should_stop = False
        self.zh_vocab_inv = None  # 新增反向词汇表
    def print_training_progress(self):
        # 获取显存使用情况（针对NVIDIA GPU）
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated()/1e9
            self.gui.log(f"GPU显存占用: {mem:.2f}GB")
        # 获取CPU使用率
        self.gui.log(f"CPU使用率: {psutil.cpu_percent()}%")

    def train(self, train_files, test_files):
        # 数据加载
        processor = DataProcessor(self.gui)
        ko_train, zh_train = processor.load_dataset(train_files)
        ko_test, zh_test = processor.load_dataset(test_files)

        # 构建词汇表
        ko_vocab = self.build_vocab(ko_train)
        zh_vocab = self.build_vocab(zh_train)
        
        # 构建反向词汇表
        self.zh_vocab_inv = {v: k for k, v in zh_vocab.items()}
        
        # 准备数据
        train_loader = self.prepare_data(ko_train, zh_train, ko_vocab, zh_vocab)
        test_loader = self.prepare_data(ko_test, zh_test, ko_vocab, zh_vocab)

        # 初始化模型
        model = Seq2SeqModel(
            input_dim=len(ko_vocab),
            output_dim=len(zh_vocab),
            device=self.device
        ).to(self.device)

        # 训练配置
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

        # 训练循环
        best_loss = float('inf')
        for epoch in range(10):
            if self.should_stop:
                break

            # 训练阶段
            model.train()
            total_loss = 0
            # 新增进度监控（每5个batch记录一次）
            batch_count = 0
            for src, trg in train_loader:
                src, trg = src.to(self.device), trg.to(self.device)
                optimizer.zero_grad()
                output = model(src.permute(1, 0), trg.permute(1, 0))  # 调整为 (seq_len, batch)
                loss = self.calculate_loss(output, trg, criterion)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                # 新增监控逻辑
                batch_count += 1
                if batch_count % 5 == 0:
                    self.print_training_progress()
                    self.gui.log(f"当前batch处理进度: {batch_count}/{len(train_loader)}")
            # 验证阶段（修改参数传递方式）
            val_loss, bleu1, bleu4 = self.evaluate(model, test_loader, criterion)
            scheduler.step(val_loss)  # 明确使用验证损失的数值部分

            # 更新GUI日志（同步修改日志格式）
            self.gui.log(f"Epoch {epoch+1}: "
                        f"Train Loss {total_loss/len(train_loader):.3f} | "
                        f"Val Loss {val_loss:.3f} | " 
                        f"BLEU-1 {bleu1:.2f} | BLEU-4 {bleu4:.2f}")
            self.gui.progress['value'] = (epoch+1)*10

        # 在训练结束后保存模型和词汇表
        self.save_artifacts(model, ko_vocab, zh_vocab)

        return ko_vocab, zh_vocab
        
    def prepare_data(self, ko_sents, zh_sents, src_vocab, trg_vocab):
        """将文本数据转换为填充后的张量，并返回 DataLoader"""
        max_length = 100  # 新增最大长度限制
        # 转换为索引序列（添加截断逻辑）
        src_tensors = [
            torch.tensor(
                [src_vocab['<sos>']] + 
                [src_vocab.get(word, src_vocab['<pad>']) for word in sent[:max_length]] + 
                [src_vocab['<eos>']], 
                dtype=torch.long
            ) for sent in ko_sents
        ]
        trg_tensors = [
            torch.tensor(
                [trg_vocab['<sos>']] + 
                [trg_vocab.get(word, trg_vocab['<pad>']) for word in sent[:max_length]] + 
                [trg_vocab['<eos>']], 
                dtype=torch.long
            ) for sent in zh_sents
        ]

        # 填充序列
        src_padded = torch.nn.utils.rnn.pad_sequence(
            src_tensors, 
            padding_value=src_vocab['<pad>'],
            batch_first=True
        )
        trg_padded = torch.nn.utils.rnn.pad_sequence(
            trg_tensors,
            padding_value=trg_vocab['<pad>'],
            batch_first=True
        )

        # 创建数据集
        dataset = TensorDataset(src_padded, trg_padded)
        
        # 返回 DataLoader
        return DataLoader(
            dataset,
            batch_size=8,  # 从32减少到8
            shuffle=True,
            num_workers=0,  # 在Windows系统保持为0
            drop_last=True
        )
        
    def calculate_loss(self, output, trg, criterion):
        # 计算损失
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        return criterion(output, trg)
        
    def evaluate(self, model, data, criterion):
        # 模型验证
        model.eval()
        total_loss = 0
        all_trg = []
        all_pred = []
        
        # 添加类型断言
        assert self.zh_vocab_inv is not None, "反向词汇表未初始化"
        
        with torch.no_grad():
            for src, trg in data:
                output = model(src, trg, teacher_ratio=0)
                loss = self.calculate_loss(output, trg, criterion)
                total_loss += loss.item()
                
                # 收集预测结果和真实标签
                pred = output.argmax(dim=-1).cpu().numpy()
                trg = trg[:, 1:].cpu().numpy()  # 跳过<sos>
                
                # 转换为单词列表（添加安全转换）
                for i in range(len(trg)):
                    ref = [[self.zh_vocab_inv.get(idx, '<unk>') for idx in trg[i] if idx not in [0, 2]]]
                    hyp = [self.zh_vocab_inv.get(idx, '<unk>') for idx in pred[i] if idx not in [0, 2]]
                    
                    all_trg.append(ref)
                    all_pred.append(hyp)
        
        # 计算BLEU分数
        bleu4 = corpus_bleu(all_trg, all_pred, weights=(0.25, 0.25, 0.25, 0.25))
        bleu1 = corpus_bleu(all_trg, all_pred, weights=(1, 0, 0, 0))
        
        return total_loss / len(data), bleu1, bleu4
        
    def build_vocab(self, sentences):
        # 词汇表构建
        vocab = {'<pad>':0, '<sos>':1, '<eos>':2}
        idx = 3
        for sent in sentences:
            for word in sent:
                if word not in vocab:
                    vocab[word] = idx
                    idx +=1
        return vocab
        
    def save_artifacts(self, model, ko_vocab, zh_vocab):
        """保存模型和词汇表"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(os.getcwd(), "Translate Model")
        
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存词汇表（直接保存在Translate Model目录）
            ko_vocab_file = os.path.join(save_dir, f"ko_vocab_{timestamp}.pkl")
            zh_vocab_file = os.path.join(save_dir, f"zh_vocab_{timestamp}.pkl")
            with open(ko_vocab_file, 'wb') as f:
                pickle.dump(ko_vocab, f)
            with open(zh_vocab_file, 'wb') as f:
                pickle.dump(zh_vocab, f)
                
            # 保存模型（直接保存在Translate Model目录）
            model_file = os.path.join(save_dir, f"translation_model_{timestamp}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'input_dim': len(ko_vocab),
                    'output_dim': len(zh_vocab)
                }
            }, model_file)
            
            self.gui.log(f"模型和词汇表已直接保存至：{save_dir}")
        except Exception as e:
            self.log(f"训练出错：{str(e)}")
        finally:
            self.gui.root.after(0, self.reset_ui)
            
    def update_vocab(self, ko_vocab, zh_vocab):
        self.gui.ko_text.delete(1.0, tk.END)
        self.gui.zh_text.delete(1.0, tk.END)
        
        for word, idx in list(ko_vocab.items())[:50]:
            self.gui.ko_text.insert(tk.END, f"{word}: {idx}\n")
            
        for word, idx in list(zh_vocab.items())[:50]:
            self.gui.zh_text.insert(tk.END, f"{word}: {idx}\n")
            
    def log(self, message):
        self.gui.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.gui.log_text.see(tk.END)
        
    def reset_ui(self):
        self.gui.start_btn.config(state=tk.NORMAL)
        self.gui.stop_btn.config(state=tk.DISABLED)
        self.gui.progress['value'] = 0

if __name__ == "__main__":
    if platform.system() == 'Windows':
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
        
    app = TranslationGUI()
    app.root.mainloop()