import re
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
import tkinter as tk
from tkinter import filedialog
import csv
from datetime import datetime
import os
import chardet
import pickle
from konlpy.tag import Okt
import jieba
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 防止GUI冲突
import random
# 创建交互界面
# 在create_gui函数中添加日志组件
def create_gui():
    root = tk.Tk()
    root.title("语料库文件选择和保存")
    
    # 创建主容器使用Grid布局
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # 左侧面板（文件操作和词汇表预览区域）
    left_frame = tk.Frame(main_frame)
    left_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
    
    # 右侧面板（日志区域）
    right_frame = tk.Frame(main_frame)
    right_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
    
    # 在现有组件后添加词汇表预览区域
    vocab_frame = tk.Frame(left_frame)
    vocab_frame.pack(pady=10)

    # 韩语词汇表预览
    ko_label = tk.Label(vocab_frame, text="韩语词汇表预览")
    ko_label.grid(row=0, column=0, padx=5)
    ko_scroll = tk.Scrollbar(vocab_frame)
    ko_text = tk.Text(vocab_frame, height=10, width=25, yscrollcommand=ko_scroll.set)
    ko_scroll.config(command=ko_text.yview)
    ko_text.grid(row=1, column=0, padx=5)
    ko_scroll.grid(row=1, column=1, sticky='ns')

    # 中文词汇表预览
    zh_label = tk.Label(vocab_frame, text="中文词汇表预览")
    zh_label.grid(row=0, column=2, padx=5)
    zh_scroll = tk.Scrollbar(vocab_frame)
    zh_text = tk.Text(vocab_frame, height=10, width=25, yscrollcommand=zh_scroll.set)
    zh_scroll.config(command=zh_text.yview)
    zh_text.grid(row=1, column=2, padx=5)
    zh_scroll.grid(row=1, column=3, sticky='ns')

    # 在训练函数中添加词汇表更新逻辑
    def update_vocab_preview(ko_vocab, zh_vocab):
        ko_text.delete(1.0, tk.END)
        zh_text.delete(1.0, tk.END)
        
        # 显示前50个词汇（带滚动条可查看全部）
        for word, idx in list(ko_vocab.items())[:40]:
            ko_text.insert(tk.END, f"{word}: {idx}\n")
        for word, idx in list(zh_vocab.items())[:40]:
            zh_text.insert(tk.END, f"{word}: {idx}\n")

    # 修改后的训练函数
    def train_with_selected_files():
        if not train_file_paths or not test_file_paths:
            status_label.config(text="请先选择训练集和测试集文件")
            return
        # 直接传递文件路径列表
        ko_vocab, zh_vocab = train_model(train_file_paths, test_file_paths)
        update_vocab_preview(ko_vocab, zh_vocab)
        status_label.config(text="模型训练完成")

    # 全局变量用于存储文件路径
    global train_file_paths, test_file_paths
    train_file_paths = []
    test_file_paths = []

    # 显示文件路径的文本框
    train_file_text = tk.Text(left_frame, height=5, width=50)
    train_file_text.pack(pady=5)
    test_file_text = tk.Text(left_frame, height=5, width=50)
    test_file_text.pack(pady=5)

    # 选择训练集文件按钮
    def select_train_files():
        global train_file_paths
        train_file_paths = filedialog.askopenfilenames()
        # 清空文本框
        train_file_text.delete(1.0, tk.END)
        # 显示文件路径
        for path in train_file_paths:
            train_file_text.insert(tk.END, path + "\n")
        status_label.config(text=f"训练集文件: {len(train_file_paths)} 个")

    # 选择测试集文件按钮
    def select_test_files():
        global test_file_paths
        test_file_paths = filedialog.askopenfilenames()
        # 清空文本框
        test_file_text.delete(1.0, tk.END)
        # 显示文件路径
        for path in test_file_paths:
            test_file_text.insert(tk.END, path + "\n")
        status_label.config(text=f"测试集文件: {len(test_file_paths)} 个")

    # 开始按钮
    start_button = tk.Button(left_frame, text="开始训练", command=train_with_selected_files)
    start_button.pack(pady=10)

    # 停止按钮（目前只是占位，可根据需求添加停止逻辑）
    stop_button = tk.Button(left_frame, text="停止", command=lambda: status_label.config(text="停止操作未实现"))
    stop_button.pack(pady=10)

    # 选择训练集文件按钮
    select_train_button = tk.Button(left_frame, text="选择训练集文件", command=select_train_files)
    select_train_button.pack(pady=10)

    # 选择测试集文件按钮
    select_test_button = tk.Button(left_frame, text="选择测试集文件", command=select_test_files)
    select_test_button.pack(pady=10)

    status_label = tk.Label(left_frame, text="")
    status_label.pack(pady=10)

    # 日志区域修改到右侧
    log_label = tk.Label(right_frame, text="处理日志")
    log_label.pack(anchor='w')
    
    global log_text
    log_scroll = tk.Scrollbar(right_frame)
    log_text = tk.Text(right_frame, height=25, width=60, yscrollcommand=log_scroll.set)
    log_scroll.config(command=log_text.yview)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    # 配置网格列权重
    main_frame.columnconfigure(0, weight=3)  # 左侧占3份
    main_frame.columnconfigure(1, weight=2)  # 右侧日志占2份
    root.geometry("800x640")  # 调整初始窗口大小

    root.mainloop()  # 确认这行存在且是最后执行的
# 修改后的加载函数
def load_dataset(file_paths):
    okt = Okt()
    korean_sentences = []
    chinese_sentences = []
    error_log = []
    total_rows = 0
    error_count = 0
    length_distribution = {'ko': [], 'zh': []}
    
    for file_path in file_paths:
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            try:
                # Fall back to GBK if UTF-8 fails
                with open(file_path, 'r', encoding='gbk') as file:
                    content = file.read()
            except UnicodeDecodeError:
                # If both fail, try chardet
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    encoding = chardet.detect(raw_data)['encoding']
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
        
        # Process the content
        lines = content.splitlines()
        reader = csv.reader(lines)
        
        try:
            next(reader)  # Skip header
        except StopIteration:
            error_msg = f"文件 {os.path.basename(file_path)} 是空文件"
            log_text.insert(tk.END, error_msg + '\n')
            error_log.append(error_msg)
            continue
            
        for row_num, row in enumerate(reader, start=2):
            total_rows += 1
            try:
                if len(row) < 2:
                    raise ValueError("列数不足")
                    
                ko_sentence = okt.morphs(row[0].strip())
                zh_sentence = list(jieba.cut(row[1].strip()))
                
                if not ko_sentence or not zh_sentence:
                    raise ValueError("存在空句子")
                    
                korean_sentences.append(ko_sentence)
                chinese_sentences.append(zh_sentence)
                length_distribution['ko'].append(len(ko_sentence))
                length_distribution['zh'].append(len(zh_sentence))
                
            except Exception as e:
                error_msg = f"文件 {os.path.basename(file_path)} 第 {row_num} 行: {str(e)}"
                log_text.insert(tk.END, error_msg + '\n')
                error_log.append(error_msg)
                error_count += 1
                
    generate_statistics_report(error_log, total_rows, error_count, length_distribution)
    return korean_sentences, chinese_sentences
# 新增统计功能函数
def generate_statistics_report(error_log, total, errors, lengths):
    # 生成错误日志文件
    log_path = os.path.join('C:/Users/zhengchunji/PycharmProjects/PythonProject/Translate Model', f'error_log_{datetime.now().strftime("%Y%m%d%H%M%S")}.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(error_log))
    
    # 计算统计指标
    valid_ratio = (total - errors) / total * 100
    stats = [
        f"=== 数据质量报告 ===",
        f"总数据行数: {total}",
        f"有效数据行: {total - errors} ({valid_ratio:.1f}%)",
        f"无效数据行: {errors}",
        f"韩语句子平均长度: {np.mean(lengths['ko']):.1f} 词",
        f"中文句子平均长度: {np.mean(lengths['zh']):.1f} 词"
    ]
    
    # 更新GUI显示
    log_text.insert(tk.END, '\n'.join(stats) + '\n')
    
    # 生成长度分布图
    plot_length_distribution(lengths)

def plot_length_distribution(lengths):
    plt.figure(figsize=(10, 6))
    plt.hist([lengths['ko'], lengths['zh']], 
             bins=20, 
             label=['Korean', 'Chinese'],
             alpha=0.7)
    plt.title('Sentence Length Distribution')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.legend()
    
    plot_path = os.path.join('C:/Users/zhengchunji/PycharmProjects/PythonProject/Translate Model', f'length_distribution_{datetime.now().strftime("%Y%m%d%H%M%S")}.png')
    plt.savefig(plot_path)
    plt.close()
    
# 构建词汇表
def build_vocab(sentences):
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    index = 3
    for sentence in sentences:
        # 假设 sentence 是单词列表，确保每个元素是单个单词
        for word in sentence:
            if isinstance(word, str) and word.strip():  # 确保 word 是字符串且不为空
                if word not in vocab:
                    vocab[word] = index
                    index += 1
    return vocab

# 文本转张量
def text_to_tensor(sentences, vocab):
    tensors = []
    for sentence in sentences:
        tensor = [vocab['<sos>']] + [vocab.get(word, vocab['<pad>']) for word in sentence] + [vocab['<eos>']]
        tensors.append(torch.tensor(tensor, dtype=torch.long))
    return tensors
# 定义编码器
# 修改Encoder类定义，移除嵌套的Seq2Seq类
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # 统一维度处理
        if src.dim() == 3:  # (batch_size, seq_len, features)
            src = src.permute(1, 0, 2)  # (seq_len, batch_size, features)
        elif src.dim() == 2:  # (seq_len, batch_size)
            pass
        else:
            src = src.unsqueeze(1)  # (seq_len, 1)
        # 添加调试信息
        # print(f"编码器输入形状: {src.shape}")
        embedded = self.dropout(self.embedding(src))
        # 统一维度处理
        if embedded.dim() == 4:
            embedded = embedded.squeeze(0)  # 移除多余的维度
        embedded = embedded.permute(1, 0, 2)  # (seq_len, batch_size, emb_dim)
            
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell
# 添加Decoder类定义
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell
# 将Seq2Seq类移到外部定义
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # 确保输入维度一致
        if src.dim() == 2:  # 单个样本 (seq_len,)
            src = src.unsqueeze(1)  # 添加batch维度 -> (seq_len, 1)
        
        # 统一处理trg维度
        if trg.dim() == 1:  # 单个样本 (seq_len)
            trg = trg.unsqueeze(1)  # 添加batch维度 -> (seq_len, 1)
        batch_size = src.size(1)
        trg_len = trg.size(0)
        # 修改输出初始化逻辑
        outputs = torch.zeros(trg_len, batch_size, self.decoder.output_dim).to(self.device)
        # 修正隐藏状态初始化逻辑
        hidden, cell = self.encoder(src)
        
        # 修正后的调试信息位置
        print(f"[DEBUG] Encoder hidden shape: {hidden.shape}")
        print(f"[DEBUG] Target batch_size: {batch_size}")
        
        # 确保层数匹配 (num_layers, batch_size, hidden_dim)
        if hidden.size(0) != self.decoder.n_layers:
            hidden = hidden[:self.decoder.n_layers]
            cell = cell[:self.decoder.n_layers]
            
        # 修改批次处理逻辑
        if hidden.size(1) != batch_size:
            # 使用安全的维度调整方式
            hidden = hidden.repeat(1, batch_size, 1).contiguous()
            cell = cell.repeat(1, batch_size, 1).contiguous()
            
        # 修改输入处理逻辑
        input = trg[0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
            
        return outputs.permute(1, 0, 2)  # 调整为(batch_size, seq_len, vocab_size)

# 在train_model函数开头添加
from torch.utils.data import DataLoader, TensorDataset

def train_model(train_files, test_files):
    print("开始加载训练集和测试集...")
    
    # 添加初始化代码
    train_losses = []
    test_losses = []
    
    # 合并训练集文件
    all_ko_train = []
    all_zh_train = []
    for train_file in train_files:
        ko_train, zh_train = load_dataset([train_file])
        all_ko_train.extend(ko_train)
        all_zh_train.extend(zh_train)

    # 合并测试集文件
    all_ko_test = []
    all_zh_test = []
    for test_file in test_files:
        ko_test, zh_test = load_dataset([test_file])
        all_ko_test.extend(ko_test)
        all_zh_test.extend(zh_test)

    print("数据集加载完成。")

    # 构建词汇表
    print("开始构建词汇表...")
    korean_vocab = build_vocab(all_ko_train)
    chinese_vocab = build_vocab(all_zh_train)
    print("词汇表构建完成。")

    # 构建反向词汇表
    vocab_reverse_chinese = {v: k for k, v in chinese_vocab.items()}

    print("开始将文本转换为张量...")
    # 文本转张量
    ko_train_tensor = text_to_tensor(all_ko_train, korean_vocab)
    zh_train_tensor = text_to_tensor(all_zh_train, chinese_vocab)
    ko_test_tensor = text_to_tensor(all_ko_test, korean_vocab)
    zh_test_tensor = text_to_tensor(all_zh_test, chinese_vocab)
    print("文本转换为张量完成。")
    # 在训练循环前添加
    print("开始初始化模型和设置超参数...")

    # 定义超参数
    INPUT_DIM = len(korean_vocab)
    OUTPUT_DIM = len(chinese_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    N_EPOCHS = 10
    CLIP = 1
    PAD_IDX = korean_vocab['<pad>']

    # 创建TensorDataset（移到此处）
    ko_train_padded = torch.nn.utils.rnn.pad_sequence(ko_train_tensor, batch_first=True, padding_value=PAD_IDX)
    zh_train_padded = torch.nn.utils.rnn.pad_sequence(zh_train_tensor, batch_first=True, padding_value=PAD_IDX)
    assert ko_train_padded.dim() == 2, f"韩语训练数据维度错误：{ko_train_padded.dim()}，应为2D"
    assert zh_train_padded.dim() == 2, f"中文训练数据维度错误：{zh_train_padded.dim()}，应为2D"
    
    train_data = TensorDataset(ko_train_padded, zh_train_padded)
    # 在创建TensorDataset后添加
    print(f"训练数据维度检查 - 源: {ko_train_tensor[0].shape}, 目标: {zh_train_tensor[0].shape}")
    
    # 修改DataLoader为批次模式
    train_loader = DataLoader(train_data, 
                            batch_size=32,  # 添加合理的批次大小
                            shuffle=True,   # 启用数据打乱
                            num_workers=4)

    # 初始化模型（保持原结构）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)  # 补充损失函数定义

    for epoch in range(N_EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # 获取全部训练数据
        # 在训练循环中添加维度检查
        src_all, trg_all = next(iter(train_loader))
        assert src_all.dim() == 2, f"源数据维度错误，期望2维，实际得到{src_all.dim()}维，形状：{src_all.shape}"
        assert trg_all.dim() == 2, f"目标数据维度错误，期望2维，实际得到{trg_all.dim()}维，形状：{trg_all.shape}"
        src_all = src_all.to(device)
        trg_all = trg_all.to(device)
    
        output = model(src_all, trg_all)
        output_dim = output.shape[-1]
        
        # 维度处理
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg_all[:, 1:].reshape(-1)
        
        # 删除批次匹配检查
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        
        # 记录损失
        train_loss = loss.item()
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}')
    
        # 创建测试数据集（添加在训练数据之后）
        test_data = TensorDataset(
            torch.nn.utils.rnn.pad_sequence(ko_test_tensor, batch_first=True, padding_value=PAD_IDX),
            torch.nn.utils.rnn.pad_sequence(zh_test_tensor, batch_first=True, padding_value=PAD_IDX)
         )

        # 修改测试集加载
        # 在train_model函数中修改测试集加载
        test_loader = DataLoader(test_data, 
                       batch_size=32,  # 保持与训练批次一致
                       shuffle=False,
                       num_workers=4)
    
        # 将测试代码移到epoch循环内部
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for src_batch, trg_batch in test_loader:
                src_batch = src_batch.to(device)
                trg_batch = trg_batch.to(device)
                output = model(src_batch, trg_batch)
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg_batch[:, 1:].reshape(-1)
                loss = criterion(output, trg)
                test_loss += loss.item()
            test_loss = test_loss / len(ko_test_tensor)
            test_losses.append(test_loss)
        print(f'Epoch: {epoch+1:02}, Test Loss: {test_loss:.3f}')  # 现在epoch在作用域内

    # 移除循环外的测试代码
    model.eval()
    references = []
    candidates = []
    with torch.no_grad():
        for src, trg in zip(ko_test_tensor, zh_test_tensor):
            src = src.unsqueeze(1).to(device)
            trg = trg.unsqueeze(1).to(device)
            output = model(src, trg, teacher_forcing_ratio=0)  # 禁用教师强制
            output = output.argmax(2)
            output = output.squeeze(1).tolist()
            trg = trg.squeeze(1).tolist()
            reference = [[vocab_reverse_chinese[idx] for idx in trg if idx not in [chinese_vocab['<pad>'], chinese_vocab['<sos>'], chinese_vocab['<eos>']]]]
            candidate = [vocab_reverse_chinese[idx] for idx in output if idx not in [chinese_vocab['<pad>'], chinese_vocab['<sos>'], chinese_vocab['<eos>']]]
            references.append(reference)
            candidates.append(candidate)

    bleu_score = corpus_bleu(references, candidates)
    print(f'BLEU Score: {bleu_score:.4f}')

    # 检查并创建文件夹
    model_folder = 'Translate Model'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_path = os.path.join(model_folder, f'model_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

    # 新增词汇表保存代码
    vocab_path = os.path.join(model_folder, f'korean_vocab_{timestamp}.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(korean_vocab, f)
    
    vocab_path = os.path.join(model_folder, f'chinese_vocab_{timestamp}.pkl') 
    with open(vocab_path, 'wb') as f:
        pickle.dump(chinese_vocab, f)

    # 打印训练集和测试集损失
    print('Train Losses:', train_losses)
    print('Test Losses:', test_losses)
    # 添加返回语句
    return korean_vocab, chinese_vocab

# 运行交互界面
# 在文件最末尾确保正确调用主函数
if __name__ == "__main__":
    create_gui()  # 确认这行存在且是最后执行的
