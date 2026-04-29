import torch
import torch.nn as nn
import pickle
import os
import re
from konlpy.tag import Okt
import jieba

# --- 文本清洗 (必须与训练代码一致) ---
def clean_text(sentence):
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence.strip()

# --- 模型定义 (必须与训练代码完全一致) ---
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
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

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src, src_len)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = top1
        return outputs

def translate_sentence(sentence, model, ko_vocab, zh_vocab, device, max_len=50):
    model.eval()
    
    # 1. 文本清洗
    sentence = clean_text(sentence)
    
    # 2. 分词
    try:
        okt = Okt()
        tokens = okt.morphs(sentence)
    except Exception as e:
        print(f"韩文分词器启动失败: {e}，尝试使用空格分词...")
        tokens = sentence.split()
    
    print(f"分词结果: {tokens}")
    
    # 转索引
    unk_idx = ko_vocab.get('<unk>', 3)
    indices = [ko_vocab.get('<sos>', 1)] + [ko_vocab.get(token, unk_idx) for token in tokens] + [ko_vocab.get('<eos>', 2)]
    
    src_tensor = torch.LongTensor(indices).unsqueeze(1).to(device)
    src_len = [len(indices)]
    
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, src_len)
    
    trg_indices = [zh_vocab.get('<sos>', 1)]
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        
        top1 = output.argmax(1).item()
        trg_indices.append(top1)
        
        if top1 == zh_vocab.get('<eos>', 2):
            break
    
    # 转回文字
    inv_zh_vocab = {v: k for k, v in zh_vocab.items()}
    translated_tokens = [inv_zh_vocab.get(idx, '<unk>') for idx in trg_indices]
    
    return "".join([t for t in translated_tokens if t not in ['<sos>', '<eos>', '<pad>']])

# --- 主程序 ---
if __name__ == "__main__":
    device = torch.device('cpu')
    model_dir = 'Translate Model'
    # 模型和词汇表文件路径（指向刚刚生成的带时间戳的文件）
    model_path = os.path.join(model_dir, 'model_20260428055353.pth')
    ko_vocab_path = os.path.join(model_dir, 'korean_vocab_20260428055353.pkl')
    zh_vocab_path = os.path.join(model_dir, 'chinese_vocab_20260428055353.pkl')

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}。请确保训练至少完成了一个 Epoch。")
    else:
        # 加载词汇表
        with open(ko_vocab_path, 'rb') as f:
            korean_vocab = pickle.load(f)
        with open(zh_vocab_path, 'rb') as f:
            chinese_vocab = pickle.load(f)

        # 初始化模型
        INPUT_DIM = len(korean_vocab)
        OUTPUT_DIM = len(chinese_vocab)
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        HID_DIM = 256
        N_LAYERS = 2
        
        encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, 0)
        decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, 0)
        model = Seq2Seq(encoder, decoder, device).to(device)
        
        # 加载参数
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型加载成功！")

        while True:
            sentence = input("\n请输入韩文 (输入 q 退出): ")
            if sentence.lower() == 'q':
                break
            translation = translate_sentence(sentence, model, korean_vocab, chinese_vocab, device)
            print(f"中文翻译: {translation}")
