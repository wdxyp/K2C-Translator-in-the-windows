import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import re
from konlpy.tag import Okt
import jieba

# --- 1. 文本清洗 (必须与训练代码一致) ---
def clean_text(sentence):
    if not isinstance(sentence, str): return ""
    sentence = re.sub(r'[^\w\s\uAC00-\uD7A3\u4e00-\u9fa5]', '', sentence)
    return sentence.strip()

# --- 2. 模型定义 (必须与 V3.0 训练代码完全一致) ---
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((hid_dim * 2) + emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0)

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
        encoder_outputs, hidden = self.encoder(src, src_len)
        input = trg[0,:]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            input = top1
        return outputs

# --- 3. 翻译函数 ---
def translate_sentence(sentence, model, ko_vocab, zh_vocab, device, max_len=50):
    model.eval()
    sentence = clean_text(sentence)
    
    # 分词
    try:
        okt = Okt()
        tokens = okt.morphs(sentence)
    except:
        tokens = sentence.split()
    print(f"分词结果: {tokens}")
    
    # 转索引
    unk_idx = ko_vocab.get('<unk>', 3)
    indices = [ko_vocab['<sos>']] + [ko_vocab.get(token, unk_idx) for token in tokens] + [ko_vocab['<eos>']]
    src_tensor = torch.LongTensor(indices).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(indices)])
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)
    
    trg_indices = [zh_vocab['<sos>']]
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
        
        top1 = output.argmax(1).item()
        trg_indices.append(top1)
        if top1 == zh_vocab['<eos>']:
            break
    
    inv_zh_vocab = {v: k for k, v in zh_vocab.items()}
    translated_tokens = [inv_zh_vocab.get(idx, '<unk>') for idx in trg_indices]
    return "".join([t for t in translated_tokens if t not in ['<sos>', '<eos>', '<pad>']])

# --- 4. 主程序 ---
if __name__ == "__main__":
    device = torch.device('cpu')
    model_dir = 'Translate Model'
    
    # 注意：请在这里填入 V3.0 训练后生成的最新文件名
    # 示例：'model_v3_20260428xxxxxx.pth'
    model_path = os.path.join(model_dir, 'best_model_v3_attn.pth') 
    ko_vocab_path = os.path.join(model_dir, 'ko_vocab_v3_20260428061400.pkl') # 这里需要根据实际生成的文件名修改
    zh_vocab_path = os.path.join(model_dir, 'zh_vocab_v3_20260428061400.pkl') # 这里需要根据实际生成的文件名修改

    # 尝试查找最新的词汇表文件
    import glob
    ko_vocabs = glob.glob(os.path.join(model_dir, 'ko_vocab_v3_*.pkl'))
    zh_vocabs = glob.glob(os.path.join(model_dir, 'zh_vocab_v3_*.pkl'))
    if ko_vocabs: ko_vocab_path = max(ko_vocabs, key=os.path.getctime)
    if zh_vocabs: zh_vocab_path = max(zh_vocabs, key=os.path.getctime)

    if not os.path.exists(model_path):
        print(f"找不到模型文件: {model_path}，请先运行 V3.0 训练脚本。")
    else:
        with open(ko_vocab_path, 'rb') as f: ko_vocab = pickle.load(f)
        with open(zh_vocab_path, 'rb') as f: zh_vocab = pickle.load(f)
        
        INPUT_DIM = len(ko_vocab)
        OUTPUT_DIM = len(zh_vocab)
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        HID_DIM = 512
        N_LAYERS = 1
        
        attn = Attention(HID_DIM)
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, 0)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, 0, attn)
        model = Seq2Seq(enc, dec, device).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("V3.0 Attention 模型加载成功！")
        
        while True:
            sentence = input("\n请输入韩文 (输入 q 退出): ")
            if sentence.lower() == 'q': break
            if not sentence.strip(): continue
            
            translation = translate_sentence(sentence, model, ko_vocab, zh_vocab, device)
            print(f"中文翻译: {translation}")
