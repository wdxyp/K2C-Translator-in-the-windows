import os
import re
import pickle
import unicodedata
from datetime import datetime
from collections import Counter
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import openpyxl


def _normalize_text(sentence: str) -> str:
    if not isinstance(sentence, str):
        return ""

    s = sentence.replace("\r\n", "\n").replace("\r", "\n")

    circled_map = {
        "⓪": 0,
        "①": 1,
        "②": 2,
        "③": 3,
        "④": 4,
        "⑤": 5,
        "⑥": 6,
        "⑦": 7,
        "⑧": 8,
        "⑨": 9,
        "⑩": 10,
        "⑪": 11,
        "⑫": 12,
        "⑬": 13,
        "⑭": 14,
        "⑮": 15,
        "⑯": 16,
        "⑰": 17,
        "⑱": 18,
        "⑲": 19,
        "⑳": 20,
    }
    for ch, n in circled_map.items():
        s = s.replace(ch, f"({n})")

    s = s.replace("…", "...")
    s = re.sub(r"-{1,4}>", "->", s)
    for arrow in ("→", "⇒", "➡", "⟶", "⟹", "➔", "➜", "➝", "➞", "➟", "➠"):
        s = s.replace(arrow, "->")

    for q in ("“", "”", "„", "‟", "＂"):
        s = s.replace(q, '"')

    s = unicodedata.normalize("NFKC", s)
    return s


def _separate_punct_boundaries(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""

    s = text
    s = s.replace("->", " -> ")
    s = s.replace("...", " ... ")

    for ch in [
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
        ",",
        ":",
        ";",
        "?",
        "!",
        "/",
        "\\",
        "$",
        "#",
        "@",
        "~",
        "&",
        "*",
        "%",
        "+",
        "=",
        '"',
        "_",
        "-",
        "·",
    ]:
        s = s.replace(ch, f" {ch} ")

    s = re.sub(r"(?<!\d)\.(?!\d)", " . ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_text(sentence: str) -> str:
    if not isinstance(sentence, str):
        return ""
    s = _normalize_text(sentence)
    s = s.replace("\n", " ")

    kept = []
    for ch in s:
        if ch.isspace():
            kept.append(" ")
            continue
        if "0" <= ch <= "9" or "A" <= ch <= "Z" or "a" <= ch <= "z":
            kept.append(ch)
            continue
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            kept.append(ch)
            continue
        if 0x4E00 <= code <= 0x9FFF:
            kept.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat and cat[0] == "P":
            kept.append(ch)
            continue
        if cat in ("Sm", "Sc"):
            kept.append(ch)
            continue

    s = "".join(kept)
    s = _separate_punct_boundaries(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_tokenizers():
    try:
        from konlpy.tag import Okt

        okt = Okt()

        def tok_ko(s: str):
            return okt.morphs(s)

        ko_name = "Okt"
    except Exception as e:
        raise RuntimeError(
            "Kaggle 环境未能导入 konlpy/Okt。请先在 Notebook 单独执行安装：\n"
            "!pip -q install konlpy jpype1\n"
            "如果仍失败，请确认 Kaggle 已启用 Internet，并重启 Kernel。\n"
            f"原始错误: {e}"
        )

    try:
        import jieba

        def tok_zh(s: str):
            return jieba.lcut(s)

        zh_name = "jieba"
    except Exception as e:
        raise RuntimeError(
            "Kaggle 环境未能导入 jieba。请先在 Notebook 单独执行安装：\n"
            "!pip -q install jieba\n"
            "如果仍失败，请确认 Kaggle 已启用 Internet，并重启 Kernel。\n"
            f"原始错误: {e}"
        )

    return tok_ko, tok_zh, ko_name, zh_name


def read_corpus(file_paths):
    all_korean = []
    all_chinese = []
    for path in file_paths:
        try:
            wb = openpyxl.load_workbook(path, data_only=True)
            ws = wb.active
            for row in ws.iter_rows(min_row=2, values_only=True):
                if row is None or len(row) < 4:
                    continue
                ko, zh = row[1], row[3]
                if ko and zh:
                    all_korean.append(str(ko))
                    all_chinese.append(str(zh))
            wb.close()
        except Exception as e:
            print(f"读取文件出错: {path} / {e}")
    return all_korean, all_chinese


def build_vocab(tokenized_sentences, min_freq=2, max_size=30000):
    counter = Counter()
    for tokens in tokenized_sentences:
        counter.update(tokens)

    most_common = counter.most_common(max_size)
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for word, freq in most_common:
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


class TranslationDataset(Dataset):
    def __init__(self, ko_tokens, zh_tokens, ko_vocab, zh_vocab):
        self.ko_data = ko_tokens
        self.zh_data = zh_tokens
        self.ko_vocab = ko_vocab
        self.zh_vocab = zh_vocab

    def __len__(self):
        return len(self.ko_data)

    def __getitem__(self, idx):
        ko_idx = [self.ko_vocab["<sos>"]] + [self.ko_vocab.get(token, self.ko_vocab["<unk>"]) for token in self.ko_data[idx]] + [
            self.ko_vocab["<eos>"]
        ]
        zh_idx = [self.zh_vocab["<sos>"]] + [self.zh_vocab.get(token, self.zh_vocab["<unk>"]) for token in self.zh_data[idx]] + [
            self.zh_vocab["<eos>"]
        ]
        return torch.LongTensor(ko_idx), torch.LongTensor(zh_idx)


def collate_fn(batch):
    ko_batch, zh_batch = zip(*batch)
    ko_padded = nn.utils.rnn.pad_sequence(ko_batch, padding_value=0)
    zh_padded = nn.utils.rnn.pad_sequence(zh_batch, padding_value=0)
    return ko_padded, zh_padded


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
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

    def forward(self, src):
        src_len = (src != 0).sum(dim=0)
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
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

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size, device=self.device)

        encoder_outputs, hidden = self.encoder(src)
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = torch.rand(1, device=self.device).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs


def train_model(train_loader, test_loader, ko_vocab, zh_vocab, device, model_folder, max_epochs=50):
    input_dim = len(ko_vocab)
    output_dim = len(zh_vocab)
    enc_emb_dim = 256
    dec_emb_dim = 256
    hid_dim = 512
    n_layers = 1
    enc_dropout = 0.5
    dec_dropout = 0.5

    attn = Attention(hid_dim)
    enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, enc_dropout if n_layers > 1 else 0)
    dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers, dec_dropout if n_layers > 1 else 0, attn)
    model = Seq2Seq(enc, dec, device).to(device)

    if torch.cuda.device_count() >= 2:
        model = nn.DataParallel(model, dim=1)
        print("使用多GPU DataParallel, device_count=", torch.cuda.device_count())
    else:
        print("使用单GPU/CPU, device_count=", torch.cuda.device_count())

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_test_loss = float("inf")
    patience = 10
    no_improve = 0

    best_model_path = os.path.join(model_folder, "best_model_v3_0_1_attn.pth")
    best_ko_vocab_path = os.path.join(model_folder, "best_ko_vocab_v3_0_1_attn.pkl")
    best_zh_vocab_path = os.path.join(model_folder, "best_zh_vocab_v3_0_1_attn.pkl")

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for i, (src, trg) in enumerate(train_loader):
            src = src.to(device, non_blocking=True)
            trg = trg.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                output = model(src, trg)
                output_dim = output.shape[-1]
                output = output[1:].reshape(-1, output_dim)
                trg_flat = trg[1:].reshape(-1)
                loss = criterion(output, trg_flat)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            if (i + 1) % 100 == 0:
                print(f"Epoch: {epoch+1:02} Batch: {i+1}/{len(train_loader)} Loss: {loss.item():.4f}")

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for src, trg in test_loader:
                src = src.to(device, non_blocking=True)
                trg = trg.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    output = model(src, trg, 0)
                    output_dim = output.shape[-1]
                    output = output[1:].reshape(-1, output_dim)
                    trg_flat = trg[1:].reshape(-1)
                    loss = criterion(output, trg_flat)
                test_loss += float(loss.item())

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        avg_test_loss = test_loss / max(1, len(test_loader))
        print(f"Epoch: {epoch+1:02} Train Loss: {avg_train_loss:.4f} Test Loss: {avg_test_loss:.4f}")

        scheduler.step(avg_test_loss)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            no_improve = 0
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, best_model_path)
            with open(best_ko_vocab_path, "wb") as f:
                pickle.dump(ko_vocab, f)
            with open(best_zh_vocab_path, "wb") as f:
                pickle.dump(zh_vocab, f)
            print("保存最佳模型")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("早停触发，训练结束")
                break

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    final_model_path = os.path.join(model_folder, f"model_v3_0_1_{timestamp}.pth")
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, final_model_path)
    with open(os.path.join(model_folder, f"ko_vocab_v3_0_1_{timestamp}.pkl"), "wb") as f:
        pickle.dump(ko_vocab, f)
    with open(os.path.join(model_folder, f"zh_vocab_v3_0_1_{timestamp}.pkl"), "wb") as f:
        pickle.dump(zh_vocab, f)
    print("训练完成")


def _find_xlsx_files(search_root: str):
    patterns = [
        os.path.join(search_root, "*.xlsx"),
        os.path.join(search_root, "*", "*.xlsx"),
        os.path.join(search_root, "*", "*", "*.xlsx"),
        os.path.join(search_root, "*", "*", "*", "*.xlsx"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    dedup = []
    seen = set()
    for f in files:
        if f not in seen:
            seen.add(f)
            dedup.append(f)
    return dedup


def main():
    corpus_roots = ["/kaggle/input"]
    corpus_files = []
    for r in corpus_roots:
        if os.path.exists(r):
            corpus_files.extend(_find_xlsx_files(r))

    corpus_files = [f for f in corpus_files if os.path.isfile(f)]
    if not corpus_files:
        raise RuntimeError("未找到任何 .xlsx 语料文件，请把数据集添加到 /kaggle/input")

    print("检测到语料文件数:", len(corpus_files))
    for f in corpus_files[:20]:
        print("  ", f)
    if len(corpus_files) > 20:
        print("  ...")

    tok_ko, tok_zh, ko_name, zh_name = _build_tokenizers()
    print("tokenizer ko:", ko_name)
    print("tokenizer zh:", zh_name)

    ko_sents, zh_sents = read_corpus(corpus_files)
    print("pairs read:", len(ko_sents))

    ko_sents = [clean_text(s) for s in ko_sents]
    zh_sents = [clean_text(s) for s in zh_sents]
    nonempty = [(k, z) for k, z in zip(ko_sents, zh_sents) if k and z]
    ko_sents = [k for k, _ in nonempty]
    zh_sents = [z for _, z in nonempty]
    print("pairs after clean+drop empty:", len(ko_sents))

    ko_tokens = [tok_ko(s) for s in ko_sents]
    zh_tokens = [tok_zh(s) for s in zh_sents]

    ko_vocab = build_vocab(ko_tokens)
    zh_vocab = build_vocab(zh_tokens)
    print("ko_vocab size:", len(ko_vocab))
    print("zh_vocab size:", len(zh_vocab))

    ko_train, ko_test, zh_train, zh_test = train_test_split(ko_tokens, zh_tokens, test_size=0.1, random_state=42)
    train_ds = TranslationDataset(ko_train, zh_train, ko_vocab, zh_vocab)
    test_ds = TranslationDataset(ko_test, zh_test, ko_vocab, zh_vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "gpu_count:", torch.cuda.device_count())

    train_loader = DataLoader(
        train_ds,
        batch_size=64 if device.type == "cuda" else 32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=64 if device.type == "cuda" else 32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(device.type == "cuda"),
    )

    model_folder = "/kaggle/working/Translate Model"
    os.makedirs(model_folder, exist_ok=True)
    train_model(train_loader, test_loader, ko_vocab, zh_vocab, device, model_folder, max_epochs=50)


if __name__ == "__main__":
    main()
