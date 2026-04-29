import pickle
import os
import glob
from collections import Counter

def analyze_vocab(vocab_path):
    if not os.path.exists(vocab_path):
        print(f"找不到文件: {vocab_path}")
        return

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    print(f"\n--- 词汇表分析: {os.path.basename(vocab_path)} ---")
    print(f"总词汇量: {len(vocab)}")
    
    # 打印前 20 个高频词（排除特殊字符）
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
    words = [word for word in vocab.keys() if word not in special_tokens]
    
    print(f"前 15 个普通词汇: {words[:15]}")
    
    if '<unk>' in vocab:
        print(f"<unk> 索引: {vocab['<unk>']}")
    else:
        print("警告: 词汇表中没有 <unk> 标记！")

def find_latest_vocabs(model_dir='Translate Model'):
    ko_files = glob.glob(os.path.join(model_dir, 'ko_vocab_v3_*.pkl'))
    zh_files = glob.glob(os.path.join(model_dir, 'zh_vocab_v3_*.pkl'))
    
    latest_ko = max(ko_files, key=os.path.getctime) if ko_files else None
    latest_zh = max(zh_files, key=os.path.getctime) if zh_files else None
    
    return latest_ko, latest_zh

if __name__ == "__main__":
    model_dir = 'Translate Model'
    latest_ko, latest_zh = find_latest_vocabs(model_dir)
    
    if latest_ko:
        analyze_vocab(latest_ko)
    else:
        print("未发现韩文词汇表文件。")
        
    if latest_zh:
        analyze_vocab(latest_zh)
    else:
        print("未发现中文词汇表文件。")
