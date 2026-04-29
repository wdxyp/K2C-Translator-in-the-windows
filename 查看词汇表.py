import pickle
import os

def check_vocab(vocab_path, name):
    if not os.path.exists(vocab_path):
        print(f"错误: 找不到文件 {vocab_path}")
        return
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    print(f"\n=== {name} 统计 ===")
    print(f"总词汇量: {len(vocab)}")
    
    # 检查一些基础词汇
    test_words = ['안녕하세요', '안녕', '사랑', '오늘', '나', '너'] if name == "韩语" else ['你好', '爱', '今天', '我', '你']
    print("\n基础词汇检查:")
    for word in test_words:
        status = "存在" if word in vocab else "不存在 (将被视为 <unk>)"
        idx = vocab.get(word, "N/A")
        print(f"  - {word}: {status} (索引: {idx})")
    
    # 打印前 20 个高频词（除了特殊标记）
    print("\n前 20 个高频词参考 (部分):")
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    count = 0
    for word, idx in sorted_vocab:
        if word not in ['<pad>', '<sos>', '<eos>', '<unk>']:
            print(f"  {word}", end=' | ')
            count += 1
        if count >= 20: break
    print("\n")

if __name__ == "__main__":
    # 请根据您文件夹下最新的文件名修改此处
    model_dir = 'Translate Model'
    ko_vocab_path = os.path.join(model_dir, 'korean_vocab_20260428055353.pkl')
    zh_vocab_path = os.path.join(model_dir, 'chinese_vocab_20260428055353.pkl')
    
    check_vocab(ko_vocab_path, "韩语")
    check_vocab(zh_vocab_path, "中文")
