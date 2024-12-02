import spacy
import pandas as pd
from pathlib import Path
from collections import Counter
import os

def process_train_set(data_dir: Path):
    # 加载拉丁语模型
    nlp = spacy.load("la_core_web_lg")  # 修改为 la_core_web_lg
    
    # 读取训练集文本文件
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    word_freq = Counter()
    
    # 处理每个文本文件
    for file in txt_files:
        file_path = data_dir / file
        with open(file_path, 'r', encoding='utf-8') as f:
            train_text = f.read()
        
        # 使用spaCy处理文本
        doc = nlp(train_text)
        
        # 统计词频和词性（不考虑停用词和标点符号）
        for token in doc:
            if not token.is_punct:  # 排除标点符号
                word_freq[(token.text, token.pos_)] += 1  # 统计词和词性
    
    # 将结果转换为DataFrame
    freq_df = pd.DataFrame(word_freq.items(), columns=['Word_POS', 'Frequency'])
    freq_df[['Word', 'POS']] = pd.DataFrame(freq_df['Word_POS'].tolist(), index=freq_df.index)
    freq_df = freq_df.drop(columns=['Word_POS'])
    
    # 保存为CSV文件
    output_dir = data_dir / 'spacy'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'word_frequency.csv'
    freq_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"词频和词性统计已保存到: {output_file}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]  # 获取当前脚本的父目录
    process_train_set(base_dir / 'experiments' / 'lda' / 'spacy' / 'train_set')  # 指定训练集目录