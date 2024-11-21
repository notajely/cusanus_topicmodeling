import os
import json
from bs4 import BeautifulSoup
from cltk.lemmatize.lat import LatinBackoffLemmatizer
from tqdm import tqdm
import csv

# 设置项目路径
BASE_DIR = '/Users/jessie/Documents/Projects/Cusanus_Topic_Modeling'
input_dir = os.path.join(BASE_DIR, 'data/raw')
csv_output_path = os.path.join(BASE_DIR, 'data/differences_output.csv')
lemma_mapping_path = os.path.join(BASE_DIR, 'data/external/lemma.xml')

# 初始化 CLTK 拉丁词形还原工具
lemmatizer = LatinBackoffLemmatizer()

# 解析 lemma.xml 文件，构建 id_lemma 到词形的映射表
lemma_mapping = {}
with open(lemma_mapping_path, 'r', encoding='utf-8') as lemma_file:
    lemma_soup = BeautifulSoup(lemma_file, 'lxml')
    for lemma_entry in lemma_soup.find_all('lemma'):
        lemma_id = lemma_entry.get('id_lemma')
        lemma_value = lemma_entry.get('name').strip().lower()
        if lemma_id and lemma_value:
            lemma_mapping[lemma_id] = lemma_value

# 用于存储 h 和 v 版本的文件对比信息
differences = []

# 遍历文件夹中的 h 和 v 版本文件
for file_name in tqdm(os.listdir(input_dir), desc="处理文件"):
    if file_name.startswith('h') and file_name.endswith('.xml'):
        h_file_path = os.path.join(input_dir, file_name)
        v_file_name = 'v' + file_name[1:]
        v_file_path = os.path.join(input_dir, v_file_name)

        # 确保 v 版本文件存在
        if os.path.exists(v_file_path):
            # 加载 h 版本文件
            with open(h_file_path, 'r', encoding='utf-8') as h_file:
                h_soup = BeautifulSoup(h_file, 'lxml')

            # 加载 v 版本文件
            with open(v_file_path, 'r', encoding='utf-8') as v_file:
                v_soup = BeautifulSoup(v_file, 'lxml')

            # 提取 h 版本中的 lemma 信息
            h_lemmas = []
            for w in h_soup.find_all('w'):
                word_id = w.get('id')
                lemma_l = w.get('lemma_l', '').lower()
                original_word = w.get_text().lower()

                # 使用映射表来获取实际的 lemma
                lemma = lemma_mapping.get(lemma_l, lemma_l)  # 如果映射存在，则获取词形；否则保留原始编号
                if word_id and lemma:
                    h_lemmas.append({'id': word_id, 'original': original_word, 'lemma': lemma})

            # 提取 v 版本中的词汇并进行 CLTK 词形还原
            v_lemmas = []
            for w in v_soup.find_all('w'):
                original_word = w.get_text().lower()
                cltk_lemma = lemmatizer.lemmatize([original_word])[0][1]
                if not cltk_lemma:
                    cltk_lemma = original_word  # 如果没有找到词形，还原为原词
                v_lemmas.append({'original': original_word, 'lemma': cltk_lemma})

            # 对比 h 版本和 v 版本的词形还原结果
            for h_lemma, v_lemma in zip(h_lemmas, v_lemmas):
                if h_lemma['original'] == v_lemma['original'] and h_lemma['lemma'] != v_lemma['lemma']:
                    differences.append({
                        'word_id': h_lemma['id'],
                        'original_word': h_lemma['original'],
                        'h_lemma': h_lemma['lemma'],
                        'cltk_lemma': v_lemma['lemma']
                    })

# 保存差异到 CSV 文件
with open(csv_output_path, 'w', encoding='utf-8', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Word ID', 'Original Word', 'h Lemma', 'CLTK Lemma'])
    for diff in differences:
        writer.writerow([diff['word_id'], diff['original_word'], diff['h_lemma'], diff['cltk_lemma']])

# 输出差异
print("CLTK 词形还原与 h 版本 lemma_l 的差异已保存到 CSV 文件中。")
print(f"总差异数: {len(differences)}")
