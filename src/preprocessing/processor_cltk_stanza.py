import os
from bs4 import BeautifulSoup
from cltk.lemmatize.lat import LatinBackoffLemmatizer
import stanza
import re
from tqdm import tqdm
import csv
from collections import Counter
import requests
import json

class CLTKStanzaProcessor:
    def __init__(self, input_dir, preprocessed_dir, result_dir):
        """
        初始化处理器，使用CLTK进行词形还原，Stanza进行词性标注
        
        Parameters:
            input_dir (str): 输入文件目录路径
            preprocessed_dir (str): 预处理文件保存目录路径
            result_dir (str): 结果文件保存目录路径
        """
        self.input_dir = input_dir
        self.preprocessed_dir = preprocessed_dir
        self.result_dir = result_dir
        
        # 创建必要的目录
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 初始化CLTK词形还原工具
        self.lemmatizer = LatinBackoffLemmatizer()
        # 初始化Stanza词性标注工具
        stanza.download('la')
        self.nlp = stanza.Pipeline('la')
        
        # 加载停用词
        self.latin_stopwords = self._load_stopwords()

    def _load_stopwords(self):
        """
        加载拉丁语停用词
        """
        stopwords_url = 'https://raw.githubusercontent.com/aurelberra/stopwords/master/stopwords_latin.txt'
        response = requests.get(stopwords_url)
        response.encoding = 'utf-8'
        stopwords = set(line.strip() for line in response.text.splitlines() if line.strip())
        
        # 添加额外的停用词
        additional_stopwords = {
            'ego', 'mei', 'mihi', 'me', 'tu', 'tui', 'tibi', 'te',
            'nos', 'noster', 'nobis', 'vos', 'vester',
            'sui', 'sibi', 'se',
            'ab', 'ex', 'ad', 'in', 'de', 'per', 'cum', 'sub', 'pro',
            'ante', 'post', 'supra', 'et', 'ac', 'aut', 'nec', 'sed',
            'ut', 'si', 'atque', 'qui', 'quae', 'quod', 'quis', 'quid', 'non', 'ne'
        }
        stopwords.update(additional_stopwords)
        return stopwords

    def preprocess_text(self, words):
        """
        过滤停用词
        """
        return [word for word in words if word.lower() not in self.latin_stopwords]

    def pos_tag_text(self, text):
        """
        使用Stanza进行词性标注
        """
        doc = self.nlp(text)
        return [(word.text, word.upos) for sentence in doc.sentences for word in sentence.words]

    def process_file(self, file_path, preprocessed_path):
        """
        处理单个文件：词形还原和词性标注
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'lxml')
            paragraphs = []

            for fw_tag in soup.find_all('fw', {'type': 'n'}):
                section_content = []
                next_sibling = fw_tag.find_next_sibling()

                while next_sibling and next_sibling.name != 'fw':
                    if next_sibling.name == 'p':
                        words = []
                        for w in next_sibling.find_all('w'):
                            if re.search(r'[äöüß]', w.get_text()) or re.match(r'^cum\W*\d*$', w.get_text()):
                                continue

                            original_word = w.get('rend', w.get_text()).lower()
                            # 使用CLTK进行词形还原
                            lemma = self.lemmatizer.lemmatize([original_word])[0][1]
                            words.append(lemma)

                        filtered_words = self.preprocess_text(words)
                        section_content.append(' '.join(filtered_words))

                    next_sibling = next_sibling.find_next_sibling()

                paragraphs.append({'content': ' '.join(section_content)})

            # 保存预处理结果
            with open(preprocessed_path, 'w', encoding='utf-8') as preprocessed_file:
                for idx, paragraph in enumerate(paragraphs, start=1):
                    preprocessed_file.write(f"Paragraph {idx}:\n")
                    preprocessed_file.write(f"{paragraph['content']}\n\n")

            return paragraphs

    def calculate_statistics(self, paragraphs, document_id):
        """
        计算文档统计信息
        """
        doc_stats = {
            "document_id": document_id,
            "total_paragraphs": len(paragraphs),
            "total_words": 0,
            "total_types": 0,
            "pos_distribution": {},
            "lemmatized_content": []
        }
        unique_words = set()
        
        for paragraph in paragraphs:
            words = paragraph['content'].split()
            doc_stats["total_words"] += len(words)
            unique_words.update(words)

            # 使用Stanza进行词性标注
            pos_tags = self.pos_tag_text(paragraph['content'])
            for _, pos in pos_tags:
                doc_stats["pos_distribution"][pos] = doc_stats["pos_distribution"].get(pos, 0) + 1

            doc_stats["lemmatized_content"].append(paragraph['content'])

        doc_stats["total_types"] = len(unique_words)
        doc_stats["unique_words"] = list(unique_words)
        
        return doc_stats

    def save_statistics(self, overall_stats, document_stats):
        """
        保存统计结果到单个CSV文件，包含词频、POS和类型数统计
        词频按降序排列
        """
        csv_path = os.path.join(self.result_dir, "cltk_stanza_statistics.csv")
        
        # 准备统计数据
        statistics = {
            "Word": [],
            "Frequency": [],
            "POS Tag": [],
            "POS Frequency": [],
            "total_words": overall_stats["total_words"],
            "types": len(overall_stats["unique_words"])
        }
        
        # 获取词频数据和POS标签频率数据，并按频率降序排序
        word_items = sorted(
            overall_stats["word_frequencies"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        pos_items = sorted(
            overall_stats["pos_distribution"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 找出最长的列表长度
        max_length = max(len(word_items), len(pos_items))
        
        # 写入CSV文件
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            
            # 写入总体统计信息
            writer.writerow(["total_words", "types"])
            writer.writerow([statistics["total_words"], statistics["types"]])
            writer.writerow([])  # 空行分隔
            
            # 写入词频和POS分布表头
            writer.writerow(["Word", "Frequency", "POS Tag", "POS Frequency"])
            
            # 写入词频和POS分布数据
            for i in range(max_length):
                word_data = word_items[i] if i < len(word_items) else ("", "")
                pos_data = pos_items[i] if i < len(pos_items) else ("", "")
                
                writer.writerow([
                    word_data[0],  # Word
                    word_data[1],  # Frequency
                    pos_data[0],   # POS Tag
                    pos_data[1]    # POS Frequency
                ])
        
        print(f"统计结果已保存到 {csv_path}")

    def process_all_files(self):
        """
        处理所有文件的主流程
        """
        overall_stats = {
            "total_words": 0,
            "unique_words": set(),
            "pos_distribution": {},
            "word_frequencies": {}
        }
        document_stats = []

        # 获取所有XML文件
        xml_files = [f for f in os.listdir(self.input_dir) if f.endswith('.xml')]
        
        # 使用tqdm显示进度
        for file_name in tqdm(xml_files, desc="处理文件"):
            input_path = os.path.join(self.input_dir, file_name)
            preprocessed_path = os.path.join(
                self.preprocessed_dir, 
                file_name.replace('.xml', '_lemmatized.txt')
            )

            try:
                # 处理文件
                paragraphs = self.process_file(input_path, preprocessed_path)
                if paragraphs:
                    # 计算统计信息
                    doc_stats = self.calculate_statistics(paragraphs, file_name)
                    document_stats.append(doc_stats)

                    # 更新总体统计
                    overall_stats["total_words"] += doc_stats["total_words"]
                    overall_stats["unique_words"].update(doc_stats["unique_words"])
                    
                    # 更新词频
                    for word in doc_stats["unique_words"]:
                        overall_stats["word_frequencies"][word] = overall_stats["word_frequencies"].get(word, 0) + 1
                    
                    # 更新POS分布
                    for pos, count in doc_stats["pos_distribution"].items():
                        overall_stats["pos_distribution"][pos] = overall_stats["pos_distribution"].get(pos, 0) + count

            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")

        # 保存统计结果
        self.save_statistics(overall_stats, document_stats)
        print("所有文件处理完成！")

def main():
    print("请输入以下路径信息（可以是相对路径或绝对路径）：")
    print("\n示例格式：")
    print("源文件目录：'data/raw' 或 '/Users/username/project/data/raw'")
    print("预处理文件目录：'data/preprocessed' 或 '/Users/username/project/data/preprocessed'")
    print("结果文件目录：'results' 或 '/Users/username/project/results'\n")

    # 获取用户输入
    input_dir = input("请输入源文件目录路径（包含XML文件的目录）: ").strip()
    preprocessed_dir = input("请输入预处理文件保存目录路径（用于保存中间处理结果）: ").strip()
    result_dir = input("请输入结果文件保存目录路径（用于保存最终统计结果）: ").strip()
    
    # 验证路径
    if not os.path.exists(input_dir):
        print(f"错误：输入目录 {input_dir} 不存在！")
        return
        
    # 创建输出目录（如果不存在）
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    print("\n路径确认：")
    print(f"源文件目录: {os.path.abspath(input_dir)}")
    print(f"预处理文件目录: {os.path.abspath(preprocessed_dir)}")
    print(f"结果文件目录: {os.path.abspath(result_dir)}")
    
    if input("\n确认以上路径正确？(y/n): ").lower() != 'y':
        print("已取消操作")
        return
        
    # 创建处理器实例
    processor = CLTKStanzaProcessor(
        input_dir=input_dir,
        preprocessed_dir=preprocessed_dir,
        result_dir=result_dir
    )
    
    # 执行处理
    processor.process_all_files()

if __name__ == "__main__":
    main()