import os
from bs4 import BeautifulSoup
import stanza
import re
from tqdm import tqdm
import csv
import json
import requests
from collections import Counter

class StanzaProcessor:
    def __init__(self, input_dir, preprocessed_dir, result_dir):
        """
        初始化处理器，使用Stanza进行词形还原和词性标注
        
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
        
        # 初始化Stanza
        stanza.download('la')
        self.nlp = stanza.Pipeline('la', processors='tokenize,pos,lemma')
        
        # 加载停用词
        self.latin_stopwords = self._load_stopwords()

    def _load_stopwords(self):
        """加载拉丁语停用词"""
        stopwords_url = 'https://raw.githubusercontent.com/aurelberra/stopwords/master/stopwords_latin.txt'
        response = requests.get(stopwords_url)
        response.encoding = 'utf-8'
        stopwords = set(line.strip() for line in response.text.splitlines() if line.strip())
        
        additional_stopwords = {
            'ego', 'mei', 'mihi', 'me', 'tu', 'tui', 'tibi', 'te',
            'nos', 'noster', 'nobis', 'vos', 'vester',
            'sui', 'sibi', 'se', 'sum', 'qui'
        }
        stopwords.update(additional_stopwords)
        return stopwords

    def preprocess_text(self, words):
        """过滤停用词"""
        return [word for word in words if word.lower() not in self.latin_stopwords]

    def process_file(self, file_path, preprocessed_path):
        """处理单个文件：提取段落、清理内容、去停用词和词形还原"""
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
                            original_word = w.get('rend', w.get_text()).lower()
                            
                            # 跳过德语单词和特殊符号
                            if re.search(r'[äöüß]', original_word) or re.match(r'^cum\\W*\\d*$', original_word):
                                continue
                            
                            words.append(original_word)

                        # 使用Stanza进行词形还原
                        doc = self.nlp(' '.join(words))
                        lemmatized_words = [word.lemma for sent in doc.sentences for word in sent.words]
                        
                        # 去停用词
                        filtered_words = self.preprocess_text(lemmatized_words)
                        section_content.append(' '.join(filtered_words))

                    next_sibling = next_sibling.find_next_sibling()

                paragraphs.append({'content': ' '.join(section_content)})

            # 保存预处理结果
            with open(preprocessed_path, 'w', encoding='utf-8') as f:
                for idx, paragraph in enumerate(paragraphs, start=1):
                    f.write(f"Paragraph {idx}:\n")
                    f.write(f"{paragraph['content']}\n\n")

            return paragraphs

    def calculate_statistics(self, paragraphs, document_id):
        """计算文档级统计信息"""
        doc_stats = {
            "document_id": document_id,
            "total_paragraphs": len(paragraphs),
            "total_words": 0,
            "unique_words": set(),
            "pos_distribution": {},
            "lemmatized_content": []
        }
        
        for paragraph in paragraphs:
            words = paragraph['content'].split()
            doc_stats["total_words"] += len(words)
            doc_stats["unique_words"].update(words)
            
            # POS标注
            doc = self.nlp(paragraph['content'])
            for sent in doc.sentences:
                for word in sent.words:
                    doc_stats["pos_distribution"][word.upos] = doc_stats["pos_distribution"].get(word.upos, 0) + 1
            
            doc_stats["lemmatized_content"].append(paragraph['content'])
        
        return doc_stats

    def process_all_files(self):
        """处理所有文件的主流程"""
        overall_stats = {
            "total_words": 0,
            "unique_words": set(),
            "pos_distribution": {},
            "word_frequencies": {}
        }
        document_stats = []

        xml_files = [f for f in os.listdir(self.input_dir) if f.endswith('.xml')]
        
        for file_name in tqdm(xml_files, desc="处理文件"):
            input_path = os.path.join(self.input_dir, file_name)
            preprocessed_path = os.path.join(
                self.preprocessed_dir, 
                file_name.replace('.xml', '_lemmatized.txt')
            )

            try:
                paragraphs = self.process_file(input_path, preprocessed_path)
                if paragraphs:
                    doc_stats = self.calculate_statistics(paragraphs, file_name)
                    document_stats.append(doc_stats)

                    # 更新总体统计
                    overall_stats["total_words"] += doc_stats["total_words"]
                    overall_stats["unique_words"].update(doc_stats["unique_words"])
                    
                    # 更新词频和POS分布
                    for word in doc_stats["unique_words"]:
                        overall_stats["word_frequencies"][word] = overall_stats["word_frequencies"].get(word, 0) + 1
                    
                    for pos, count in doc_stats["pos_distribution"].items():
                        overall_stats["pos_distribution"][pos] = overall_stats["pos_distribution"].get(pos, 0) + count

            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")

        # 保存统计结果
        self.save_statistics(overall_stats, document_stats)
        print("所有文件处理完成！")

    def save_statistics(self, overall_stats, document_stats):
        """保存统计结果到CSV文件"""
        # 保存总体统计到CSV
        csv_path = os.path.join(self.result_dir, "stanza_statistics.csv")
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # 基本统计信息
            writer.writerow(["total_words", "types"])
            writer.writerow([overall_stats["total_words"], len(overall_stats["unique_words"])])
            writer.writerow([])
            
            # 写入词频和POS分布（降序排列）
            writer.writerow(["Word", "Frequency", "POS Tag", "POS Frequency"])
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
            
            # 合并词频和POS统计
            max_length = max(len(word_items), len(pos_items))
            for i in range(max_length):
                word_data = word_items[i] if i < len(word_items) else ("", "")
                pos_data = pos_items[i] if i < len(pos_items) else ("", "")
                writer.writerow([
                    word_data[0],  # Word
                    word_data[1],  # Frequency
                    pos_data[0],   # POS Tag
                    pos_data[1]    # POS Frequency
                ])
        
        print(f"统计结果已保存到: {csv_path}")

def main():
    print("请输入以下路径信息（可以是相对路径或绝对路径）：")
    print("\n示例格式：")
    print("源文件目录：'data/raw' 或 '/Users/username/project/data/raw'")
    print("预处理文件目录：'data/preprocessed' 或 '/Users/username/project/data/preprocessed'")
    print("结果文件目录：'results' 或 '/Users/username/project/results'\n")

    input_dir = input("请输入源文件目录路径（包含XML文件的目录）: ").strip()
    preprocessed_dir = input("请输入预处理文件保存目录路径（用于保存中间处理结果）: ").strip()
    result_dir = input("请输入结果文件保存目录路径（用于保存最终统计结果）: ").strip()
    
    if not os.path.exists(input_dir):
        print(f"错误：输入目录 {input_dir} 不存在！")
        return
        
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    print("\n路径确认：")
    print(f"源文件目录: {os.path.abspath(input_dir)}")
    print(f"预处理文件目录: {os.path.abspath(preprocessed_dir)}")
    print(f"结果文件目录: {os.path.abspath(result_dir)}")
    
    if input("\n确认以上路径正确？(y/n): ").lower() != 'y':
        print("已取消操作")
        return
        
    processor = StanzaProcessor(
        input_dir=input_dir,
        preprocessed_dir=preprocessed_dir,
        result_dir=result_dir
    )
    
    processor.process_all_files()

if __name__ == "__main__":
    main()