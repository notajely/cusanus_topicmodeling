import os
from bs4 import BeautifulSoup
import stanza
from tqdm import tqdm
import csv
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
import re

class CusanusProcessor:
    def __init__(self, input_dir, preprocessed_dir, result_dir, lemma_file_path):
        """初始化处理器"""
        self.input_dir = input_dir
        self.preprocessed_dir = preprocessed_dir
        self.result_dir = result_dir
        
        # 创建必要的目录
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 初始化Stanza（只初始化一次）
        print("初始化Stanza...")
        stanza.download('la', processors='tokenize,pos,lemma')
        self.nlp = stanza.Pipeline('la', processors='tokenize,pos,lemma', use_gpu=True)
        
        # 加载lemma映射和停用词（缓存到实例变量）
        self.lemma_mapping = self._load_lemma_mapping(lemma_file_path)
        self.stopwords = self._load_stopwords()
        
        # 缓存已处理的词形还原结果
        self.lemma_cache = {}

    def _load_lemma_mapping(self, lemma_file_path):
        """从XML文件加载lemma映射，包含详细的清理步骤"""
        lemma_mapping = {}
        try:
            with open(lemma_file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'lxml')  # 使用lxml解析器
                for lemma_entry in soup.find_all('lemma'):
                    lemma_id = lemma_entry.get('id_lemma')
                    lemma_name = lemma_entry.get('name')

                    # 详细的lemma名称处理逻辑
                    if lemma_name:
                        # 1. 移除开头的*和其他注释
                        lemma_name = re.sub(r'^\*.*?\s', '', lemma_name).strip()

                        # 2. 处理括号，优先使用括号前的内容
                        if '(' in lemma_name:
                            lemma_value = lemma_name.split('(')[0].strip().lower()
                        else:
                            lemma_value = lemma_name.strip().lower()

                        # 3. 处理多词lemma，选择第一个有效词
                        if lemma_value:
                            lemma_value_parts = lemma_value.split()
                            if lemma_value_parts:
                                lemma_value = lemma_value_parts[0]

                        # 4. 移除语法类别后缀
                        lemma_value = re.split(
                            r'\b(?:cj\.|adv\.|praep\.|f\.|m\.|n\.|pl\.|sg\.|dat\.|acc\.|nom\.|gen\.|abl\.)\b', 
                            lemma_value
                        )[0].strip()

                        # 5. 移除不必要的描述性内容
                        lemma_value = re.sub(
                            r'\b(?:provincia|region|place|saec\.|asia minor|africa|italia|hispania)\b.*', 
                            '', 
                            lemma_value
                        ).strip()

                        # 6. 确保最终的lemma值有效且有ID
                        if lemma_id and lemma_value:
                            lemma_mapping[lemma_id] = lemma_value

            print(f"已加载 {len(lemma_mapping)} 个lemma映射")
            return lemma_mapping
                
        except Exception as e:
            print(f"加载lemma映射文件时出错: {e}")
            print(f"文件路径: {lemma_file_path}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _load_stopwords(self):
        """加载拉丁语停用词"""
        stopwords = {
            'et', 'in', 'ad', 'ut', 'cum', 'de', 'non', 'est', 'ex', 'per',
            'qui', 'sum', 'is', 'hic', 'quia', 'se', 'ab', 'suus', 'ipse',
            'vel', 'sic', 'sed', 'etiam', 'si', 'tam', 'enim', 'autem',
            'nec', 'quis', 'iam', 'tunc', 'sicut', 'ante', 'post', 'sub',
            'pro', 'apud', 'inter', 'super', 'ergo', 'ita', 'sive', 'nam',
            'nisi', 'cur', 'quid', 'unde', 'ubi', 'quam', 'quod', 'quoque',
            'atque', 'ac', 'ne', 'aut', 'seu', 'quo', 'qua', 'quare',
            'sine', 'modo', 'contra', 'dum', 'quasi', 'nunc', 'igitur',
            'tamen', 'adhuc', 'item', 'quidem', 'quando', 'quippe'
        }
        return stopwords

    def _get_lemma(self, word, lemma_id=None):
        """获取词的lemma形式，使用缓存提高效率"""
        if lemma_id and lemma_id in self.lemma_mapping:
            return self.lemma_mapping[lemma_id]
        
        word = word.lower()
        if word in self.lemma_cache:
            return self.lemma_cache[word]
        
        try:
            doc = self.nlp(word)
            lemma = doc.sentences[0].words[0].lemma
            self.lemma_cache[word] = lemma
            return lemma
        except Exception as e:
            print(f"处理词 {word} 时出错: {e}")
            return word

    def process_file(self, file_path, preprocessed_path):
        """处理单个文件：按fw标签分段，提取和处理内容"""
        paragraphs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'xml')
                
                for fw_tag in soup.find_all('fw', {'type': 'n'}):
                    section_words = []
                    words_to_process = []  # 存储需要处理的词和它们的位置
                    positions = []  # 存储对应位置
                    next_sibling = fw_tag.find_next_sibling()
                    
                    while next_sibling and next_sibling.name != 'fw':
                        if next_sibling.name == 'p':
                            for w in next_sibling.find_all('w'):
                                word = w.get('rend', w.get_text()).lower()
                                lemma_id = w.get('lemma')
                                
                                if lemma_id and lemma_id in self.lemma_mapping:
                                    section_words.append(self.lemma_mapping[lemma_id])
                                else:
                                    if word in self.lemma_cache:
                                        section_words.append(self.lemma_cache[word])
                                    else:
                                        section_words.append(word)  # 先放入原词
                                        words_to_process.append(word)
                                        positions.append(len(section_words) - 1)
                        
                        next_sibling = next_sibling.find_next_sibling()
                    
                    # 批量处理未知词形
                    if words_to_process:
                        try:
                            doc = self.nlp(' '.join(words_to_process))
                            for i, word in enumerate(doc.sentences[0].words):
                                if i < len(positions):  # 确保索引在范围内
                                    lemma = word.lemma
                                    self.lemma_cache[words_to_process[i]] = lemma  # 更新缓存
                                    section_words[positions[i]] = lemma  # 更新对应位置的词
                        except Exception as e:
                            print(f"批量处理词形还原时出错: {e}")
                    
                    # 过滤停用词和短词
                    filtered_words = [
                        word for word in section_words 
                        if (word not in self.stopwords 
                            and len(word) > 2 
                            and not any(c.isdigit() for c in word))
                    ]
                    
                    if filtered_words:
                        paragraphs.append(' '.join(filtered_words))
                
                # 保存预处理结果
                with open(preprocessed_path, 'w', encoding='utf-8') as f:
                    for idx, paragraph in enumerate(paragraphs, start=1):
                        f.write(f"Paragraph {idx}:\n{paragraph}\n\n")
                
                return paragraphs
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return []

    def _calculate_document_statistics(self, paragraphs):
        """计算单个文档的统计信息"""
        stats = {
            "word_frequencies": Counter(),
            "pos_distribution": Counter()
        }
        
        # 合并所有段落并分词
        words = ' '.join(paragraphs).split()
        
        # 计算词频
        stats["word_frequencies"].update(words)
        
        # 批量进行词性标注
        try:
            doc = self.nlp(' '.join(words))
            for sent in doc.sentences:
                for word in sent.words:
                    stats["pos_distribution"][word.upos] += 1
        except Exception as e:
            print(f"计算词性分布时出错: {e}")
        
        return stats
    
    def _save_statistics(self, stats):
        """保存统计信息到CSV文件"""
        # 保存词频统计
        word_freq_path = os.path.join(self.result_dir, 'word_frequencies.csv')
        with open(word_freq_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Word', 'Frequency'])
            for word, freq in sorted(stats["word_frequencies"].items(), 
                                   key=lambda x: (-x[1], x[0])):  # 按频率降序，词字母升序
                writer.writerow([word, freq])
        
        # 保存词性分布统计
        pos_dist_path = os.path.join(self.result_dir, 'pos_distribution.csv')
        with open(pos_dist_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['POS Tag', 'Frequency'])
            for pos, count in sorted(stats["pos_distribution"].items(), 
                                   key=lambda x: (-x[1], x[0])):
                writer.writerow([pos, count])
        
        print(f"统计信息已保存到 {self.result_dir}")
    
    def process_all_files(self):
        """处理输入目录中的所有文件"""
        xml_files = [f for f in os.listdir(self.input_dir) if f.endswith('.xml')]
        
        all_stats = {
            "word_frequencies": Counter(),
            "pos_distribution": Counter()
        }
        
        print(f"开始处理 {len(xml_files)} 个文件...")
        
        for file_name in tqdm(xml_files):
            input_path = os.path.join(self.input_dir, file_name)
            preprocessed_path = os.path.join(self.preprocessed_dir, f"processed_{file_name}")
            
            paragraphs = self.process_file(input_path, preprocessed_path)
            if paragraphs:
                doc_stats = self._calculate_document_statistics(paragraphs)
                
                # 更新总体统计信息
                all_stats["word_frequencies"].update(doc_stats["word_frequencies"])
                all_stats["pos_distribution"].update(doc_stats["pos_distribution"])
        
        self._save_statistics(all_stats)
        print("所有文件处理完成！")

def main():
    input_dir = "data/preprocessed_testset"
    preprocessed_dir = "data/preprocessed_testset/processed"
    result_dir = "results/preprocessed_test_result"
    lemma_file_path = "data/external/lemma.xml"
    
    if not all([os.path.exists(p) for p in [input_dir, lemma_file_path]]):
        print("错误：输入目录或lemma文件不存在！")
        return
    
    processor = CusanusProcessor(
        input_dir=input_dir,
        preprocessed_dir=preprocessed_dir,
        result_dir=result_dir,
        lemma_file_path=lemma_file_path
    )
    
    processor.process_all_files()

if __name__ == "__main__":
    main()