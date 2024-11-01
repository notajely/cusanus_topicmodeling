import os
from bs4 import BeautifulSoup
import tqdm
from collections import defaultdict
import yaml

# 加载配置文件，获取项目根目录和相关路径
CONFIG_PATH = 'config.yaml'

try:
    with open(CONFIG_PATH, 'r') as config_file:
        config = yaml.safe_load(config_file)
    BASE_DIR = config.get('BASE_DIR', os.getcwd())
    print("配置文件加载成功，项目根目录为: ", BASE_DIR)
except FileNotFoundError:
    print(f"配置文件未找到: {CONFIG_PATH}，使用当前工作目录。")
    BASE_DIR = os.getcwd()

# 数据预处理部分
# 定义停用词列表路径并加载
stopwords_path = os.path.join(BASE_DIR, config.get('STOPWORDS_PATH', 'data/external/stopwords_latin.txt'))
try:
    with open(stopwords_path, 'r') as f:
        latin_stopwords = set(f.read().splitlines())
    print("停用词列表加载成功。")
except FileNotFoundError:
    print(f"停用词文件未找到: {stopwords_path}")
    latin_stopwords = set()

# 词形表加载函数
# 从 lemma.xml 文件中加载词形，还原词汇的标准形式
def load_lemmas(filepath):
    lemmas = defaultdict(str)
    try:
        with open(filepath, 'r') as file:
            soup = BeautifulSoup(file, 'xml')
            for lemma in soup.find_all('lemma'):
                lemmas[lemma['name']] = lemma['name']
                for variant in lemma.find_all('variant'):
                    lemmas[variant['name']] = lemma['name']
        print("词形表加载成功。")
    except FileNotFoundError:
        print(f"词形文件未找到: {filepath}")
    return lemmas

lemmas_path = os.path.join(BASE_DIR, config.get('LEMMAS_PATH', 'data/processed/lemma.xml'))
lemmas = load_lemmas(lemmas_path)

# XML 文件预处理函数
# 处理 XML 文件，检查是否包含 lemma_l 属性，并根据情况进行词形还原和停用词去除
def preprocess_text(xml_file):
    try:
        with open(xml_file, 'r') as file:
            soup = BeautifulSoup(file, 'xml')
            words = []
            for w in soup.find_all('w'):
                # 检查词汇是否有 lemma_l 属性
                lemma_attr = w.get('lemma_l')
                if lemma_attr:
                    # 如果有 lemma_l 属性，直接使用
                    word = lemma_attr.lower()
                else:
                    # 如果没有 lemma_l 属性，则使用词汇文本并进行词形还原
                    raw_word = w.get_text().lower()
                    word = lemmas.get(raw_word, raw_word)
                # 去除停用词
                if word not in latin_stopwords:
                    words.append(word)
            return ' '.join(words)
    except FileNotFoundError:
        print(f"XML 文件未找到: {xml_file}")
        return ""

# 处理目录中的文件，保存结果
# 读取原始文件并处理，生成经过词形还原和停用词去除的文本文件
def process_directory(input_dir, output_dir):
    input_dir = os.path.join(BASE_DIR, input_dir)
    output_dir = os.path.join(BASE_DIR, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]

    for xml_file in tqdm.tqdm(xml_files):
        input_path = os.path.join(input_dir, xml_file)
        output_path = os.path.join(output_dir, xml_file.replace('.xml', '.txt'))
        if os.path.exists(output_path):
            print(f"文件已存在，跳过处理: {output_path}")
            continue
        try:
            processed_text = preprocess_text(input_path)
            with open(output_path, 'w') as out_file:
                out_file.write(processed_text)
        except Exception as e:
            print(f"处理文件 {xml_file} 时出错: {e}")

# 运行数据预处理
process_directory('data/raw', 'data/processed')
