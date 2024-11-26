import os
from pathlib import Path

# 项目结构
PROJECT_STRUCTURE = {
    "src": {
        "__init__.py": """
# src模块初始化文件
""",
        
        "data_loader.py": """
from pathlib import Path
from typing import List, Tuple, Dict
from gensim.corpora import Dictionary
import logging

def load_documents(data_dir: Path, subset: str = 'train') -> List[List[str]]:
    \"\"\"加载文档数据
    
    Args:
        data_dir: 数据目录路径
        subset: 'train' 或 'test'
        
    Returns:
        处理后的文档列表
    \"\"\"
    texts = []
    data_path = data_dir / subset
    
    try:
        for file_path in data_path.glob('*.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().split()  # 简单分词
                texts.append(text)
                logging.info(f"成功加载文档: {file_path}")
    except Exception as e:
        logging.error(f"加载文档时发生错误: {str(e)}")
        raise
        
    return texts

def prepare_corpus(texts: List[List[str]]) -> Tuple[Dictionary, List[List[int]]]:
    \"\"\"准备语料库
    
    Args:
        texts: 文档列表
        
    Returns:
        (dictionary, corpus) 元组
    \"\"\"
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus
""",
        
        "evaluator.py": """
from typing import Dict, Any, List
import numpy as np
from gensim.models import LdaModel
from scipy.stats import entropy
from itertools import combinations

def evaluate_single_parameter(params: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"评估单个参数组合的性能
    
    Args:
        params: 参数字典，包含模型参数和评估所需的数据
        
    Returns:
        包含评估结果的字典
    \"\"\"
    try:
        corpus = params.pop('corpus')
        dictionary = params.pop('dictionary')
        texts = params.pop('texts')
        
        # 训练模型
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            **params
        )
        
        # 计算评估指标
        coherence = calculate_coherence(model, texts)
        perplexity = model.log_perplexity(corpus)
        topic_diversity = calculate_topic_diversity(model)
        
        return {
            'params': params,
            'coherence': coherence,
            'perplexity': perplexity,
            'topic_diversity': topic_diversity,
            'score': coherence * topic_diversity / abs(perplexity)
        }
    except Exception as e:
        print(f"评估参数时发生错误: {str(e)}")
        return {
            'params': params,
            'error': str(e),
            'score': float('-inf')
        }

def calculate_coherence(model: LdaModel, texts: List[List[str]]) -> float:
    \"\"\"计算主题一致性\"\"\"
    # 这里实现主题一致性计算逻辑
    return 0.5  # 示例返回值

def calculate_topic_diversity(model: LdaModel) -> float:
    \"\"\"计算主题多样性\"\"\"
    # 获取所有主题的词分布
    topic_words = []
    for topic_id in range(model.num_topics):
        top_words = [word for word, _ in model.show_topic(topic_id, topn=20)]
        topic_words.append(set(top_words))
    
    # 计算主题间的平均Jaccard距离
    distances = []
    for t1, t2 in combinations(topic_words, 2):
        jaccard = 1 - len(t1.intersection(t2)) / len(t1.union(t2))
        distances.append(jaccard)
    
    return np.mean(distances) if distances else 0.0
""",
        
        "lda_trainer.py": """
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Any
from pathlib import Path
import itertools
import logging
from .evaluator import evaluate_single_parameter
from .data_loader import load_documents, prepare_corpus

class LDATrainer:
    def __init__(self, data_dir: Path):
        \"\"\"初始化LDA训练器
        
        Args:
            data_dir: 数据目录路径
        \"\"\"
        self.data_dir = data_dir
        self.texts = load_documents(data_dir)
        self.dictionary, self.corpus = prepare_corpus(self.texts)
    
    def grid_search(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        \"\"\"执行参数网格搜索
        
        Args:
            param_grid: 参数网格字典
            
        Returns:
            评估结果列表
        \"\"\"
        parameter_combinations = self._generate_parameter_combinations(param_grid)
        
        # 为每个参数组合添加必要的数据
        for params in parameter_combinations:
            params.update({
                'corpus': self.corpus,
                'dictionary': self.dictionary,
                'texts': self.texts
            })
        
        try:
            with Pool(processes=cpu_count()) as pool:
                results = pool.map(evaluate_single_parameter, parameter_combinations)
            return results
        except Exception as e:
            logging.error(f"网格搜索过程中发生错误: {str(e)}")
            raise
    
    def _generate_parameter_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        \"\"\"生成参数组合\"\"\"
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = itertools.product(*values)
        return [dict(zip(keys, combo)) for combo in combinations]
"""
    },
    
    "scripts": {
        "run_lda.py": """
from pathlib import Path
import logging
import json
from datetime import datetime
from src.lda_trainer import LDATrainer

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 设置路径
    data_dir = Path("data/processed")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义参数网格
    param_grid = {
        "num_topics": [5, 10, 15],
        "alpha": ["symmetric", "auto"],
        "passes": [10, 20],
        "random_state": [42]
    }
    
    try:
        # 初始化训练器
        trainer = LDATrainer(data_dir)
        
        # 执行网格搜索
        logging.info("开始参数网格搜索...")
        results = trainer.grid_search(param_grid)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = results_dir / f"results_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"结果已保存至: {results_path}")
        
        # 输出最佳结果
        best_result = max(results, key=lambda x: x.get('score', float('-inf')))
        logging.info(f"\\n最佳结果:")
        logging.info(f"参数: {best_result['params']}")
        logging.info(f"分数: {best_result['score']:.4f}")
        
    except Exception as e:
        logging.error(f"执行过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()
"""
    }
}

def create_directory_structure(base_path: str, structure: dict):
    """递归创建目录结构和文件"""
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        
        if isinstance(content, dict):
            # 如果是目录
            os.makedirs(path, exist_ok=True)
            create_directory_structure(path, content)
        else:
            # 如果是文件
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content.strip())

def main():
    # 获取当前目录
    current_dir = os.getcwd()
    
    # 创建项目结构
    create_directory_structure(current_dir, PROJECT_STRUCTURE)
    
    # 创建其他必要的目录
    for directory in ['data/processed/train', 'data/processed/test', 'results']:
        os.makedirs(os.path.join(current_dir, directory), exist_ok=True)
    
    print("项目结构已创建完成！")
    print("\n目录结构：")
    for root, dirs, files in os.walk(current_dir):
        level = root.replace(current_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

if __name__ == "__main__":
    main()