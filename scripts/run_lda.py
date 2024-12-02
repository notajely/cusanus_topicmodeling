from pathlib import Path
import logging
import json
from datetime import datetime
import sys
from tqdm import tqdm
import time
import pickle
import pandas as pd
from itertools import product

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.lda_trainer import LDATrainer
from src.result_analyzer import ResultAnalyzer

def setup_logging(log_dir: Path) -> Path:
    """设置日志配置"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"lda_experiment_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def save_results_to_excel(results_df: pd.DataFrame, save_path: Path):
    """保存结果到Excel文件"""
    try:
        # 创建新的DataFrame，按照指定格式重组数据
        formatted_results = []
        for _, row in results_df.iterrows():
            # 从模型中获取主题词
            model = row['model']
            topics = []
            if model:
                for topic_id in range(row['num_topics']):
                    topic_words = [word for word, _ in model.show_topic(topic_id, topn=20)]
                    topics.append(', '.join(topic_words))
            
            entry = {
                'Exp. ID': f"LDA-{len(formatted_results)+1}",
                'Lemmatization Method': 'standard',
                'Threshold': f"{row['min_freq']}-{row['max_freq']}",
                'alpha': row['alpha'],
                'eta': row['eta'],
                'n_topics': row['num_topics'],
                'n_passes': 10,
                'optimal_score': row['optimal_score'],
                'npmi_score': row['npmi_score'],         # 添加 NPMI 分数
                'diversity_score': row['diversity_score'], # 添加多样性分数
                'perplexity': row['perplexity'],         # 添加困惑度
                'topic_words': topics[0] if topics else ''
            }
            
            # 添加其他主题的词
            for i, topic in enumerate(topics[1:], 1):
                entry[f'Topic_{i}'] = topic
            
            # 添加文档主题分布
            document_topics = row.get('document_topics', [])
            for doc_id, doc_topics in enumerate(document_topics):
                entry[f'Doc_{doc_id}_topics'] = ', '.join([f"Topic {topic_id}: {prob:.2f}" for topic_id, prob in doc_topics])
            
            formatted_results.append(entry)
        
        formatted_df = pd.DataFrame(formatted_results)
        
        # 如果文件已存在，读取现有数据并追加
        if save_path.exists():
            existing_df = pd.read_excel(save_path)
            formatted_df = pd.concat([existing_df, formatted_df], ignore_index=True)
        
        # 使用xlsxwriter引擎保存，以便应用格式
        with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
            formatted_df.to_excel(writer, index=False, sheet_name='Summary')
            
            workbook = writer.book
            worksheet = writer.sheets['Summary']
            
            # 设置格式
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D9E1F2',
                'border': 1,
                'text_wrap': True,
                'align': 'center',
                'valign': 'vcenter'
            })
            
            data_format = workbook.add_format({
                'text_wrap': True,
                'valign': 'top',
                'border': 1
            })
            
            # 更新列宽设置
            base_columns = {
                'A': 10,   # Exp. ID
                'B': 20,   # Lemmatization Method
                'C': 15,   # Threshold
                'D': 10,   # alpha
                'E': 10,   # eta
                'F': 10,   # n_topics
                'G': 10,   # n_passes
                'H': 15,   # optimal_score
                'I': 15,   # npmi_score
                'J': 15,   # diversity_score
                'K': 15,   # perplexity
                'L': 100,  # topic_words
            }
            
            # 为每个主题和文档主题分布添加列宽
            topic_columns = {chr(ord('L') + i): 100 for i in range(row['num_topics'] - 1)}
            doc_topic_columns = {chr(ord('L') + len(topic_columns) + i): 100 for i in range(len(document_topics))}
            column_widths = {**base_columns, **topic_columns, **doc_topic_columns}
            
            for col, width in column_widths.items():
                worksheet.set_column(f'{col}:{col}', width)
            
            # 应用格式
            for col_num, value in enumerate(formatted_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            for row in range(1, len(formatted_df) + 1):
                for col in range(len(formatted_df.columns)):
                    worksheet.write(row, col, formatted_df.iloc[row-1, col], data_format)
        
        logging.info(f"结果已保存到: {save_path}")
        
    except Exception as e:
        logging.error(f"保存结果到Excel时发生错误: {str(e)}")

def run_experiment(base_dir: Path):
    """运行LDA实验"""
    try:
        # 设置目录
        data_dir = base_dir  / 'experiments' / 'lda' / 'spacy'
        experiment_dir = base_dir / "experiments/lda/spacy"
        log_dir = base_dir / "experiments/lda/spcay/logs"
        results_dir = experiment_dir / "experiments/lda/spacy/results"
        
        for dir_path in [experiment_dir, log_dir, results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        log_file = setup_logging(log_dir)
        
        # 创建结果保存路径
        results_file = results_dir / "all_experiments_results.xlsx"
        
        # 初始化trainer
        trainer = LDATrainer(data_dir, experiment_dir)
        
        # 定义参数网格
        param_grid = {
            'min_freqs': [2],
            'max_freqs': [100],
            'num_topics': [10],
            'alpha_range': ["symmetric", 0.3, 0.5],
            'eta_range': ["symmetric", 0.3, 0.5]  # "symmetric", 0.3, 
        }
        
        # 设置训练轮数
        n_passes = 10  # 这里设置你想要的训练轮数
        
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(product(*param_grid.values()))
        total_combinations = len(param_values)
        
        logging.info(f"总参数组合数: {total_combinations}")
        
        # 初始化 ResultAnalyzer
        result_analyzer = ResultAnalyzer(base_dir)
        
        # 创建主进度条
        with tqdm(total=total_combinations, desc="参数组合评估进度") as pbar:
            for i in range(total_combinations):
                current_values = param_values[i]
                current_combination = dict(zip(param_names, current_values))
                
                try:
                    # 运行模型训练代码
                    current_results_df = trainer.grid_search(
                        param_grid={k: [v] for k, v in current_combination.items()},
                        n_passes=n_passes
                    )
                    
                    if current_results_df is not None and not current_results_df.empty:
                        # 添加时间戳列
                        current_results_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # 保存当前结果到Excel
                        save_results_to_excel(current_results_df, results_file)
                        
                        # 获取最佳模型
                        best_model = current_results_df.iloc[0]['model']
                        
                        # 修改这里：使用测试集的文档名称和语料库
                        test_filenames = [doc[0] for doc in trainer.test_texts]  # 获取测试集文档名
                        test_corpus = [trainer.dictionary.doc2bow(doc[1]) for doc in trainer.test_texts]
                    
                        # 创建测试集的文档主题分布汇总
                        doc_topics_df = result_analyzer.create_document_topic_summary(
                            model=best_model,
                            corpus=test_corpus,  # 使用测试集语料库
                            filenames=test_filenames,  # 使用测试集文档名
                            min_probability=0.1
                        )
                        
                        # 保存文档主题分布到单独的Excel文件
                        doc_topics_file = results_dir / f"test_document_topics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                        
                        # 使用xlsxwriter保存格式化的Excel
                        with pd.ExcelWriter(doc_topics_file, engine='xlsxwriter') as writer:
                            doc_topics_df.to_excel(writer, index=False, sheet_name='Document Topics')
                            
                            workbook = writer.book
                            worksheet = writer.sheets['Document Topics']
                            
                            # 设置格式
                            header_format = workbook.add_format({
                                'bold': True,
                                'bg_color': '#D9E1F2',
                                'border': 1,
                                'text_wrap': True,
                                'align': 'center',
                                'valign': 'vcenter'
                            })
                            
                            data_format = workbook.add_format({
                                'text_wrap': True,
                                'valign': 'top',
                                'border': 1
                            })
                            
                            # 设置列宽
                            worksheet.set_column('A:A', 30)  # Document name
                            worksheet.set_column('B:B', 50)  # Dominant Topics
                            worksheet.set_column('C:C', 15)  # Number of Topics
                            worksheet.set_column('D:Z', 12)  # Topic probabilities
                            
                            # 应用格式
                            for col_num, value in enumerate(doc_topics_df.columns.values):
                                worksheet.write(0, col_num, value, header_format)
                            
                            for row in range(1, len(doc_topics_df) + 1):
                                for col in range(len(doc_topics_df.columns)):
                                    worksheet.write(row, col, doc_topics_df.iloc[row-1, col], data_format)
                        
                        logging.info(f"文档主题分布已保存至: {doc_topics_file}")
                    
                except Exception as e:
                    logging.error(f"评估参数组合时发生错误: {str(e)}")
                    continue
                
                pbar.update(1)
        
        logging.info("实验完成！")
        logging.info(f"详细日志已保存至: {log_file}")
        logging.info(f"所有实验结果已保存至: {results_file}")
                
    except Exception as e:
        logging.error(f"实验过程中发生错误: {str(e)}")
        raise

def main():
    """主函数"""
    try:
        base_dir = Path(__file__).resolve().parents[1]
        run_experiment(base_dir)
    except Exception as e:
        logging.error(f"执行过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()