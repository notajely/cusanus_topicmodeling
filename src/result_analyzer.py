from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import json
import logging
from gensim.models import LdaModel
import xlsxwriter

class ResultAnalyzer:
    def __init__(self, base_dir: Path):
        """初始化结果分析器
        
        Args:
            base_dir: 项目根目录
            
        Raises:
            FileNotFoundError: 如果根目录不存在
        """
        if not base_dir.exists():
            raise FileNotFoundError(f"项目根目录不存在: {base_dir}")
            
        self.base_dir = base_dir
        self.summaries_dir = base_dir / 'experiments/lda/summaries'
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
    
    def create_summary_table(self, results_dict: Dict[str, Path], n_topics: int = 10) -> pd.DataFrame:
        """创建实验结果汇总表格
        
        Args:
            results_dict: 预处理方法到结果文件路径的映射
            n_topics: 主题数量
            
        Returns:
            包含实验结果汇总的DataFrame
            
        Raises:
            ValueError: 如果参数无效
        """
        if not isinstance(results_dict, dict):
            raise ValueError("results_dict必须是字典类型")
        if n_topics < 1:
            raise ValueError("n_topics必须为正整数")
            
        summary_data = []
        
        for prep_type, results_path in results_dict.items():
            try:
                if not results_path.exists():
                    logging.error(f"结果文件不存在: {results_path}")
                    continue
                    
                with open(results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                if not isinstance(results, dict):
                    logging.error(f"结果格式错误: {results_path}")
                    continue
                    
                top_5_params = results.get('top_5_params', [])
                if not isinstance(top_5_params, list):
                    logging.error(f"top_5_params 格式错误: {results_path}")
                    continue
                    
                for idx, params in enumerate(top_5_params, 1):
                    if not isinstance(params, dict):
                        logging.warning(f"跳过无效参数: index={idx}")
                        continue
                        
                    model_path = results_path.parent / f'model_{idx}.lda'
                    if model_path.exists():
                        try:
                            model = LdaModel.load(str(model_path))
                            topics_keywords = self._extract_topic_keywords(model, n_topics)
                            
                            entry = self._create_entry(prep_type, idx, params, n_topics)
                            entry.update(topics_keywords)
                            summary_data.append(entry)
                        except Exception as e:
                            logging.error(f"处理模型 {idx} 时发生错误: {str(e)}", exc_info=True)
                            continue
                    else:
                        logging.warning(f"模型文件不存在: {model_path}")
                    
            except Exception as e:
                logging.error(f"处理 {prep_type} 结果时发生错误: {str(e)}", exc_info=True)
                continue
        
        if not summary_data:
            logging.warning("没有找到有效的结果数据")
            return pd.DataFrame()
            
        summary_df = pd.DataFrame(summary_data)
        self._save_summary(summary_df, n_topics)
        return summary_df
    
    def _extract_topic_keywords(self, model: LdaModel, n_topics: int) -> Dict[str, str]:
        """提取每个主题的关键词
        
        Args:
            model: LDA模型
            n_topics: 主题数量
            
        Returns:
            主题ID到关键词列表的映射
            
        Raises:
            ValueError: 如果模型无效
        """
        if not isinstance(model, LdaModel):
            raise ValueError("无效的LDA模型")
            
        topics_keywords = {}
        for topic_id in range(n_topics):
            words = [word for word, _ in model.show_topic(topic_id, topn=10)]
            topics_keywords[f'Topic_{topic_id+1}'] = ', '.join(words)
        return topics_keywords
    
    def _create_entry(self, prep_type: str, idx: int, params: Dict, n_topics: int) -> Dict[str, Any]:
        """创建基本信息条目
        
        Args:
            prep_type: 预处理方法
            idx: 模型索引
            params: 模型参数
            n_topics: 主题数量
            
        Returns:
            包含模型信息的字典
        """
        try:
            return {
                'Exp. ID': f"{prep_type[:3]}-{idx}",
                'Lemmatization Method': prep_type,
                'Threshold': f"{params.get('min_freq', 'N/A')}-{params.get('max_freq', 'N/A')}",
                'alpha': params.get('alpha', 'N/A'),
                'eta': params.get('eta', 'auto'),
                'n_topics': n_topics,
                'n_passes': params.get('n_passes', 'N/A'),
                'optimal score': float(params.get('test_score', 0.0)),
                'topic_words': params.get('topic_words', '')
            }
        except Exception as e:
            logging.error(f"创建条目时发生错误: {str(e)}", exc_info=True)
            return {}
    
    def _save_summary(self, summary_df: pd.DataFrame, n_topics: int) -> None:
        """保存汇总表格为多种格式
        
        Args:
            summary_df: 汇总数据DataFrame
            n_topics: 主题数量
        """
        try:
            # 保存CSV
            summary_df.to_csv(self.summaries_dir / 'experiment_summary.csv', index=False)
            
            # 保存Excel（带格式）
            self._save_formatted_excel(summary_df, n_topics)
            
            # 保存Markdown
            with open(self.summaries_dir / 'experiment_summary.md', 'w', encoding='utf-8') as f:
                f.write(summary_df.to_markdown(index=False))
                
        except Exception as e:
            logging.error(f"保存汇总表格时发生错误: {str(e)}", exc_info=True)
    
    def _save_formatted_excel(self, summary_df: pd.DataFrame, n_topics: int) -> None:
        """保存格式化的Excel文件
        
        Args:
            summary_df: 汇总数据DataFrame
            n_topics: 主题数量
        """
        try:
            excel_path = self.summaries_dir / 'experiment_summary.xlsx'
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                summary_df.to_excel(writer, index=False, sheet_name='Summary')
                
                workbook = writer.book
                worksheet = writer.sheets['Summary']
                
                # 设置列宽
                column_widths = self._get_column_widths(n_topics)
                for col, width in column_widths.items():
                    worksheet.set_column(f'{col}:{col}', width)
                
                # 应用格式
                self._apply_excel_formatting(workbook, worksheet, summary_df)
                
        except Exception as e:
            logging.error(f"保存Excel文件时发生错误: {str(e)}", exc_info=True)
    
    def _get_column_widths(self, n_topics: int) -> Dict[str, int]:
        """获取Excel列宽设置
        
        Args:
            n_topics: 主题数量
            
        Returns:
            列标识符到列宽的映射
        """
        base_columns = {
            'A': 10,  # Exp. ID
            'B': 20,  # Lemmatization Method
            'C': 15,  # Threshold
            'D': 10,  # alpha
            'E': 10,  # eta
            'F': 10,  # n_topics
            'G': 10,  # n_passes
            'H': 15,  # optimal score
            'I': 100  # topic_words
        }
        topic_columns = {chr(ord('J') + i): 100 for i in range(n_topics)}
        return {**base_columns, **topic_columns}
    
    def _apply_excel_formatting(self, workbook: xlsxwriter.Workbook, 
                              worksheet: xlsxwriter.worksheet.Worksheet, 
                              df: pd.DataFrame) -> None:
        """应用Excel格式化
        
        Args:
            workbook: Excel工作簿
            worksheet: 工作表
            df: 数据DataFrame
        """
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
        
        # 应用表头格式
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # 应用数据格式
        for row in range(1, len(df) + 1):
            for col in range(len(df.columns)):
                worksheet.write(row, col, df.iloc[row-1, col], data_format)
    
    def create_document_topic_summary(self, model: LdaModel, corpus: List[List[int]], 
                                    filenames: List[str], min_probability: float = 0.1) -> pd.DataFrame:
        """创建文档-主题分布汇总表
        
        Args:
            model: 训练好的LDA模型
            corpus: 文档的BOW表示
            filenames: 文档文件名列表
            min_probability: 最小概率阈值，低于此值的主题分布将被过滤掉
            
        Returns:
            包含文档名称和主题分布的DataFrame
            
        Raises:
            ValueError: 如果参数无效
        """
        if not isinstance(model, LdaModel):
            raise ValueError("无效的LDA模型")
        if len(corpus) != len(filenames):
            raise ValueError("corpus和filenames长度不匹配")
        if not (0 <= min_probability <= 1):
            raise ValueError("min_probability必须在0到1之间")
            
        try:
            # 创建文档主题分布列表
            doc_topics = []
            
            for doc_idx, (bow, filename) in enumerate(zip(corpus, filenames)):
                # 获取文档的主题分布
                topic_dist = sorted(
                    model.get_document_topics(bow),
                    key=lambda x: x[1],  # 按概率排序
                    reverse=True  # 降序排列
                )
                
                # 过滤低概率的主题并格式化
                significant_topics = [
                    f"Topic {topic_id + 1}: {prob:.3f}"
                    for topic_id, prob in topic_dist
                    if prob >= min_probability
                ]
                
                # 创建条目
                entry = {
                    'Document': filename,
                    'Dominant Topics': ' | '.join(significant_topics),
                    'Number of Topics': len(significant_topics)
                }
                
                # 添加每个主题的概率（用于排序和过滤）
                for topic_id in range(model.num_topics):
                    prob = next((prob for t_id, prob in topic_dist if t_id == topic_id), 0.0)
                    entry[f'Topic_{topic_id + 1}'] = prob
                
                doc_topics.append(entry)
            
            # 创建DataFrame并按主题数量和最大主题概率排序
            df = pd.DataFrame(doc_topics)
            topic_columns = [f'Topic_{i+1}' for i in range(model.num_topics)]
            df['max_prob'] = df[topic_columns].max(axis=1)
            df = df.sort_values(['Number of Topics', 'max_prob'], ascending=[False, False])
            df = df.drop('max_prob', axis=1)
            
            # 保存结果
            self._save_document_topic_summary(df)
            
            return df
            
        except Exception as e:
            logging.error(f"创建文档主题分布汇总时发生错误: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def _save_document_topic_summary(self, df: pd.DataFrame) -> None:
        """保存文档主题分布汇总
        
        Args:
            df: 文档主题分布DataFrame
        """
        try:
            # 保存为Excel文件
            excel_path = self.summaries_dir / 'document_topic_distribution.xlsx'
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Document Topics')
                
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
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                for row in range(1, len(df) + 1):
                    for col in range(len(df.columns)):
                        worksheet.write(row, col, df.iloc[row-1, col], data_format)
            
            # 同时保存为CSV文件
            df.to_csv(self.summaries_dir / 'document_topic_distribution.csv', index=False)
            
        except Exception as e:
            logging.error(f"保存文档主题分布汇总时发生错误: {str(e)}", exc_info=True)