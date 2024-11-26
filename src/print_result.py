from pathlib import Path
import logging
import json
import sys
from datetime import datetime
import pandas as pd

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def print_model_results(result_path: Path):
    """打印模型结果"""
    try:
        if not result_path.exists():
            logging.error(f"结果文件不存在: {result_path}")
            return

        # 读取结果
        with open(result_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # 打印评估指标
        metrics = results.get('evaluation_metrics', {})
        logging.info("\n=== 整体评估指标 ===")
        logging.info(f"最优分数: {metrics.get('optimal_score', 'N/A'):.4f}")
        logging.info(f"平均分数: {metrics.get('avg_score', 'N/A'):.4f}")
        logging.info(f"标准差: {metrics.get('std_score', 'N/A'):.4f}")

        # 创建一个列表来存储所有模型的数据
        models_data = []
        
        # 打印每个模型的详细信息
        top_models = results.get('top_5_params', [])
        logging.info("\n=== 前5个最佳模型 ===")
        
        for i, model in enumerate(top_models, 1):
            logging.info(f"\n模型 {i}:")
            logging.info(f"参数设置:")
            logging.info(f"- min_freq: {model.get('min_freq', 'N/A')}")
            logging.info(f"- max_freq: {model.get('max_freq', 'N/A')}")
            logging.info(f"- alpha: {model.get('alpha', 'N/A')}")
            logging.info(f"- eta: {model.get('eta', 'N/A')}")
            logging.info(f"- passes: {model.get('n_passes', 'N/A')}")
            logging.info(f"测试分数: {model.get('test_score', 'N/A'):.4f}")
            
            # 收集模型数据
            model_data = {
                '模型序号': i,
                '最小词频': model.get('min_freq', 'N/A'),
                '最大词频': model.get('max_freq', 'N/A'),
                'alpha': model.get('alpha', 'N/A'),
                'eta': model.get('eta', 'N/A'),
                'passes': model.get('n_passes', 'N/A'),
                '测试分数': model.get('test_score', 'N/A')
            }
            
            # 打印主题词
            logging.info("\n主题词:")
            topic_words = model.get('topic_words', '').split('\n')
            for j, topic in enumerate(topic_words, 1):
                if topic:
                    logging.info(topic)
                    model_data[f'主题{j}'] = topic
            
            models_data.append(model_data)
            logging.info("-" * 80)

        # 创建DataFrame并显示
        df = pd.DataFrame(models_data)
        logging.info("\n=== 模型结果汇总表 ===")
        logging.info("\n" + df.to_string())
        
        # 保存为Excel和CSV
        summaries_dir = result_path.parent.parent / 'summaries'
        summaries_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = summaries_dir / f'model_summary_{timestamp}.xlsx'
        csv_path = summaries_dir / f'model_summary_{timestamp}.csv'
        
        df.to_excel(excel_path, index=False)
        df.to_csv(csv_path, index=False)
        
        logging.info(f"\n汇总表已保存至:")
        logging.info(f"Excel: {excel_path}")
        logging.info(f"CSV: {csv_path}")

    except Exception as e:
        logging.error(f"读取结果时发生错误: {str(e)}")

def main():
    """主函数"""
    try:
        setup_logging()
        base_dir = Path(__file__).resolve().parents[1]
        
        # 定位结果文件
        results_path = base_dir / "experiments/lda/results/detailed_results.json"
        print_model_results(results_path)
        
    except Exception as e:
        logging.error(f"执行过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()