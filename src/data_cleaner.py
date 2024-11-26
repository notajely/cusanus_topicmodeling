import os
from pathlib import Path
import re
from typing import List, Set, Dict
from collections import Counter

def load_latin_ordinals() -> Set[str]:
    """加载拉丁语序数词列表"""
    ordinals = {
        'primus',      # 第一
        'secundus',    # 第二
        'tertius',     # 第三
        'quartus',     # 第四
        'quintus',     # 第五
        'sextus',      # 第六
        'septimus',    # 第七
        'octavus',     # 第八
        'nonus',       # 第九
        'decimus',     # 第十
        'undecimus',   # 第十一
        'duodecimus',  # 第十二
        'tertius decimus',  # 第十三
        'quartus decimus',  # 第十四
        'quintus decimus',  # 第十五
        # 阴性形式
        'prima', 'secunda', 'tertia', 'quarta', 'quinta',
        'sexta', 'septima', 'octava', 'nona', 'decima',
        # 中性形式
        'primum', 'secundum', 'tertium', 'quartum', 'quintum',
        'sextum', 'septimum', 'octavum', 'nonum', 'decimum'
    }
    return ordinals

def clean_text(text: str, ordinals: Set[str]) -> tuple[str, Dict[str, int]]:
    """清理文本中的无关内容，并返回清理统计"""
    removed_words = Counter()
    
    # 1. 移除 "Paragraph N:" 模式
    text = re.sub(r'Paragraph\s+\d+:', '', text)
    removed_words['PARAGRAPH'] = len(re.findall(r'Paragraph\s+\d+:', text))
    
    # 2. 移除标点符号（保留空格）
    text = re.sub(r'[.,;:!?"\'()\[\]{}]', ' ', text)
    
    # 3. 分词并清理
    words = text.split()
    cleaned_words = []
    
    for word in words:
        word = word.strip()
        
        # 跳过空字符串
        if not word:
            continue
            
        # 跳过纯数字
        if word.isdigit():
            removed_words['NUMBERS'] += 1
            continue
            
        # 跳过序数词
        if word.lower() in ordinals:
            removed_words[word] += 1
            continue
            
        # 跳过包含 "paragraph" 的词（不区分大小写）
        if 'paragraph' in word.lower():
            removed_words['PARAGRAPH'] += 1
            continue
        
        cleaned_words.append(word)
    
    # 4. 合并词语，确保单词间只有一个空格
    cleaned_text = ' '.join(cleaned_words)
    
    # 5. 移除多余的空格
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text, dict(removed_words)

def process_file(file_path: Path, ordinals: Set[str]) -> Dict[str, int]:
    """处理单个文件"""
    try:
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 清理文本并获取统计
        cleaned_content, removed_stats = clean_text(content, ordinals)
        
        # 如果有清理内容，才写入文件
        if removed_stats:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            print(f"\n处理文件: {file_path.name}")
            print("删除的内容:")
            for item, count in removed_stats.items():
                if item == 'NUMBERS':
                    print(f"  - 数字: {count}处")
                elif item == 'PARAGRAPH':
                    print(f"  - Paragraph标记: {count}处")
                else:
                    print(f"  - {item}: {count}次")
            
        return removed_stats
        
    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {str(e)}")
        return {}

def clean_dataset(data_dir: Path) -> None:
    """清理数据集中的无关内容"""
    try:
        # 加载序数词列表
        ordinals = load_latin_ordinals()
        print("\n要清理的内容:")
        print("1. 拉丁语序数词:")
        for word in sorted(ordinals):
            print(f"  - {word}")
        print("2. Paragraph标记")
        print("3. 数字")
        print("4. 标点符号")
        
        total_stats = Counter()
        files_processed = 0
        files_modified = 0
        
        # 处理训练集
        train_dir = data_dir / 'train_set'
        if train_dir.exists():
            print("\n处理训练集...")
            for file_path in train_dir.glob('*.txt'):
                stats = process_file(file_path, ordinals)
                if stats:
                    files_modified += 1
                    total_stats.update(stats)
                files_processed += 1
        
        # 处理测试集
        test_dir = data_dir / 'test_set'
        if test_dir.exists():
            print("\n处理测试集...")
            for file_path in test_dir.glob('*.txt'):
                stats = process_file(file_path, ordinals)
                if stats:
                    files_modified += 1
                    total_stats.update(stats)
                files_processed += 1
        
        # 打印总结
        print("\n清理任务完成!")
        print(f"处理文件总数: {files_processed}")
        print(f"修改文件数量: {files_modified}")
        if total_stats:
            print("\n总共删除的内容:")
            for item, count in total_stats.most_common():
                if item == 'NUMBERS':
                    print(f"  - 数字: {count}处")
                elif item == 'PARAGRAPH':
                    print(f"  - Paragraph标记: {count}处")
                else:
                    print(f"  - {item}: {count}次")
        else:
            print("\n没有找到需要清理的内容")
            
    except Exception as e:
        print(f"清理数据集时发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    # 设置工作目录为项目根目录
    project_dir = Path('/Users/jessie/Documents/Projects/Cusanus_Topic_Modeling')
    os.chdir(project_dir)
    
    # 获取数据目录
    data_dir = Path('data')
    
    print(f"当前工作目录: {os.getcwd()}")
    print(f"数据目录: {data_dir.absolute()}")
    
    # 检查目录是否存在
    if not (data_dir / 'train_set').exists():
        raise FileNotFoundError(f"找不到训练集目录: {data_dir / 'train_set'}")
    if not (data_dir / 'test_set').exists():
        raise FileNotFoundError(f"找不到测试集目录: {data_dir / 'test_set'}")
    
    # 运行清理
    clean_dataset(data_dir)