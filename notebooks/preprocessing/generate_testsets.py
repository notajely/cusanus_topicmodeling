import os
import random
from tqdm import tqdm
import shutil

class TestsetGenerator:
    def __init__(self, base_dir, test_percentage=0.1):
        """初始化测试集生成器"""
        self.base_dir = base_dir
        self.test_percentage = test_percentage
        
        # 定义各种预处理方法的目录
        self.preprocessed_dirs = {
            'stanza': os.path.join(base_dir, 'data/preprocessed/stanza'),
            'cltk_stanza': os.path.join(base_dir, 'data/preprocessed/cltk_stanza'),
            'cusanus': os.path.join(base_dir, 'data/preprocessed/cusanus')
        }
        
        # 定义对应的测试集目录
        self.testset_dirs = {
            'stanza': os.path.join(base_dir, 'data/testset/stanza'),
            'cltk_stanza': os.path.join(base_dir, 'data/testset/cltk_stanza'),
            'cusanus': os.path.join(base_dir, 'data/testset/cusanus')
        }
        
        # 创建测试集目录
        for dir_path in self.testset_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def copy_test_documents(self, method):
        """复制选中的测试文档到对应的测试集目录"""
        print(f"\n处理 {method} 的测试集...")
        
        preprocessed_dir = self.preprocessed_dirs[method]
        testset_dir = self.testset_dirs[method]
        
        # 确保源目录存在
        if not os.path.exists(preprocessed_dir):
            print(f"预处理目录不存在: {preprocessed_dir}")
            return
        
        # 获取所有预处理后的文档
        all_docs = [f for f in os.listdir(preprocessed_dir) if f.endswith('_lemmatized.txt')]
        total_docs = len(all_docs)
        
        # 计算测试集大小并随机选择文档
        test_size = min(max(1, int(total_docs * self.test_percentage)), total_docs)  # 修改这行
        test_docs = random.sample(all_docs, test_size)
        
        # 复制文件
        for doc in tqdm(test_docs, desc=f"复制 {method} 测试文档"):
            source_path = os.path.join(preprocessed_dir, doc)
            dest_path = os.path.join(testset_dir, doc)
            
            try:
                shutil.copy2(source_path, dest_path)
            except Exception as e:
                print(f"复制文件 {doc} 时出错: {e}")
        
        print(f"已选择并复制 {len(test_docs)} 个文档作为 {method} 的测试集")

    def generate_all_testsets(self):
        """为所有预处理方法生成测试集"""
        print(f"开始生成测试集（测试集比例：{self.test_percentage * 100}%）")
        
        for method in self.preprocessed_dirs.keys():
            self.copy_test_documents(method)

def main():
    # 设置项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 创建测试集生成器并生成测试集
    generator = TestsetGenerator(base_dir, test_percentage=0.1)
    generator.generate_all_testsets()

if __name__ == "__main__":
    main()