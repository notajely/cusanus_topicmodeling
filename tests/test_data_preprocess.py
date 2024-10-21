import unittest
from src.data_preprocess import preprocess_tei_data

class TestPreprocessTeiData(unittest.TestCase):
    def setUp(self):
        # 创建一个临时的 XML 文件内容用于测试
        self.test_content = '''<TEI>
                                <text>
                                    <body>
                                        <p>This is a test sermon.</p>
                                    </body>
                                </text>
                            </TEI>'''

    def test_preprocess_tei_data(self):
        # 使用临时内容进行测试
        with open('test_file.xml', 'w', encoding='utf-8') as f:
            f.write(self.test_content)
        
        result = preprocess_tei_data('test_file.xml')
        self.assertEqual(result, 'This is a test sermon.')

    def tearDown(self):
        # 删除临时的 XML 文件
        import os
        if os.path.exists('test_file.xml'):
            os.remove('test_file.xml')

if __name__ == '__main__':
    unittest.main()