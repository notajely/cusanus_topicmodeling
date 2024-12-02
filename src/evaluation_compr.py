import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 读取added_words.txt文件
with open('expert_evaluation/added_words.txt', 'r') as f:
    added_words = [line.strip() for line in f.readlines()]

# 读取spacy_trainset_word_frequency.csv文件
df = pd.read_csv('expert_evaluation/spacy_trainset_word_frequency.csv')

# 过滤出added_words中的词频
filtered_df = df[df['Word'].isin(added_words)]

# 创建一个新的DataFrame，包含added_words和对应的词频
result_df = filtered_df[['Word', 'Frequency']]

# 保存为CSV文件
csv_file_path = 'expert_evaluation/added_words_frequencies.csv'
result_df.to_csv(csv_file_path, index=False)

# 设置区间
bins = [0, 2, 100] + list(range(200, 2001, 200)) + [float('inf')]
labels = ['1-2)', 'Within Threshold (2-100)'] + [f'{i}-{i+199}' for i in range(200, 2001, 200)] + ['2000+']

# 计算每个区间的词频计数
counts, _ = pd.cut(filtered_df['Frequency'], bins=bins, retbins=True, labels=labels)

# 生成条形图
plt.figure(figsize=(14, 6))  # 增加图表宽度
count_values = counts.value_counts().sort_index()

# 使用seaborn绘制条形图
sns.barplot(x=count_values.index, y=count_values.values, palette='Blues')

# 添加数据标签
for i in range(len(count_values)):
    plt.text(i, count_values.values[i], int(count_values.values[i]), ha='center', va='bottom')

# 设置图表标题和标签
plt.title('Granular Frequency Distribution of Added Words', fontsize=16)
plt.xlabel('Frequency Range', fontsize=14)
plt.ylabel('Word Count', fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=12)
plt.grid(axis='y', alpha=0.75)

# 保存图表
plt.tight_layout()
plt.savefig('expert_evaluation/added_words_frequency_distribution.png')
plt.show()