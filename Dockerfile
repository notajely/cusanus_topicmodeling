# 使用 Python 3.10 作为基础镜像
FROM python:3.10

# 设置工作目录
WORKDIR /app

# 复制当前项目的文件到容器中
COPY . .

# 安装 conda 环境管理工具（可选）
RUN apt-get update && apt-get install -y wget bzip2 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && /opt/conda/bin/conda init bash

# 将 conda 加入 PATH
ENV PATH="/opt/conda/bin:${PATH}"

# 创建并激活 conda 虚拟环境
RUN conda create --name latinbert python=3.10 && echo "source activate latinbert" > ~/.bashrc

# 安装 PyTorch（根据系统 GPU/CPU 需求自定义安装）
RUN pip install torch torchvision torchaudio

# 安装项目的 requirements 文件中的依赖
RUN pip install -r requirements.txt

# 安装拉丁语的分词器模型
RUN python3 -c "from cltk.data.fetch import FetchCorpus; corpus_downloader = FetchCorpus(language='lat'); corpus_downloader.import_corpus('lat_models_cltk')"

# 下载预训练 BERT 模型
RUN ./scripts/download.sh

# 设置默认命令
CMD ["bash"]
