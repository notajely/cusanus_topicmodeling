
# Cusanus Sermon Topic Modeling

## Project Overview

This project applies **Latent Dirichlet Allocation (LDA)** to analyze and discover themes in **293 sermons** by Nicholas of Cusa (Cusanus).

## Structure

- **`data/`** - Contains raw and processed data.
- **`results/`** - Stores model outputs and visualizations.
- **`notebooks/`** - Contains the Jupyter notebooks for each stage of analysis:
  - `01_preprocessing.ipynb`: Preprocessing the sermon texts.
  - `02_exploratory_data_analysis.ipynb`: Data exploration and initial visualizations.
  - `03_topic_modeling_and_experiments.ipynb`: Topic modeling and parameter tuning.
  - `04_model_evaluation.ipynb`: Evaluating the LDA model.
  - `05_final_model_visualization.ipynb`: Final topic visualizations.
  - `06_results_visualization.ipynb`: Visualizing model results and topics.
  - `07_project_summary.ipynb`: Summarizing findings.

## Getting Started

1. **Clone the repository**:

   ```bash
   git clone <repository_url>
   cd Cusanus_Topic_Modeling
   ```
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Preprocess the data**: Run the `01_preprocessing.ipynb` notebook to clean the sermon texts.
4. **Run the topic modeling pipeline**: Follow the notebooks in sequence to perform data analysis and modeling.

## Contact

For questions or collaboration, feel free to contact the project maintainer.
