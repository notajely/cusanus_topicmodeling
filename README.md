# Cusanus Sermon Topic Modeling

## Project Overview

This project applies **Latent Dirichlet Allocation (LDA)** and other potential topic modeling techniques to analyze and discover themes in **293 sermons** by Nicholas of Cusa (Cusanus). The project includes data preprocessing, exploratory analysis, topic modeling, and result evaluation.

## Project Structure

- **`data/`**

  - **`raw/`**: Contains the original XML sermon files.
  - **`processed/`**: Stores the cleaned and processed sermon data.
- **`notebooks/`**

  - `01_preprocessing.ipynb`: Preprocessing the sermon texts from the raw XML format.
  - `02_exploratory_data_analysis.ipynb`: Initial data exploration and visualization.
  - `03_topic_modeling_and_experiments.ipynb`: LDA topic modeling and parameter tuning experiments.
  - `04_model_evaluation.ipynb`: Evaluation of the LDA model using coherence scores and other metrics.
  - `05_final_model_visualization.ipynb`: Visualization of the final topic model results.
  - `06_results_visualization.ipynb`: Further visualization of modeling results, including topic distributions.
  - `07_project_summary.ipynb`: Summary of the entire project, including key findings and insights.
- **`results/`**

  - Stores the model outputs, visualizations, and evaluation metrics.
- **`src/`**

  - `data_preprocess.py`: Script for preprocessing the raw XML files into a format suitable for topic modeling.
- **`tests/`**

  - Test scripts for ensuring data processing and modeling components work as expected.


# Cusanus Sermon Topic Modeling

**Cusanus Sermon Topic Modeling** is a project focused on extracting thematic insights from the 293 sermons of Nicholas of Cusa using topic modeling techniques, specifically **Latent Dirichlet Allocation (LDA)**. The project workflow includes data preprocessing, topic modeling, hyperparameter tuning, model evaluation, and visualization.

## Requirements

To install all dependencies, run:

```bash
pip install -r requirements.txt
```
