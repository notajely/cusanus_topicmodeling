import os
import gensim
from gensim import corpora
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
import collections
import json
import time
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import matplotlib.pyplot as plt
from gensim.models import TfidfModel

def main():
    # Set working directory
    os.chdir('/Users/jessie/Documents/Projects/Cusanus_Topic_Modeling')

    # Logging setup
    experiment_id = f"lda_experiment_{time.strftime('%Y%m%d%H%M%S')}"
    experiment_dir = os.path.join('experiments', experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    log_file_path = os.path.join(experiment_dir, f"{experiment_id}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    # Load test set
    testset_path = os.path.join('data', 'testset_paragraphs_level.json')
    with open(testset_path, 'r', encoding='utf-8') as json_file:
        testset_data = json.load(json_file)

    # Extract paragraph texts
    documents = [paragraph["content"].split() for doc in testset_data["documents"] for paragraph in doc["paragraphs"]]

    # Dictionary and corpus for LDA
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    tfidf = TfidfModel(corpus)

    # Calculate and filter common words based on TF-IDF
    word_tfidf = collections.defaultdict(float)
    word_counts = collections.defaultdict(int)
    for doc in corpus:
        for word_id, tfidf_value in tfidf[doc]:
            word_tfidf[word_id] += tfidf_value
            word_counts[word_id] += 1
    average_tfidf = {word_id: total_tfidf / word_counts[word_id] for word_id, total_tfidf in word_tfidf.items()}
    tfidf_threshold = 0.2
    common_words = {dictionary[word_id] for word_id, avg_tfidf in average_tfidf.items() if avg_tfidf < tfidf_threshold}
    removed_common_words_path = os.path.join(experiment_dir, "removed_common_words_tfidf.txt")
    with open(removed_common_words_path, "w", encoding="utf-8") as f:
        for word in common_words:
            f.write(word + "\n")

    # Filter common words
    filtered_corpus = [
        [(id, freq) for id, freq in doc if dictionary[id] not in common_words]
        for doc in corpus
    ]
    dictionary.filter_tokens(bad_ids=[dictionary.token2id[word] for word in common_words if word in dictionary.token2id])
    dictionary.compactify()
    filtered_corpus = [dictionary.doc2bow(text) for text in documents]

    # LDA hyperparameter tuning
    alpha_values = [0.01, 0.05, 0.1]
    eta_values = [0.01, 0.05, 0.1]
    results = []

    for alpha in alpha_values:
        for eta in eta_values:
            lda_model = gensim.models.LdaModel(
                corpus=filtered_corpus,
                id2word=dictionary,
                num_topics=10,
                random_state=42,
                passes=50,
                iterations=200,
                alpha=alpha,
                eta=eta
            )
            coherence_model_npmi = CoherenceModel(model=lda_model, texts=documents, dictionary=dictionary, coherence='c_npmi')
            avg_npmi = coherence_model_npmi.get_coherence()
            unique_words = set()
            total_words = 0
            for topic in lda_model.show_topics(num_topics=10, num_words=20, formatted=False):
                words = [word for word, _ in topic[1]]
                unique_words.update(words)
                total_words += len(words)
            topic_diversity = len(unique_words) / total_words
            results.append({'alpha': alpha, 'eta': eta, 'avg_npmi': float(avg_npmi), 'topic_diversity': float(topic_diversity)})
            logger.info(f"Alpha: {alpha}, Eta: {eta}, Avg NPMI: {avg_npmi:.4f}, Topic Diversity: {topic_diversity:.2f}")

    # Find and log the best parameters
    best_result = max(results, key=lambda x: (x['avg_npmi'], x['topic_diversity']))
    for result in results:
        print(f"Alpha: {result['alpha']}, Eta: {result['eta']}, Avg NPMI: {result['avg_npmi']:.4f}, Topic Diversity: {result['topic_diversity']:.2f}")
    print(f"\nBest result: Alpha: {best_result['alpha']}, Eta: {best_result['eta']}, Avg NPMI: {best_result['avg_npmi']:.4f}, Topic Diversity: {best_result['topic_diversity']:.2f}")

    # Plot and save graphs
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for alpha in sorted(set(alpha_values)):
        alpha_npmi = [result['avg_npmi'] for result in results if result['alpha'] == alpha]
        eta_subset = [result['eta'] for result in results if result['alpha'] == alpha]
        plt.plot(eta_subset, alpha_npmi, marker='o', label=f'Alpha = {alpha}: Effect of Eta on NPMI')
    plt.axvline(best_result['eta'], color='gray', linestyle='--', label=f'Best Eta = {best_result["eta"]}')
    plt.xlabel('Eta')
    plt.ylabel('Avg NPMI')
    plt.ylim([-0.440, -0.420])
    plt.title('NPMI vs Eta for different Alpha values')
    plt.legend()

    plt.subplot(1, 2, 2)
    for alpha in sorted(set(alpha_values)):
        alpha_diversity = [result['topic_diversity'] for result in results if result['alpha'] == alpha]
        eta_subset = [result['eta'] for result in results if result['alpha'] == alpha]
        plt.plot(eta_subset, alpha_diversity, marker='o', label=f'Alpha = {alpha}: Effect of Eta on Topic Diversity')
    plt.axvline(best_result['eta'], color='gray', linestyle='--', label=f'Best Eta = {best_result["eta"]}')
    plt.xlabel('Eta')
    plt.ylabel('Topic Diversity')
    plt.ylim([0.85, 0.92])
    plt.title('Topic Diversity vs Eta for different Alpha values')
    plt.legend()
    plt.tight_layout()
    image_path = os.path.join(experiment_dir, 'LDA_NPMI_TopicDiversity_vs_Alpha_Eta.png')
    plt.savefig(image_path)
    plt.show()

    # Save parameters and model results
    params = {"num_topics": 10, "passes": 50, "iterations": 200, "alpha": best_result['alpha'], "eta": best_result['eta'], "tfidf_threshold": tfidf_threshold}
    params_path = os.path.join(experiment_dir, 'params.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)

    lda_model = gensim.models.LdaModel(
        corpus=filtered_corpus,
        id2word=dictionary,
        num_topics=10,
        random_state=42,
        passes=50,
        iterations=200,
        alpha=best_result['alpha'],
        eta=best_result['eta']
    )
    topics = lda_model.show_topics(num_topics=10, num_words=10, formatted=False)
    topic_keywords = {topic_id: [word for word, _ in words] for topic_id, words in topics}

    # Save topics and metrics
    results_file_path = os.path.join(experiment_dir, 'lda_results.txt')
    with open(results_file_path, 'w', encoding='utf-8') as f:
        f.write("LDA generated topics:\n")
        for idx, topic in lda_model.print_topics(num_topics=10, num_words=10):
            f.write(f"Topic {idx}: {topic}\n")
        f.write(f"\nPerplexity: {lda_model.log_perplexity(filtered_corpus)}\n")
        f.write(f"Topic Coherence (NPMI): {best_result['avg_npmi']}\n")
        f.write(f"Topic Diversity: {best_result['topic_diversity']}\n")

    # Save document-topic distributions
    doc_topic_distributions = []
    for i, doc in enumerate(filtered_corpus):
        doc_topics = lda_model.get_document_topics(doc)
        doc_index, para_index = 0, i
        while para_index >= len(testset_data["documents"][doc_index]["paragraphs"]):
            para_index -= len(testset_data["documents"][doc_index]["paragraphs"])
            doc_index += 1
        document_id = testset_data["documents"][doc_index]["document_id"]
        paragraph_num = para_index
        paragraph_content = testset_data["documents"][doc_index]["paragraphs"][paragraph_num]["content"]
        if paragraph_content:
            for topic_id, prob in doc_topics:
                doc_topic_distributions.append({"Document": document_id, "Paragraph": paragraph_num, "Content": paragraph_content, "Topic": topic_id, "Topic Keywords": topic_keywords[topic_id], "Probability": float(prob)})

    # Save distributions to CSV and JSON
    csv_output_path = os.path.join(experiment_dir, 'document_topic_distributions.csv')
    topic_distributions_df = pd.DataFrame(doc_topic_distributions)
    topic_distributions_df.to_csv(csv_output_path, index=False)

    json_output_path = os.path.join(experiment_dir, 'document_topic_distributions.json')
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(doc_topic_distributions, json_file, ensure_ascii=False, indent=4)

    # Save pyLDAvis visualization
    lda_visualization = gensimvis.prepare(lda_model, filtered_corpus, dictionary, n_jobs=1)
    pyldavis_path = os.path.join(experiment_dir, 'lda_visualization.html')
    pyLDAvis.save_html(lda_visualization, pyldavis_path)

    print(f"\nBest result saved to: {experiment_dir}")

if __name__ == '__main__':
    main()