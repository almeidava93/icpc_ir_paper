# This module handles the evaluation step
# It generates metrics and graphs, and saves them to disk

from tqdm import tqdm
import jsonlines
import pandas as pd
from pathlib import Path

# Custom modules
from eval_functions import calculate_kappa_with_jaccard, precision_at_k, average_precision_at_k, plot_metrics, preprocess_labels


# Create relevant directories if they do not exist yet
metrics_path = Path('results','metrics')
if not metrics_path.exists():
    metrics_path.mkdir(parents=True, exist_ok=True)

results_path = Path('results','graphs')
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

# Load labeled data
eval_dataset_path = Path("data", "eval_dataset.csv")
true_labels_df = pd.read_csv(eval_dataset_path, index_col=0).dropna(subset=["relevant_results"]) # drop empty values in the relevant_results column

# Load results from information retrieval algorithms
queries_results_path = Path("results","queries_results.csv")
queries_results_df = pd.read_csv(queries_results_path, index_col=0).dropna()

# Load value counts for weighted mean
query_value_counts_path = Path("data","query_value_counts.csv")
query_value_counts = pd.read_csv(query_value_counts_path, index_col=0)

# List of algorithms to evaluate
targets = [
        "bm25_raw",
        "bm25_preprocessed",
        "levenshtein_distance",
        "openai_embeddings",
        "openai_embeddings_text-embedding-3-small",
        "openai_embeddings_text-embedding-3-large",
        "cohere_embeddings",
        "cohere_embeddings_v3.0_search_document",
        "cohere_embeddings_v3.0_search_query",
        "cohere_embeddings_v3.0_classification",
        "cohere_embeddings_v3.0_clustering",
        "gemini_embeddings_semantic_similarity",
        "gemini_embeddings_retrieval_query",
        "gemini_embeddings_retrieval_document",
        "gemini_embeddings_classification",
        "gemini_embeddings_clustering",
    ] 

# Compute all relevant metrics about each result of each algorithm
metrics = []
metrics_with_frequency = []
for index, row in tqdm(true_labels_df.iterrows()):
    all_targets_metric = {}
    for target in targets:
        retrieved = queries_results_df.iloc[index][target].split('|')
        reference = row['relevant_results'].split('|')
        result = [True if code in reference else False for code in retrieved]
        has_relevant_items = True if sum(result) > 0 else False
        query = queries_results_df.iloc[index]['query']
        value_count = query_value_counts.loc[query, 'count']
        target_metric = {
            f'{target}|P@1': precision_at_k(result, 1),
            f'{target}|P@5': precision_at_k(result, 5),
            f'{target}|P@10': precision_at_k(result, 10),
            f'{target}|AP@1': average_precision_at_k(result, 1),
            f'{target}|AP@5': average_precision_at_k(result, 5),
            f'{target}|AP@10': average_precision_at_k(result, 10),
            f'{target}|has_relevant_items': has_relevant_items,
            'value_count': int(value_count),
        }
        all_targets_metric = all_targets_metric | target_metric
    metrics.append(all_targets_metric)
    for _ in range(value_count):
        metrics_with_frequency.append(all_targets_metric)

# Save metrics to disk
metrics_results_path = Path(metrics_path,"metrics.jsonl")
with jsonlines.open(metrics_results_path, mode='w') as writer:
    for row in tqdm(metrics):
        writer.write(row)

metrics_with_frequency_results_path = Path(metrics_path,"metrics_with_frequency.jsonl")
with jsonlines.open(metrics_with_frequency_results_path, mode='w') as writer:
    for row in tqdm(metrics_with_frequency):
        writer.write(row)


# Aggregate metrics for each algorithm for comparison
# 1) Simple mean as the aggregation function
pooled_metrics = []
metrics_df = pd.DataFrame.from_records(metrics)
for col in metrics_df.columns:
    if '|' in col:
        model, metric = col.split('|')
        value = metrics_df[col].mean()
        pooled_metrics.append({
            'Model': model,
            'Metric': metric,
            'Value': value
        })

# Save metrics to disk
metrics_agg_simple_mean_path = Path(metrics_path,"metrics_agg_simple_mean.jsonl")
with jsonlines.open(metrics_agg_simple_mean_path, mode='w') as writer:
    for row in tqdm(pooled_metrics):
        writer.write(row)

df = pd.DataFrame(pooled_metrics)

# The following function calls generate the graphs and save them to disk
# Plot all metrics aggregated with simple mean in lineplot
plot_metrics(
    df,
    title = 'Comparison of Models over P@k - Aggregation with simple mean',
    filename = 'metrics_P@k_mean_lineplot',
    type='line',
    target_metrics=['P@1', 'P@5', 'P@10']
)

plot_metrics(
    df,
    title = 'Comparison of Models over AP@k - Aggregation with simple mean',
    filename = 'metrics_AP@k_mean_lineplot',
    type='line',
    target_metrics=['AP@1', 'AP@5', 'AP@10']
)

# Plot all metrics aggregated with simple mean in barplot
plot_metrics(
    df,
    title = 'Comparison of Models over P@k - Aggregation with simple mean',
    filename = 'metrics_P@k_mean_barplot',
    type='bar',
    target_metrics=['P@1', 'P@5', 'P@10']
)

plot_metrics(
    df,
    title = 'Comparison of Models over AP@k - Aggregation with simple mean',
    filename = 'metrics_AP@k_mean_barplot',
    type='bar',
    target_metrics=['AP@1', 'AP@5', 'AP@10']
)


# 2) Weighted mean as the aggregation function
pooled_metrics = []
metrics_df = pd.DataFrame.from_records(metrics)
for col in metrics_df.columns:
    if '|' in col:
        model, metric = col.split('|')
        sum_of_products = (metrics_df[col]*metrics_df['value_count']).sum()
        sum_of_counts = metrics_df['value_count'].sum()
        weighted_mean = sum_of_products / sum_of_counts
        pooled_metrics.append({
            'Model': model,
            'Metric': metric,
            'Value': weighted_mean
        })
# Save metrics to disk
metrics_agg_weighted_mean_path = Path(metrics_path,"metrics_agg_weighted_mean.jsonl")
with jsonlines.open(metrics_agg_weighted_mean_path, mode='w') as writer:
    for row in tqdm(pooled_metrics):
        writer.write(row)

df = pd.DataFrame(pooled_metrics)


# Plot all metrics aggregated with weighted mean in lineplot
plot_metrics(
    df,
    title = 'Comparison of Models over P@k - Aggregation with weighted mean',
    filename = 'metrics_P@k_weighted_mean_lineplot',
    type='line',
    target_metrics=['P@1', 'P@5', 'P@10']
)

plot_metrics(
    df,
    title = 'Comparison of Models over AP@k - Aggregation with weighted mean',
    filename = 'metrics_AP@k_weighted_mean_lineplot',
    type='line',
    target_metrics=['AP@1', 'AP@5', 'AP@10']
)

plot_metrics(
    df,
    title = 'Comparison of Models over P@k - Aggregation with weighted mean',
    filename = 'metrics_P@k_weighted_mean_barplot',
    type='bar',
    target_metrics=['P@1', 'P@5', 'P@10']
)

plot_metrics(
    df,
    title = 'Comparison of Models over AP@k - Aggregation with weighted mean',
    filename = 'metrics_AP@k_weighted_mean_barplot',
    type='bar',
    target_metrics=['AP@1', 'AP@5', 'AP@10']
)


# Measure the mean and standard deviation of time to retrieve and save to disk
metrics_time_to_retrieve_path = Path(metrics_path,"metrics_time_to_retrieve.jsonl")
metrics_time_to_retrieve = []

for model in targets:
    temp_df = pd.to_timedelta(queries_results_df[f"{model}_time"])
    mean = temp_df.mean().microseconds*(10**-6) # get mean and convert to seconds
    std = temp_df.std().microseconds*(10**-6) # get std and convert to seconds
    metrics_time_to_retrieve.append({
            'Model': model,
            'Metric': "Mean",
            'Value': mean,
        })
    metrics_time_to_retrieve.append({
            'Model': model,
            'Metric': "Standard Deviation",
            'Value': std,
        })    
    
with jsonlines.open(metrics_time_to_retrieve_path, mode='w') as writer:
    for row in tqdm(metrics_time_to_retrieve):
        writer.write(row)

# Compute agreement rate between the two annotators
# Ensure annotations are preprocessed as sets
annotations_a = preprocess_labels(true_labels_df['reviewer 1'])
annotations_b = preprocess_labels(true_labels_df['reviewer 2'])

# Extract all unique labels across both reviewers
all_labels = set()
for labels in annotations_a:
    all_labels.update(labels)
for labels in annotations_b:
    all_labels.update(labels)
all_labels = sorted(all_labels)  # Ensure labels are in consistent order

# Compute Kappa using the function
kappa_result = calculate_kappa_with_jaccard(annotations_a.tolist(), annotations_b.tolist(), all_labels)

agreement_rate_path = Path(metrics_path,"agreement_rate.jsonl")

with jsonlines.open(agreement_rate_path, mode='w') as writer:
    writer.write({"kappa": kappa_result})