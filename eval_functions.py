import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib import cm
import numpy as np

def precision_at_k(true_labels, k):
    """
    Calculate the precision at k for a list of true labels.
    
    Parameters:
    - true_labels: list of int
        A list containing the true labels of the items (1 if relevant, 0 otherwise).
    - k: int
        The position up to which to calculate precision.

    Returns:
    - float
        The precision at k.
    """
    if k == 0:
        return 0.0
    
    # Take the first k items from the true_labels list
    top_k_labels = true_labels[:k]
    
    # Count the number of relevant items in the top k positions
    relevant_count = sum(top_k_labels)
    
    # Calculate precision at k
    precision = relevant_count / k
    
    return precision


def average_precision_at_k(true_labels, k):
    """
    Calculate the average precision at k for a list of true labels.
    
    Parameters:
    - true_labels: list of int
        A list containing the true labels of the items (1 if relevant, 0 otherwise).
    - k: int
        The position up to which to calculate average precision.
        
    Returns:
    - float
        The average precision at k.
    """
    if k == 0:
        return 0.0

    # Initialize variables
    num_relevant = 0
    sum_precision = 0.0
    
    # Loop through the list up to the kth position
    for i in range(min(k, len(true_labels))):
        is_relevant = true_labels[i] == 1
        
        if is_relevant:
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            sum_precision += precision_at_i
            
    return sum_precision / min(k, num_relevant) if num_relevant > 0 else 0.0


def plot_metrics(
    df: pd.DataFrame,
    title: str,
    filename: str,
    type: str,
    target_metrics: list[str] = None,
):
    """
    Generate and save a plot (line or bar) for the given metrics.

    Parameters:
    - df (pd.DataFrame): DataFrame containing data to plot. Should have 'Metric', 'Value', and 'Model' columns.
    - title (str): Title of the plot.
    - filename (str): Name of the file to save the plot (without extension).
    - plot_type (str): Type of plot ('line' or 'bar').
    - target_metrics (list[str], optional): List of metrics to filter and plot. Defaults to None.

    Raises:
    - AssertionError: If plot_type is not 'line' or 'bar'.
    """
    # Ensure the plot type is valid
    assert type in ['line', 'bar'], 'Plot type should be "line" or "bar"'

    # Filter the DataFrame if target_metrics is specified
    if target_metrics:
        df = df[df['Metric'].isin(target_metrics)]

    if type == 'line':
        # Create a line plot
        plt.figure(figsize=(12, 8))

        colormap = cm.get_cmap('magma')
        colors = [colormap(i / len(df['Model'].unique())) for i in range(len(df['Model'].unique()))]

        for idx, model in enumerate(df['Model'].unique()):
            # Filter data for the current model
            subset = df[df['Model'] == model]

            # Plot metrics for the model
            plt.plot(
                subset['Metric'],
                subset['Value'],
                label=model,
                marker='o',
                color=colors[idx]
            )

        # Add plot details
        plt.title(title)
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.ylim([0, 1])
        plt.grid(True)

        # Add legend to the right of the plot
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Save the plot to disk
        plt.savefig(Path(f'results/graphs/{filename}.png'), bbox_inches='tight')

    elif type == 'bar':
        # Create a bar plot
        plt.figure(figsize=(15, 8))

        sns.barplot(x="Metric", y="Value", hue="Model", data=df, palette="magma")

        # Add plot details
        plt.title(title)
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.ylim([0, 1])
        plt.grid(axis='y')

        # Add legend to the right of the plot
        plt.legend(title='Model', loc='center left', bbox_to_anchor=(1, 0.5))

        # Save the plot to disk
        plt.savefig(Path(f'results/graphs/{filename}.png'), bbox_inches='tight')

    # Close the plot to free memory
    plt.close()


def proportional_agreement(row_codes_1, row_codes_2):
    """
    Calculate the proportion of agreement for two sets of codes.

    Parameters:
        row_codes_1 (set): Codes from reviewer 1.
        row_codes_2 (set): Codes from reviewer 2.

    Returns:
        float: Proportion of agreement (0 to 1).
    """
    if not row_codes_1 and not row_codes_2:  # Both are empty
        return 1.0  # Perfect agreement when both are empty

    if not row_codes_1 or not row_codes_2:  # One is empty
        return 0.0  # No agreement if one is empty and the other is not

    # Calculate the proportion of agreement
    intersection = len(row_codes_1 & row_codes_2)
    union = len(row_codes_1 | row_codes_2)
    return intersection / union if union > 0 else 0

def preprocess_labels(column):
    """
    Preprocess a column of multi-label strings into sets of labels.

    Parameters:
        column (pd.Series): A pandas Series containing multi-label strings (e.g., 'A|B|C').

    Returns:
        pd.Series: A pandas Series where each entry is a set of labels.
    """
    return column.fillna("").apply(lambda x: set(x.split('|')) if x else set())

def calculate_kappa_with_jaccard(annotations_a, annotations_b, labels):
    """
    Calculate Cohen's Kappa statistic for multi-label data using Jaccard Index for partial agreement.

    Parameters:
        annotations_a (list of sets): List of sets containing labels assigned by annotator A.
        annotations_b (list of sets): List of sets containing labels assigned by annotator B.
        labels (list): List of all possible labels in the dataset.

    Returns:
        float: Cohen's Kappa statistic.
    """
    # Number of items (rows in the dataset)
    num_items = len(annotations_a)

    # Compute Jaccard Index for each row
    jaccard_scores = []
    for labels_a, labels_b in zip(annotations_a, annotations_b):
        if not labels_a and not labels_b:
            # If both sets are empty, treat as perfect agreement
            jaccard_index = 1.0
        else:
            # Calculate Jaccard Index for partial agreement
            intersection = len(labels_a & labels_b)
            union = len(labels_a | labels_b)
            jaccard_index = intersection / union if union > 0 else 0.0
        jaccard_scores.append(jaccard_index)

    # Observed agreement (p(A)) is the average Jaccard Index
    observed_agreement = np.mean(jaccard_scores)

    # Compute label frequencies for expected agreement (p(E))
    label_counts_a = {label: sum(1 for row in annotations_a if label in row) for label in labels}
    label_counts_b = {label: sum(1 for row in annotations_b if label in row) for label in labels}

    # Normalize frequencies by the number of items to compute probabilities
    freq_a = np.array([label_counts_a[label] / num_items for label in labels])
    freq_b = np.array([label_counts_b[label] / num_items for label in labels])

    # Expected agreement (p(E)) using the probabilities of label overlaps
    expected_agreement = np.sum(freq_a * freq_b)

    # Compute Cohen's Kappa
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement) if (1 - expected_agreement) > 0 else 0.0

    return kappa