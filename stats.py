import pandas as pd
import numpy as np
import jsonlines
from pathlib import Path
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency

### NON WEIGHTED SAMPLE STATISTICAL ANALYSIS ###
# Load computed metrics for the non weighted sample 
data = []
with jsonlines.open(Path('results','metrics','metrics.jsonl')) as reader:
    for obj in reader:
        data.append(obj)
df = pd.DataFrame.from_records(data)

# Select relevant columns
relevant_cols = [col for col in df.columns if ("value_count" not in col) and ("has_relevant_items" not in col)]

# Create relevant directories if they do not exist yet
path = Path(f"results","metrics","parametric_analysis")
if not path.exists(): path.mkdir(parents=True)

# Select relevant metrics
metrics = ["P@1","P@5","P@10","AP@1","AP@5","AP@10"]

# Perform ANOVA One-way test and save results to disk
with jsonlines.open(Path('results','metrics','parametric_analysis','anova_test.jsonl'), mode='w') as writer:
    for metric in metrics:
        cols = [col for col in relevant_cols if col.split('|')[1] == metric]
        f_statistic, p_value = stats.f_oneway(*[df[col].values for col in cols])

        groups = [df[col].values for col in cols]

        # Calculate total sum of squares (SS_total)
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        SS_total = np.sum((all_data - grand_mean) ** 2)

        # Calculate between-group sum of squares (SS_between)
        group_means = [np.mean(group) for group in groups]
        group_sizes = [len(group) for group in groups]
        SS_between = sum(size * (mean - grand_mean) ** 2 for size, mean in zip(group_sizes, group_means))

        # Compute Partial Eta-Squared
        eta_squared = SS_between / SS_total

        writer.write({
            "Metric": metric,
            "F-Statistic": f_statistic,
            "p-value": p_value,
            "eta-squared": eta_squared,
        })

        values = []        
        groups = []
        for col in cols:
            groups += [col]*len(df[col])
            values += list(df[col].values)

        # Perform Tukey's HSD Post Hoc Test
        tukey = pairwise_tukeyhsd(endog=values, groups=groups, alpha=0.05)

        tukey_results_directory = Path(f"results\metrics\\parametric_analysis\\post_hoc_analysis")
        if not tukey_results_directory.exists(): tukey_results_directory.mkdir(parents=True)

        with open(tukey_results_directory/ f"{metric}.csv", mode='w', encoding='utf-8') as file:
            file.write(tukey.summary().as_csv())

# Perform Chi-squared test and save results to disk
relevant_cols = [col for col in df.columns if ("has_relevant_items" in col)]

with jsonlines.open(Path('results','metrics','relevant_results_count.jsonl'), mode='w') as writer:
    models_list = []
    data = []
    for col in relevant_cols:
        model, metric = col.split('|')
        models_list.append(model)
        data.append([
            int(df[col].sum()), # n of results with at least one relevant
            int(len(df[col]) - df[col].sum()), # n of results without a relevant one
        ])

        writer.write({
            "Model": model,
            "relevant-results-present": int(df[col].sum()),
            "relevant-results-absent": int(len(df[col]) - df[col].sum()),
        })

chi2, p, dof, expected = chi2_contingency(data)

with jsonlines.open(Path('results','metrics','chi_squared_results.jsonl'), mode='w') as writer:
    writer.write({
            "sample": "Not weighted",
            "chi_square": chi2,
            "p-value": p,
            "degrees_of_fredom": dof,
        })
    

### WEIGHTED SAMPLE STATISTICAL ANALYSIS ###
# Load computed metrics for the weighted sample
data = []
with jsonlines.open(Path('results','metrics','metrics_with_frequency.jsonl')) as reader:
    for obj in reader:
        data.append(obj)
df = pd.DataFrame.from_records(data)

# Select relevant columns
relevant_cols = [col for col in df.columns if ("value_count" not in col) and ("has_relevant_items" not in col)]

# Select relevant metrics
metrics = ["P@1","P@5","P@10","AP@1","AP@5","AP@10"]

# Perform ANOVA One-way test and save results to disk
with jsonlines.open(Path('results','metrics','parametric_analysis','anova_test_weighted_sample.jsonl'), mode='w') as writer:
    for metric in metrics:
        cols = [col for col in relevant_cols if col.split('|')[1] == metric]
        f_statistic, p_value = stats.f_oneway(*[df[col].values for col in cols])

        groups = [df[col].values for col in cols]

        # Calculate total sum of squares (SS_total)
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        SS_total = np.sum((all_data - grand_mean) ** 2)

        # Calculate between-group sum of squares (SS_between)
        group_means = [np.mean(group) for group in groups]
        group_sizes = [len(group) for group in groups]
        SS_between = sum(size * (mean - grand_mean) ** 2 for size, mean in zip(group_sizes, group_means))

        # Compute Partial Eta-Squared
        eta_squared = SS_between / SS_total

        writer.write({
            "Metric": metric,
            "F-Statistic": f_statistic,
            "p-value": p_value,
            "eta-squared": eta_squared,
        })

        values = []        
        groups = []
        for col in cols:
            groups += [col]*len(df[col])
            values += list(df[col].values)

        # Perform Tukey's HSD Post Hoc Test
        tukey = pairwise_tukeyhsd(endog=values, groups=groups, alpha=0.05)
        
        tukey_results_directory = Path(f'results','metrics','parametric_analysis','post_hoc_analysis_weighted_sample')
        if not tukey_results_directory.exists(): tukey_results_directory.mkdir(parents=True)

        with open(tukey_results_directory/ f"{metric}.csv", mode='w', encoding='utf-8') as file:
            file.write(tukey.summary().as_csv())

# Select relevant columns
relevant_cols = [col for col in df.columns if ("has_relevant_items" in col)]

# Compute Chi-squared test and save results to disk
with jsonlines.open(Path('results','metrics','relevant_results_count_weighted.jsonl'), mode='w') as writer:
    models_list = []
    data = []
    for col in relevant_cols:
        model, metric = col.split('|')
        models_list.append(model)
        data.append([
            int(df[col].sum()), # n of results with at least one relevant
            int(len(df[col]) - df[col].sum()), # n of results without a relevant one
        ])

        writer.write({
            "Model": model,
            "relevant-results-present": int(df[col].sum()),
            "relevant-results-absent": int(len(df[col]) - df[col].sum()),
        })

chi2, p, dof, expected = chi2_contingency(data)

with jsonlines.open(Path('results','metrics','chi_squared_results.jsonl'), mode='a') as writer:
    writer.write({
            "sample": "Weighted",
            "chi_square": chi2,
            "p-value": p,
            "degrees_of_fredom": dof,
        })