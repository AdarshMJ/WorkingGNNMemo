import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def categorize_values(df, column, n_bins=3):
    """Categorize values using percentile-based bins."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(df[column], percentiles)
    labels = ['Low', 'Medium', 'High'] if n_bins == 3 else ['Low', 'High']
    return pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)

def analyze_grid_categories(results_df, output_dir, min_samples=20):
    """Analyze grid results using percentile-based categories."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Categorize values
    for col in ['homophily', 'informativeness', 'percent_memorized']:
        results_df[f'{col}_cat'] = categorize_values(results_df, col, n_bins=3)
    
    # Create summary DataFrame
    summary = []
    
    # Analyze each category combination
    for h_cat in ['Low', 'Medium', 'High']:
        for i_cat in ['Low', 'Medium', 'High']:
            subset = results_df[
                (results_df['homophily_cat'] == h_cat) & 
                (results_df['informativeness_cat'] == i_cat)
            ]
            
            if len(subset) >= min_samples:
                # Calculate correlations within this category
                spearman_h = stats.spearmanr(subset['homophily'], subset['percent_memorized'])
                spearman_i = stats.spearmanr(subset['informativeness'], subset['percent_memorized'])
                
                summary.append({
                    'Homophily_Category': h_cat,
                    'Informativeness_Category': i_cat,
                    'Num_Samples': len(subset),
                    'Avg_Memorization': subset['percent_memorized'].mean(),
                    'Std_Memorization': subset['percent_memorized'].std(),
                    'Spearman_Homophily': spearman_h.correlation,
                    'Spearman_H_PValue': spearman_h.pvalue,
                    'Spearman_Info': spearman_i.correlation,
                    'Spearman_I_PValue': spearman_i.pvalue
                })
    
    summary_df = pd.DataFrame(summary)
    
    # Create visualizations
    
    # 1. Heatmap of average memorization by category
    plt.figure(figsize=(10, 8))
    heatmap_data = pd.pivot_table(
        summary_df, 
        values='Avg_Memorization', 
        index='Homophily_Category',
        columns='Informativeness_Category'
    )
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='viridis')
    plt.title('Average Memorization Rate by Category')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_heatmap.png'))
    plt.close()
    
    # 2. Box plots for each category combination
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=results_df, 
        x='homophily_cat', 
        y='percent_memorized',
        hue='informativeness_cat'
    )
    plt.title('Memorization Distribution by Category')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_boxplots.png'))
    plt.close()
    
    # 3. Correlation plots for each category
    for h_cat in ['Low', 'Medium', 'High']:
        for i_cat in ['Low', 'Medium', 'High']:
            subset = results_df[
                (results_df['homophily_cat'] == h_cat) & 
                (results_df['informativeness_cat'] == i_cat)
            ]
            
            if len(subset) >= min_samples:
                plt.figure(figsize=(10, 5))
                
                # Plot 1: Homophily vs Memorization
                plt.subplot(1, 2, 1)
                sns.regplot(data=subset, x='homophily', y='percent_memorized')
                plt.title(f'H: {h_cat}, I: {i_cat}\nHomophily vs Memorization')
                
                # Plot 2: Informativeness vs Memorization
                plt.subplot(1, 2, 2)
                sns.regplot(data=subset, x='informativeness', y='percent_memorized')
                plt.title(f'H: {h_cat}, I: {i_cat}\nInformativeness vs Memorization')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'correlation_h{h_cat}_i{i_cat}.png'))
                plt.close()
    
    return summary_df

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=True,
                       help='Path to grid results CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save analysis results')
    parser.add_argument('--min_samples', type=int, default=20,
                       help='Minimum samples required for category analysis')
    
    args = parser.parse_args()
    
    # Load results
    results_df = pd.read_csv(args.results_path)
    
    # Perform analysis
    summary_df = analyze_grid_categories(
        results_df, 
        args.output_dir,
        args.min_samples
    )
    
    # Save summary
    summary_df.to_csv(os.path.join(args.output_dir, 'category_analysis.csv'), index=False)
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
