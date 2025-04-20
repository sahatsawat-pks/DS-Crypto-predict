import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data (replace with your actual file path)
df = pd.read_csv('BTC_USD_model_comparison.csv')

# Set the style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Create a figure for MAPE analysis
plt.figure(figsize=(18, 14))

# 1. Top 15 Models by MAPE (lower is better)
plt.subplot(2, 2, 1)
top_mape = df.sort_values('mape', ascending=True).head(15)
bars = sns.barplot(x='mape', y='model', hue='features', data=top_mape, palette='viridis')
plt.title('Top 15 Models by MAPE', fontsize=16)
plt.xlabel('MAPE (lower is better)', fontsize=12)
plt.ylabel('Model Type', fontsize=12)

# Add sequence length as text annotation
for i, row in enumerate(top_mape.itertuples()):
    plt.text(0.0003, i, f"seq={row.seq_length}, scaler={row.scaler}", va='center', fontsize=8)

# 2. MAPE vs R2 scatter plot
plt.subplot(2, 2, 2)
scatter = sns.scatterplot(x='mape', y='r2', hue='model', size='feature_count', 
                         data=df, palette='tab10', sizes=(50, 200), alpha=0.7)
plt.title('MAPE vs R2 by Model Type', fontsize=16)
plt.xlabel('MAPE (lower is better)', fontsize=12)
plt.ylabel('R2 Score (higher is better)', fontsize=12)

# 3. MAPE by Model Type
plt.subplot(2, 2, 3)
violin = sns.violinplot(x='model', y='mape', data=df, palette='Set3')
plt.title('MAPE Distribution by Model Type', fontsize=16)
plt.xlabel('Model Type', fontsize=12)
plt.ylabel('MAPE (lower is better)', fontsize=12)
plt.xticks(rotation=45)

# 4. MAPE by Scaler Type
plt.subplot(2, 2, 4)
box = sns.boxplot(x='scaler', y='mape', hue='model', data=df, palette='Set2')
plt.title('MAPE by Scaler and Model Type', fontsize=16)
plt.xlabel('Scaler Type', fontsize=12)
plt.ylabel('MAPE (lower is better)', fontsize=12)
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('mape_analysis.png', dpi=300)

# 5. MAPE vs Sequence Length
plt.figure(figsize=(12, 8))
line = sns.lineplot(x='seq_length', y='mape', hue='model', 
                   data=df, marker='o', palette='tab10')
plt.title('MAPE by Sequence Length and Model Type', fontsize=16)
plt.xlabel('Sequence Length', fontsize=12)
plt.ylabel('MAPE (lower is better)', fontsize=12)
plt.grid(True)
plt.savefig('mape_by_sequence_length.png', dpi=300)

# 6. MAPE vs Training Time
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(x='training_time', y='mape', hue='model', 
                         size='feature_count', data=df, palette='tab10', 
                         sizes=(50, 200), alpha=0.7)
plt.title('Training Time vs MAPE by Model Type', fontsize=16)
plt.xlabel('Training Time (seconds)', fontsize=12)
plt.ylabel('MAPE (lower is better)', fontsize=12)
plt.savefig('training_time_vs_mape.png', dpi=300)

# 7. Heatmap of MAPE by Model and Feature Set
plt.figure(figsize=(14, 10))
heatmap_data = df.pivot_table(index='model', columns='features', values='mape', aggfunc='mean')
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu_r', fmt='.4f')  # _r reverses colormap so darker is better
plt.title('Average MAPE by Model and Feature Set', fontsize=16)
plt.tight_layout()
plt.savefig('mape_heatmap.png', dpi=300)

# 8. Combined metric plot for top models (including MAPE)
# Get top 5 models by MAPE
top_mape_models = df.sort_values('mape', ascending=True).head(5)

# Create identifiers for each model configuration
top_mape_models['model_id'] = top_mape_models.apply(
    lambda x: f"{x['model']} (seq={int(x['seq_length'])}, {x['features']}, {x['scaler']})", axis=1)

# Plot side-by-side MAPE and R2 for top models
plt.figure(figsize=(16, 8))

# MAPE subplot
plt.subplot(1, 2, 1)
sns.barplot(y='model_id', x='mape', data=top_mape_models, palette='Blues_d')
plt.title('Top 5 Models by MAPE', fontsize=16)
plt.xlabel('MAPE (lower is better)', fontsize=12)
plt.ylabel('Model Configuration', fontsize=12)

# R2 subplot
plt.subplot(1, 2, 2)
sns.barplot(y='model_id', x='r2', data=top_mape_models, palette='Greens_d')
plt.title('R2 Score of Top MAPE Models', fontsize=16)
plt.xlabel('R2 Score (higher is better)', fontsize=12)
plt.ylabel('Model Configuration', fontsize=12)

plt.tight_layout()
plt.savefig('top_mape_models.png', dpi=300)

# 9. Training Time vs Performance
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(x='training_time', y='rmse', hue='model', 
                         size='feature_count', data=df, palette='tab10', 
                         sizes=(50, 200), alpha=0.7)
plt.title('Training Time vs RMSE by Model Type', fontsize=16)
plt.xlabel('Training Time (seconds)', fontsize=12)
plt.ylabel('RMSE (lower is better)', fontsize=12)
plt.savefig('training_time_vs_performance.png', dpi=300)

# 10. Combined metric plot for top models (including RMSE)
# Get top 5 models by RMSE
top_rmse_models = df.sort_values('rmse', ascending=True).head(5)

# Create identifiers for each model configuration
top_rmse_models['model_id'] = top_rmse_models.apply(
    lambda x: f"{x['model']} (seq={int(x['seq_length'])}, {x['features']}, {x['scaler']})", axis=1)

# Plot side-by-side RMSE and R2 for top models
plt.figure(figsize=(16, 8))

# RMSE subplot
plt.subplot(1, 2, 1)
sns.barplot(y='model_id', x='rmse', data=top_rmse_models, palette='Blues_d')
plt.title('Top 5 Models by RMSE', fontsize=16)
plt.xlabel('RMSE (lower is better)', fontsize=12)
plt.ylabel('Model Configuration', fontsize=12)

# R2 subplot
plt.subplot(1, 2, 2)
sns.barplot(y='model_id', x='r2', data=top_rmse_models, palette='Greens_d')
plt.title('R2 Score of Top RMSE Models', fontsize=16)
plt.xlabel('R2 Score (higher is better)', fontsize=12)
plt.ylabel('Model Configuration', fontsize=12)

plt.tight_layout()
plt.savefig('top_rmse_models.png', dpi=300)

# Set the style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Create a figure for MAPE analysis
plt.figure(figsize=(18, 14))

# 11. Top 10 Models by R2 Score
ax1 = plt.subplot(2, 2, 1)
top_r2 = df.sort_values('r2', ascending=False).head(10)
bars = sns.barplot(x='r2', y='model', hue='features', data=top_r2, palette='viridis', ax=ax1)
ax1.set_title('Top 10 Models by R2 Score', fontsize=16)
ax1.set_xlabel('R2 Score (higher is better)', fontsize=12)
ax1.set_ylabel('Model Type', fontsize=12)

# Add sequence length as text annotation
for i, row in enumerate(top_r2.itertuples()):
    ax1.text(0.02, i, f"seq={row.seq_length}, scaler={row.scaler}", va='center', fontsize=8)

# 12. RMSE vs R2 scatter plot colored by model type
ax2 = plt.subplot(2, 2, 2)
scatter = sns.scatterplot(x='rmse', y='r2', hue='model', size='feature_count', 
                         data=df, palette='tab10', sizes=(50, 200), alpha=0.7, ax=ax2)
ax2.set_title('RMSE vs R2 by Model Type', fontsize=16)
ax2.set_xlabel('RMSE (lower is better)', fontsize=12)
ax2.set_ylabel('R2 Score (higher is better)', fontsize=12)

# 13. Model performance by sequence length
ax3 = plt.subplot(2, 2, 3)
box = sns.boxplot(x='seq_length', y='rmse', hue='model', data=df, palette='Set3', ax=ax3)
ax3.set_title('RMSE by Sequence Length and Model Type', fontsize=16)
ax3.set_xlabel('Sequence Length', fontsize=12)
ax3.set_ylabel('RMSE (lower is better)', fontsize=12)

# 14. Effect of Scaler Choice on Performance
ax4 = plt.subplot(2, 2, 4)
heatmap_data = df.pivot_table(index='model', columns='scaler', values='r2', aggfunc='mean')
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.3f', ax=ax4)
ax4.set_title('Average R2 Score by Model and Scaler', fontsize=16)

plt.tight_layout()
plt.savefig('model_performance_analysis.png', dpi=300)

print("RMSE visualizations created successfully!")