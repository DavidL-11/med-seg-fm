import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

result_file = "neopolyp_results.csv"
result_path = os.path.join(os.path.dirname(__file__), result_file)

df = pd.read_csv(result_path)

# Melt the DataFrame to have a long format for plotting
new_df = df.reset_index().melt(id_vars=['Image Name', 'Model'],
                                 value_vars=['DSC', 'NSD', 'IoU'],
                                 var_name='Metric', value_name='Score')
sns.boxplot(x='Metric', y='Score', data=new_df, hue='Model', palette='Set2')

# Print the mean dice score for each model
mean_scores = new_df.groupby(['Model', 'Metric'])['Score'].mean().reset_index()

for model in mean_scores['Model'].unique():
    model_scores = mean_scores[mean_scores['Model'] == model]
    print(f"{model} Mean Scores:")
    for _, row in model_scores.iterrows():
        print(f"  {row['Metric']}: {row['Score']:.4f}")

plt.title('Neopolyp Results')
plt.xticks(rotation=45)
plt.savefig("neopolyp_results.png")
# Set legend bottom left
plt.legend(loc='lower left', bbox_to_anchor=(0, 0), title='Model')
plt.show()

