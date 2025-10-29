import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Example data
np.random.seed(0)
df = pd.DataFrame({
    'group': np.repeat(['G1a','G1b','G1c','G1d',
                        'G2a','G2b','G2c','G2d',
                        'G3a','G3b','G3c','G3d'], 20),
    'value': np.random.randn(240)
})

levels = [('X', i) for i in ['a','b','c','d']] + \
         [('Y', i) for i in ['a','b','c','d']] + \
         [('Z', i) for i in ['a','b','c','d']]

# create composite labels
df['super'] = [x[0] for x in np.repeat(levels, 20)]
df['sub']   = [x[1] for x in np.repeat(levels, 20)]
df['label'] = df['super'] + ' – ' + df['sub']

sns.boxplot(y='label', x='value', data=df, orient='h')

# then tweak ticks manually:
yticks = plt.yticks()[0]
plt.yticks(yticks, [l.split(' – ')[1] for l in df['label'].unique()])

# Add larger group labels as text
plt.text(df['value'].min()-0.2, 2, 'X', fontweight='bold')
plt.text(df['value'].min()-0.2, 6, 'Y', fontweight='bold')
plt.text(df['value'].min()-0.2, 10, 'Z', fontweight='bold')


plt.tight_layout()
plt.show()
