import sys
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt

path = r"C:\Users\pushp\PycharmProjects\Thesis\data\vaskas_features_properties_smiles_filenames.csv"
data = pd.read_csv(path)

barrier = data['barrier']
density = gaussian_kde(barrier)
eval_points = np.linspace(np.min(barrier), np.max(barrier))
print(np.min(barrier))
print(np.max(barrier))
fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

ax.plot(density.evaluate(eval_points))
plt.show()
