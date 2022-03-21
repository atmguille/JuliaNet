import pandas as pd
import matplotlib.pyplot as plt

PROBLEMA = 2

input_file = f"data/problema_real{PROBLEMA}.txt"

df = pd.read_csv(input_file, sep=" ", skiprows=1, header=None)

df.drop(columns=range(len(df.columns))[-2:], inplace=True)

plt.figure(figsize=(5, 5))
plt.title(f"Variables problema {PROBLEMA}")
df.boxplot(showfliers=False)
plt.savefig(f"memo/img/problema{PROBLEMA}_boxplot.png")


