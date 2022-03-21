import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

problem_name = "problema_1"

matrix = [[13,1,1,0,2,0],
         [3,9,6,0,1,0],
         [0,0,16,2,0,0],
         [0,0,0,13,0,0],
         [0,0,0,0,15,0],
         [0,0,1,0,0,15]]

df_cm = pd.DataFrame(matrix, range(len(matrix)), range(len(matrix)))
plt.figure(figsize=(10,7))
#sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True)#, annot_kws={"size": 16})  # font size
plt.savefig(problem_name + '.png')
