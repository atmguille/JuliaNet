import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

problem_name = "Matriz_problema_real4_0.01_5_500_norm"

matrix_train = [[315, 2],
                [6, 166]]

matrix_test = [[133, 4],
               [4, 69]]

df_cm = pd.DataFrame(matrix_train, ['pred_'+str(i) for i in range(len(matrix_train))], ['real_'+str(i) for i in range(len(matrix_train))])
plt.figure(figsize=(5,5))
#sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True,fmt='g',cmap='Blues')#, annot_kws={"size": 16})  # font size
plt.savefig(problem_name + '_train.png')

df_cm = pd.DataFrame(matrix_test, ['pred_'+str(i) for i in range(len(matrix_test))], ['real_'+str(i) for i in range(len(matrix_test))])
plt.figure(figsize=(5,5))
#sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True,fmt='g',cmap='Blues')#, annot_kws={"size": 16})  # font size
plt.savefig(problem_name + '_test.png')
