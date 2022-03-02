import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

for tolerancia in [0.001, 0.02, 0.04]:
    MODEL_NAME = f'ECM_adaline_tolerancia_{tolerancia}'

    with open(f'.\Adaline\\tolerancias\\ECM_{tolerancia}_train.txt') as f:
        ecm_train = list(map(float, f.readlines()))

    with open(f'.\Adaline\\tolerancias\\ECM_{tolerancia}_test.txt') as f:
        ecm_test = list(map(float, f.readlines()))

    fig, ax = plt.subplots()
    plt.plot(ecm_train, label='Train')
    plt.plot(ecm_test, label=f'Test')
    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.title(MODEL_NAME)
    plt.xlabel('epoch')
    plt.ylabel('ECM')
    plt.legend()
    plt.savefig(f'{MODEL_NAME}.png')

    plt.clf()

