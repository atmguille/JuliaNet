import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

for epoca in [10, 50, 100]:
    MODEL_NAME = f'ECM_adaline_epoca_{epoca}'

    with open(f'.\Adaline\\epocas\\ECM_{epoca}_train.txt') as f:
        ecm_train = list(map(float, f.readlines()))

    with open(f'.\Adaline\\epocas\\ECM_{epoca}_test.txt') as f:
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

