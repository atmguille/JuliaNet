import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

for tasa in [0.1, 0.04, 0.5]:
    MODEL_NAME = f'ECM_adaline_tasa_{tasa}'

    with open(f'.\Adaline\\tasas\\ECM_{tasa}_train.txt') as f:
        ecm_train = list(map(float, f.readlines()))

    with open(f'.\Adaline\\tasas\\ECM_{tasa}_test.txt') as f:
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

