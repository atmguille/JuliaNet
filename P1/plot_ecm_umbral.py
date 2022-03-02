import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

for umbral in [0.001, 0.2, 1.0]:
    MODEL_NAME = f'ECM_perceptron_umbral_{umbral}'

    with open(f'.\Perceptron\\umbrales\\ECM_{umbral}_train.txt') as f:
        ecm_train = list(map(float, f.readlines()))

    with open(f'.\Perceptron\\umbrales\\ECM_{umbral}_test.txt') as f:
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

