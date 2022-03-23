import matplotlib.pyplot as plt

PROBLEMA = 'problema_real6'
PARAMS = '0.1_10-10_1000_norm'
keywords = [('acc', 'Accuracy'), ('ecm', 'ECM')]

for abr, title in keywords:
    BASE_FILENAME = 'memo/' + PROBLEMA + '/' + PROBLEMA + '_' + PARAMS + '_' + abr
    FILENAME_TRAIN = BASE_FILENAME + '_train.txt'
    FILENAME_TEST = BASE_FILENAME + '_test.txt'
    MODEL_NAME = title + '_' + PROBLEMA

    with open(FILENAME_TRAIN, 'r') as f:
        train = list(map(float, f.readlines()))

    with open(FILENAME_TEST, 'r') as f:
        test = list(map(float, f.readlines()))

    plt.figure()
    plt.plot(train, label='train')
    plt.plot(test, label='test')
    plt.title(MODEL_NAME)
    plt.xlabel('epoch')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(f'{MODEL_NAME}_{PARAMS}.png')
