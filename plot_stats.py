import matplotlib.pyplot as plt

PROBLEM = 'problem_2'
EXTRA_PARAMS = 'example'
keywords = [('acc', 'Accuracy'), ('mse', 'MSE')]

for abr, title in keywords:
    BASE_FILENAME = PROBLEM + '_' + EXTRA_PARAMS + '_' + abr
    FILENAME_TRAIN = BASE_FILENAME + '_train.txt'
    FILENAME_TEST = BASE_FILENAME + '_test.txt'
    MODEL_NAME = title + '_' + PROBLEM

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
    plt.savefig(f'img/{MODEL_NAME}_{EXTRA_PARAMS}.png')
