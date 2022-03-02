import matplotlib.pyplot as plt

ECM_TRAIN = 'ecm_train.txt'
ECM_TEST = 'ecm_test.txt'
MODEL_NAME = 'ECM'

with open(ECM_TRAIN, 'r') as f:
    ecm_train = f.readlines()

with open(ECM_TEST, 'r') as f:
    ecm_test = f.readlines()

plt.plot(ecm_train, label='train')
plt.plot(ecm_test, label='test')
plt.title(MODEL_NAME)
plt.xlabel('epoch')
plt.ylabel('ECM')
plt.legend()
plt.savefig(f'{MODEL_NAME}.png')
