import matplotlib.pyplot as plt

ECM_TRAIN = 'ecm_train_perceptron_prob1.txt'
ECM_TEST = 'ecm_test_perceptron_prob1.txt'
MODEL_NAME = 'ECM_perceptron_problema_real1'

with open(ECM_TRAIN, 'r') as f:
    ecm_train = list(map(float, f.readlines()))

with open(ECM_TEST, 'r') as f:
    ecm_test = list(map(float, f.readlines()))

plt.plot(ecm_train, label='train')
plt.plot(ecm_test, label='test')
plt.title(MODEL_NAME)
plt.xlabel('epoch')
plt.ylabel('ECM')
plt.legend()
plt.savefig(f'{MODEL_NAME}.png')
