import matplotlib.pyplot as plt
import pandas as pd

data_in = './models/161205_sgd_lab/epoch_history.csv'

data = pd.read_csv(data_in)

loss = data['loss']
val_loss = data['val_loss']

plt.figure()
plt.plot(loss, label='loss', lw=3, color='b')
plt.plot(val_loss, label='val_loss', lw=3, color='r')
plt.legend()
plt.show()
