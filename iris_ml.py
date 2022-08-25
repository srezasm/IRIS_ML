from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor

# initiate main params
iris = load_iris()
x = iris.data
y = iris.target

features = iris.feature_names
targets = iris.target_names


# prediction
sgdr = SGDRegressor(max_iter=10000)
sgdr.fit(x, y)
pred = sgdr.predict(x)

# round & positify & integer predictions
rpred = np.rint(pred)
rpred = abs(rpred)

print(y)
print(rpred)

# prediction plot
fig, axes = plt.subplots(2, 4, figsize=(12, 3), sharey=True)
for i in range(len(axes[0,:])):
    axes[0,i].scatter(x[:,i], y, c='b',label='real target')
    axes[0,i].scatter(x[:,i], pred, c='r', label = 'prediction')

# round prediction plot
for i in range(len(axes[1,:])):
    axes[1,i].scatter(x[:,i], y, c='b', label='real target')
    axes[1,i].scatter(x[:,i], rpred, c='r', marker='x', label = 'rounded prediction')
    axes[1,i].set_xlabel(features[i])

fig.suptitle(f"Real VS. Predicted Target of IRIS Dataset\nTotal Mispredictions: {np.count_nonzero(y != rpred)}")
plt.yticks(np.arange(len(targets)) ,labels=targets)
axes[0,0].legend()
axes[1,0].legend()
plt.show()
