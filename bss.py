import matplotlib.pyplot as plt
import generate_data as gd
import pca

# number of components to retain
no_comp = 1
pca_obj = pca.PCA(gd.X)
X_lower = pca_obj.transform_lower_dim(no_comp)

plt.figure()

models = [gd.X, gd.S, X_lower]
names = ['Observations (mixed signal)',
         'True Sources',
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()
