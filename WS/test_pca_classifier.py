# %%
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
from pca_classifier.pca_classifier import PcaClassifier
from pca_classifier.utils.utils import make_dataloader
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', (v1, v0), arrowprops=arrowprops)

def main():
    pc = PcaClassifier()
    # train_dataloader = make_dataloader('data/train.pkl', shuffle=False)
    valid_dataloader = make_dataloader('data/valid.pkl', shuffle=False)
    test_dataloader = make_dataloader('data/test.pkl', shuffle=False)
    pca_points = None
    y_pred = None
    for batch_idx, (x_batch, y_batch) in enumerate(valid_dataloader):
        result, trans_points = pc.predict(x_batch, fetch_points=True)
        if y_pred is None:
            y_pred = result
            pca_points = trans_points
        else:
            y_pred = np.concatenate((y_pred, result), axis=0)
            pca_points = np.concatenate((pca_points, trans_points), axis=0)
    for batch_idx, (x_batch, y_batch) in enumerate(test_dataloader):
        result, trans_points = pc.predict(x_batch, fetch_points=True)
        if y_pred is None:
            y_pred = result
            pca_points = trans_points
        else:
            y_pred = np.concatenate((y_pred, result), axis=0)
            pca_points = np.concatenate((pca_points, trans_points), axis=0)
    
    # for batch_idx, (x_batch, y_batch) in enumerate(train_dataloader):
    #     result, trans_points = pc.predict(x_batch, fetch_points=True)
    #     if y_pred is None:
    #         y_pred = result
    #         pca_points = trans_points
    #     else:
    #         y_pred = np.concatenate((y_pred, result), axis=0)
    #         pca_points = np.concatenate((pca_points, trans_points), axis=0)
    
    y_true = torch.cat((valid_dataloader.dataset[:][1],
            test_dataloader.dataset[:][1]), 0)
    
    # y_true = torch.cat((y_true,
    #         train_dataloader.dataset[:][1]), 0)

    y_true = y_true.numpy()
    print('recall is:', recall_score(y_true, y_pred))

    # here for plot pca points
    y0_idx = (y_true==0)
    y1_idx = (y_true==1)
    plt.scatter(pca_points[y0_idx,0], pca_points[y0_idx,1], c="green", s=1, alpha=0.2)
    plt.scatter(pca_points[y1_idx,0], pca_points[y1_idx,1], c="red", s=5, alpha=0.5)
    plt.scatter(pca_points[y_pred==1,0], pca_points[y_pred==1,1], c="blue", s=2)
    pca = pc.get_pca()
    coeff = pca.components_.T
    n = coeff.shape[0]
    # for i in range(n):
    #     plt.arrow(0, 0, coeff[i,0]*10, coeff[i,1]*10,color = 'r',alpha = 0.5)
        # plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
    plt.axis('equal')
    plt.show()

    


if __name__ == '__main__':
    main()
# %%
