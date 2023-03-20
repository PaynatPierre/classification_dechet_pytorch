import torch

'''
this function is made to compute the accuracy of a result gived by your model

args:
    y_pred: tensor which contains your model's prediction
    y_pred: tensor which contains your label

return:
    metric: float which representes your loss
'''
def accuracy_metric(y_pred, y_true):

    y_pred = torch.argmax(y_pred, dim=-1)
    # y_true = torch.argmax(y_true, dim=-1)

    compare = torch.flatten(torch.eq(y_pred, y_true)).float()
    metric = torch.sum(compare)/compare.shape[0]

    return metric