from model import get_vgg, get_resnet_pretrained
from config import cfg
import torch
from dataloader import get_dataloader
from utils import accuracy_metric
import statistics

'''
this function is made to evaluate a model on your test dataset

args:
    None

return:
    None
'''
def evaluate():
    # create dataloader and loss
    train_dataloader, validation_dataloader, test_dataloader = get_dataloader()
    criterion = torch.nn.CrossEntropyLoss()

    # load model and change its mode
    model = get_resnet_pretrained(cfg.EVALUATION.PRETRAINED_PATH)
    model.eval()

    # create list to collect results
    test_list_loss = []
    test_list_acc = []

    # loop on batch in test dataset
    for j, (test_datas, test_labels) in enumerate(test_dataloader):

        # moove variables place in memory on GPU RAM
        if torch.cuda.is_available():
            test_datas = test_datas.cuda()
            test_labels = test_labels.cuda()
        
        # compute result
        outputs = model(test_datas)

        # compute loss
        loss = criterion(outputs, test_labels)

        # compute mectric
        metric = accuracy_metric(outputs, test_labels)

        # save loss and metric of the current batch
        test_list_loss.append(loss.item())
        test_list_acc.append(metric.item())
    
    # print test results
    print(" ")
    print("TEST -> loss: %4.4f || metric: %4.4f" % (statistics.mean(test_list_loss), statistics.mean(test_list_acc)))
