from config import cfg
from model import get_vgg, get_mobilenet, get_resnet, get_resnet_pretrained
import torch
from dataloader import get_dataloader
import tqdm
from utils import accuracy_metric
import statistics
import os

'''
this function is made to manage all the training pipeline, including validation. It also save models checkpoint after each validation.

args:
    None

return:
    None

'''
def train():
    torch.manual_seed(0)

    # loading model, loss and optimizer
    model = get_resnet_pretrained()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)

    # create dataloader
    train_dataloader, validation_dataloader, test_dataloader = get_dataloader()

    # create list to deal with all results and save them
    train_list_loss = []
    train_list_acc = []
    val_list_loss = []
    val_list_acc = []
    loss_history = []
    acc_history = []

    # index of checkpoint and bath counting creation
    checkpoint_count = 0
    batch_after_last_validation = 0
    for epoch in range(cfg.TRAIN.NBR_EPOCH):
        
        # creation of index to count gradian accumulation since the last weights update
        gradiant_accumulation_count = 0

        # loop en batchs
        train_range = tqdm.tqdm(train_dataloader)
        for i, (datas, labels) in enumerate(train_range):

            # change model mode
            model.train()
            gradiant_accumulation_count += 1
            batch_after_last_validation += 1

            # moove variables place in memory on GPU RAM
            if torch.cuda.is_available():
                datas = datas.cuda()
                labels = labels.cuda()

            # make prediction
            outputs = model(datas)

            # compute loss
            loss = criterion(outputs, labels)

            # if torch.cuda.is_available():
            #     loss.cuda()

            # make gradiant retropropagation
            loss.backward()

            # condition to choose if you update model's weights or not
            if gradiant_accumulation_count >= cfg.TRAIN.GRADIANT_ACCUMULATION or i == len(train_dataloader) - 1:

                # reinitialisation of gradiant accumulation index
                gradiant_accumulation_count = 0

                # update model's weights
                optimizer.step()
                optimizer.zero_grad()

                # compute mectric
                metric = accuracy_metric(outputs, labels)

                # save loss and metric for the current batch
                train_list_loss.append(loss.item())
                train_list_acc.append(metric.item())

                # update tqdm line information
                train_range.set_description("TRAIN -> epoch: %4d || loss: %4.4f || metric: %4.4f" % (epoch, statistics.mean(train_list_loss), statistics.mean(train_list_acc)))
                train_range.refresh()

            # condition to choose if you have to do a validation or not
            if batch_after_last_validation + 1 > len(train_dataloader)/cfg.TRAIN.VALIDATION_RATIO:

                # remove gradiants computation for the validation
                with torch.no_grad():
                    batch_after_last_validation = 0

                    # validation loop
                    for j, (val_datas, val_labels) in enumerate(validation_dataloader):
                        
                        # change model mode
                        model.eval()

                        # moove variables place in memory on GPU RAM
                        if torch.cuda.is_available():
                            val_datas = val_datas.cuda()
                            val_labels = val_labels.cuda()
                        
                        # make prediction
                        outputs = model(val_datas)

                        # compute loss
                        loss = criterion(outputs, val_labels)

                        # compute mectric
                        metric = accuracy_metric(outputs, val_labels)

                        # save loss and metric for the current batch
                        val_list_loss.append(loss.item())
                        val_list_acc.append(metric.item())

                    # print validation results
                    print(" ")
                    print("VALIDATION -> epoch: %4d || loss: %4.4f || metric: %4.4f" % (epoch, statistics.mean(val_list_loss), statistics.mean(val_list_acc)))

                    # save model checkpoint
                    torch.save(model.state_dict(), os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH,'ckpt_' + str(checkpoint_count)) + "_metric_" + str(round(statistics.mean(val_list_acc),5)) + ".ckpt")
                    checkpoint_count += 1
                    print(" ")

                    # save loss and metric for the current epoch
                    loss_history.append(statistics.mean(val_list_loss))
                    acc_history.append(statistics.mean(val_list_acc))

                    with open(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, 'result_history.txt'), 'a+') as result_file:
                        result_file.write(f"checkpoint_{checkpoint_count} : loss = {statistics.mean(val_list_loss)} , metric = {statistics.mean(val_list_acc)} \n")

                    # print loss and metric history
                    print("loss history:")
                    print(loss_history)
                    print("acc history:")
                    print(acc_history)

                    # clear storage of short term result
                    train_list_loss = []
                    train_list_acc = []
                    val_list_loss = []
                    val_list_acc = []
        
        # clear storage of short term result
        train_list_loss = []
        train_list_acc = []