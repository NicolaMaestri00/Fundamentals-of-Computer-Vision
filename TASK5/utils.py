#####TUWIEN - WS2023 CV: Task3 - Scene recognition with Bag of Visual Words
from sklearn.metrics import confusion_matrix
import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os


def plot_confusion_matrix(cm, classes, acc, normalize=False, title = 'Confusion_matrix', save_path='./', cmap=plt.cm.Blues):
    plt.figure(figsize=(6,6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\nNormalized confusion matrix")
    else:
        print('\nConfusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title+' Acc:{:.2f}%'.format(acc), color = 'k')
    
    plt.setp(plt.getp(plt.colorbar().ax.axes, 'yticklabels'), color='k')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, color = 'k')
    plt.yticks(tick_marks, classes, color = 'k')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', color = 'k')
    plt.xlabel('Predicted label', color = 'k')
    plt.tight_layout()
    plt.savefig(save_path+'/'+title+'.png', facecolor='w', dpi=100)

def calculate_confusion_matrix(args, model, data_loader):
    val_predicted, val_labels = torch.tensor([]).to(args.DEVICE), torch.tensor([]).to(args.DEVICE)
    with torch.no_grad():
        correct, total, val_loss_sum = 0, 0, 0.
        for step, data in enumerate(data_loader):
            inputs, labels= data[0].to(args.DEVICE).float(), data[1].to(args.DEVICE)

            #val_predict
            model.eval()
            pred = model(inputs)
            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = 100*correct / total
            val_predicted = torch.cat((val_predicted, predicted), 0)
            val_labels = torch.cat((val_labels, labels), 0)

            print('\rEvaluating {:{}d}/{}'.format(step+1, len(str(len(data_loader))), len(data_loader)),end='')

    stacked = torch.stack((val_labels, val_predicted), dim=1).type(dtype=torch.int64)
    cmt = torch.zeros(len(pred[0]), len(pred[0]), dtype=torch.int64)

    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1
    return cmt, acc

def plot_result(args, model, loss_function, test_loader, label, save_path = './Best_acc_state_test.png'):
    test_loss, test_pred = [], []
    plt.figure(figsize=(15,16))
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            inputs, labels= data[0].to(args.DEVICE).float(), data[1].to(args.DEVICE)

            #test_predict
            model.eval()
            pred = model(inputs)
            _, predicted = torch.max(pred.data, 1)
            pred_label = label[predicted]
            test_pred.append(pred_label)
            inputs = inputs.cpu().numpy()
            plt.subplot(331+step)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(
                np.transpose(
                    np.reshape(
                        (inputs-np.min(inputs))/(np.max(inputs)-np.min(inputs)), (1, args.SIZE, args.SIZE)
                    ), (1,2,0)
                ), cmap='gray'
            )
            plt.title('predict: {}, True: {}'.format(str(pred_label), str(label[labels])), color='k')

            #test_loss
            loss = loss_function(pred, labels).item()
            test_loss.append(loss)
            plt.xlabel('Loss:'+str(loss), color='k')
        plt.savefig(save_path, facecolor='w')
        plt.show()
    return test_loss, test_pred