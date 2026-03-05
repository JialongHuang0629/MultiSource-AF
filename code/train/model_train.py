from core.utils.dataset import *
import time
from sklearn.metrics import accuracy_score
import torch.nn as nn
import parameter
from core.ours import MultiSourceClassifier
from core.HybridSN import HybridSNMulti
from core.SpectralFormer import SpectralFormerMulti
from core.CNN import CNNSpectralSAR
from core.CapsuleNet import FastCapsNetMulti


parameter._init()

def count_model_params(model):
    """
    返回模型的参数量
    :param model: pytorch模型
    :return: 模型的参数量
    """
    # 获取模型所有参数的数量
    total_params = sum(p.numel() for p in model.parameters())
    # 获取模型可训练参数的数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': total_params, 'trainable_params': trainable_params}
def Train(epochs, lr, model_name, train_loader, test_loader, out_features, model_savepath, log_path):
    device = torch.device("cuda:0")
    if model_name == 'ours':
        net = MultiSourceClassifier(num_classes=out_features).to(device)
    elif model_name=='HybridSN':
        net=HybridSNMulti(hsi_channels=30, sar_channels=1, patch_size=10, num_classes=out_features).to(device)
    elif model_name=='SpectralFormer':
        net=SpectralFormerMulti(hsi_channels=30, sar_channels=1, patch_size=10, num_classes=out_features).to(device)
    elif model_name=='CNN':
        net=CNNSpectralSAR(hsi_channels=30, sar_channels=4, patch_size=10, num_classes=out_features).to(device)
    elif model_name=='CapsuleNet':
        net=FastCapsNetMulti(hsi_channels=30, sar_channels=4,num_capsules=2, capsule_dim=8,class_dim=16, num_classes=out_features).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=lr,
        weight_decay=0.05
    )
    # 计算模型参数量
    model_params = count_model_params(net)
    getLog(log_path, f"Model Total Parameters: {model_params['total_params']}, Trainable Parameters: {model_params['trainable_params']}")

    max_acc = 0
    sum_time = 0
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    current_time_log = 'start time: {}'.format(current_time)
    getLog(log_path, parameter.get_taskInfo())
    getLog(log_path, '-------------------Started Training-------------------')
    getLog(log_path, current_time_log)
    for epoch in range(epochs):
        since = time.time()
        net.train()
        for i, (hsi, sar, tr_labels) in enumerate(train_loader):
            hsi = hsi.to(device)
            sar = sar.to(device)
            tr_labels = tr_labels.to(device)
            optimizer.zero_grad()
            outputs = net(hsi, sar)
            loss = criterion(outputs, tr_labels)
            loss.backward()
            optimizer.step()
        net.eval()
        count = 0
        for hsi, sar, gtlabels in test_loader:
            hsi = hsi.to(device)
            sar = sar.to(device)
            outputs = net(hsi, sar)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test =  outputs
                gty = gtlabels
                count = 1
            else:
                y_pred_test = np.concatenate( (y_pred_test, outputs) )
                gty = np.concatenate( (gty, gtlabels) )
        acc1 = accuracy_score(gty, y_pred_test)
        # model_path = f"{model_savepath}_epoch_{epoch}.pth"
        # torch.save(net, model_path)
        if acc1 > max_acc:
            torch.save(net, model_savepath)
            max_acc = acc1
        time_elapsed = time.time() - since
        sum_time += time_elapsed
        rest_time = (sum_time / (epoch + 1)) * (epochs - epoch - 1)
        currentTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        log = currentTime + ' [Epoch: %d] [%.0fs, %.0fh %.0fm %.0fs] [current loss: %.4f] acc: %.4f' %(epoch + 1, time_elapsed, (rest_time // 60) // 60, (rest_time // 60) % 60, rest_time % 60, loss.item(), acc1)
        print(log)
        getLog(log_path, log)
    print('max_acc: %.4f' %(max_acc))  
    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    finish_time_log = 'finish time: {} '.format(finish_time)
    mac_acc_log = 'max_acc: {} '.format(max_acc)
    getLog(log_path, mac_acc_log)
    getLog(log_path, finish_time_log)
    getLog(log_path, '-------------------Finished Training-------------------')

def getLog(log_path, str):
    with open(log_path, 'a+') as log:
        log.write('{}'.format(str))
        log.write('\n')

def main_train():
    model_name=parameter.get_value('model_name')
    datasetType = parameter.get_value('datasetType')
    channels = parameter.get_value('channels')
    windowSize = parameter.get_value('windowSize')
    out_features = parameter.get_value('out_features')
    lr = parameter.get_value('lr')
    epoch_nums = parameter.get_value('epoch_nums')
    batch_size = parameter.get_value('batch_size')
    num_workers = parameter.get_value('num_workers')
    random_seed = parameter.get_value('random_seed')
    model_savepath = parameter.get_value('model_savepath')
    log_path = parameter.get_value('log_path')
    train_loader, test_loader, trntst_loader, all_loader = fetchData(datasetType, channels, windowSize, batch_size, num_workers)
    set_random_seed(random_seed)
    Train(epoch_nums, lr, model_name,  train_loader, test_loader, out_features, model_savepath, log_path)