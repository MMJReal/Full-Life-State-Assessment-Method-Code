# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch as tr
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import My_MMD
from MY_models_p import DANN_Encoder
import time
from My_readdata import read_data, read_alldata, read_data_fe
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import random
import os
import joblib
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
from MY_distance import Distance



def mmd_loss(x_src, y_src, x_tar, y_pseudo, factor):
    return My_MMD.rbf_mmd(x_src, y_src, x_tar, y_pseudo, factor)


def model_train(model, optimizer, epoch, data_src, data_tar, y_pse, factor):
    tmp_train_loss = 0
    correct = 0
    correct_t = 0
    size_t = 0
    size_SS = 0
    list_src, list_tar = list(enumerate(data_src)), list(enumerate(data_tar))
    lens_s = len(data_src)
    lens_t = len(data_tar)
    Lens = max(lens_s, lens_t)
    model.train()
    for batch_id in range(0, Lens):
        optimizer.zero_grad()
        if batch_id <= (lens_s - 1):
            _, (x_src, y_src) = list_src[batch_id]
        else:
            rand_num = random.randint(0, lens_s-1)
            _, (x_src, y_src) = list_src[rand_num]
        if batch_id <= (lens_t - 1):
            _, (x_tar, y_tar) = list_tar[batch_id]
        else:
            rand_num_t = random.randint(0, lens_t-1)
            _, (x_tar, y_tar) = list_tar[rand_num_t]
        x_src, y_src = x_src.detach().to(DEVICE), y_src.to(DEVICE)
        x_tar = x_tar.to(DEVICE)

        S_lens = len(x_src)
        T_lens = len(x_tar)

        if T_lens==1 or S_lens == 1:
            x_tar = x_src
        elif S_lens < T_lens:
            x_tar = x_tar[0:S_lens, :, :]
        elif S_lens > T_lens:
            x_src = x_src[0:T_lens, :, :]
            y_src = y_src[0:T_lens, :]

        ypred, x_src_mmd, x_tar_mmd, De_s, De_t = model(x_src, x_tar)
        loss_ce = criterion(ypred, y_src)
        loss_des = criterion_des(De_s, x_src)
        loss_mmd = mmd_loss(x_src_mmd, y_src, x_tar_mmd, y_pse[batch_id, :], factor)
        pred_pse, x_tar_mmd_t1, x_sar_mmd_t1, De_t1, De_s1 = model(x_tar, x_src,)
        selct_P = pred_pse
        Selct_T = selct_P + y_src
        loss_t = criterion_t(selct_P, Selct_T)
        loss_det = criterion_det(De_t1, x_tar)
        correct_t += torch.mean(F.cosine_similarity(pred_pse, Selct_T))
        size_t += T_lens
        size_SS += S_lens
        y_pse[batch_id, 0: len(pred_pse), :] = pred_pse.detach()
        # get training loss
        correct += torch.mean(F.cosine_similarity(ypred, y_src))
        loss = loss_ce + loss_mmd + loss_des + loss_det + loss_t

        loss.backward()
        optimizer.step()
        tmp_train_loss += loss.detach()

    tmp_train_loss /= size_SS
    tmp_train_acc = correct / size_SS
    train_loss = tmp_train_loss.cpu().detach().numpy()
    train_acc = tmp_train_acc.data

    tim = time.strftime("%H:%M:%S", time.localtime())
    res_e = '{:s}, epoch: {}/{}, train loss: {:.4f}, train simlilar: {:.4f}'.format(
        tim, epoch, N_EPOCH, tmp_train_loss, tmp_train_acc)
    print("Training Target simlilar %.2f" % float(correct_t / size_t))

    tqdm.write(res_e)
    return train_acc, train_loss, model



def smooth_row(row, window_size=5):
    return np.convolve(row, np.ones(window_size)/window_size, mode='same')


def model_test_result(model, data_tar):

    results_T = []
    results_P = []
    with tr.no_grad():
        for batch_id, (x_tar, y_tar) in enumerate(data_tar):

            x_tar, y_tar = x_tar.to(DEVICE), y_tar.to(DEVICE)
            model.eval()
            ypred, _, _, _, _ = model(x_tar, x_tar)
            pred = ypred.detach().cpu().numpy()
            pred = np.array([smooth_row(row) for row in pred])
            y_tar = y_tar.detach().cpu().numpy()
            results_T.extend(y_tar)
            results_P.extend(pred)
        results_T = pd.DataFrame(results_T)
        results_P = pd.DataFrame(results_P)

    return results_T, results_P




if __name__ == '__main__':

    XJTU_path = 'G:/'
    PHM_root_path = 'G:/'
    # judge threshold
    sigma = 6
    #
    # ------------------source path and read data------------------#
    path = XJTU_path + '/data/'
    # 
    Judge_name = 'Bearing1_1'
    Source_name = 'Bearing1_2'
    std_path_s = './initial/STD/' + Source_name + 'source_std.m'
    my_step = 1  #
    train_x, train_y = read_data(path, step=my_step, pathsnames=std_path_s)
    My_shapes = train_x.shape

    # ------------------target path  and read data------------------#
    path_t = PHM_root_path + '/data/' + Judge_name + '.mat'
    std_path_t = './initial/STD/' + Judge_name + 'target_std.m'
    train_x_t, train_y_t = read_data(path_t, step=my_step, pathsnames=std_path_t)


    # -------- save path --------
    model_path = './initial/models/' + Judge_name + '/'
    os.makedirs(model_path, exist_ok=True)
    train_save_loss_path = './initial/loss/' + Judge_name
    train_save_accuracy_path = './initial/results/'
    # ------------------para of the network------------------#

    LEARNING_RATE = 0.0001
    N_EPOCH = 1
    BATCH_SIZE = 30
    criterion = nn.MSELoss(reduction='mean').to(DEVICE)
    criterion_t = nn.MSELoss(reduction='mean').to(DEVICE)
    criterion_des = nn.MSELoss(reduction='mean').to(DEVICE)
    criterion_det = nn.MSELoss(reduction='mean').to(DEVICE)

    #   ============ Source=============

    train_X = Variable(torch.Tensor(train_x))
    train_Y = Variable(torch.Tensor(train_y))
    Traindata_S = Data.TensorDataset(train_X, train_Y)
    Traindata_S = Data.DataLoader(dataset=Traindata_S, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False,
                                      num_workers=0, drop_last=False)

    # =============Target==============
    val_X = Variable(torch.Tensor(train_x_t))
    val_Y = Variable(torch.Tensor(train_y_t))
    Valdata_Ts = Data.TensorDataset(val_X, val_Y)
    Valdata_T = Data.DataLoader(dataset=Valdata_Ts, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False,
                                    num_workers=0, drop_last=False)

    Valdata_TP = Data.DataLoader(dataset=Valdata_Ts, batch_size=1, shuffle=False, pin_memory=False,
                                num_workers=0, drop_last=False)

    # data preparation complete
    print("The all datas has been load")

    # train and test
    start = time.time()
    len_dataloader = max(len(Traindata_S), len(Valdata_T))


    # save_result
    Loss = []
    Accuracy = []

    # #=====training===============
    for train_num in range(1, 7):
        print('-' * 30)
        print('The Training number is %.2f"' % float(train_num) + '\n')
        acc, loss = {}, {}
        train_acc = []
        test_acc = []
        train_loss = []
        test_loss = []
        y_pse = tr.ones(len_dataloader, BATCH_SIZE, train_y.shape[1]).float().cuda()
        y_pse = y_pse*0.1
        mdl = DANN_Encoder(output_size=train_y.shape[1], num_layers=3, hidden_size=250, step=my_step)
        mdl = mdl.to(DEVICE)
        # optimization
        opt_Adam = optim.AdamW(mdl.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
        # # -------start-----------
        for ep in tqdm(range(1, N_EPOCH + 1)):
            X_s = Traindata_S
            X_tv = Valdata_T
            tmp_train_acc, tmp_train_loss, mdl = \
                model_train(model=mdl, optimizer=opt_Adam, epoch=ep, data_src=X_s, data_tar=X_tv, y_pse=y_pse,
                            factor=1.0)

            train_loss.append(tmp_train_loss)
        loss['train' + str(train_num)] = train_loss
        loss_result = pd.DataFrame(loss)
        Loss.append(loss_result)
        #  save model
        model_paths = model_path + str(train_num) + '.pth'
        torch.save(mdl, model_paths)

      # ================test model============
        file_name_list = os.listdir(model_path)
        results_T, results_P = model_test_result(mdl, Valdata_TP)
        T_data = results_T.values
        P_data = results_P.values

        My_std_t = joblib.load(std_path_t)
        T_data = My_std_t.inverse_transform(np.asarray(T_data))
        P_data = My_std_t.inverse_transform(np.asarray(P_data))

        MY_distance, Degenerate_point = Distance(P_data, sigma)        
