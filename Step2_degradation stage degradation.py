# -*- coding: utf-8 -*-
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
import random
from MY_models_p import Classification_Dann
from torch.autograd import Variable 
import torch.utils.data as Data
import torch
import My_MMD
import os
import torch.nn.init as init
import matplotlib.animation as animation
from My_readdata import read_data, read_data_fe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  

def mmd_loss(x_src, y_src, x_tar, y_pseudo, factor=0.0):
    return My_MMD.rbf_mmd(x_src, y_src, x_tar, y_pseudo, factor)


def train_val(train_dataloader, Target_data, Target_health, val_dataloder, num_epochs=30, factors=0.5):

    Models_net.to(device)
    LOSS = []
    Crute = []
    Accu = []
    for epoch in range(num_epochs):
        Models_net.train()
        all_loss = 0
        corrects = 0
        total = 0
       
        lens_s = len(train_dataloader)
        lens_t = len(Target_data)
        lens_h = len(Target_health)
        Lens = max(lens_s, lens_t)
        list_src, list_tar, list_health = list(enumerate(train_dataloader)), \
                                          list(enumerate(Target_data)), \
                                          list(enumerate(Target_health))

        for i in range(0, Lens):
            optimizer.zero_grad()
            # Source
            if i <= (lens_s-1):
                _, (images, labels) = list_src[i]
            else:
                rand_num = random.randint(0, lens_s -1)
                _, (images, labels) = list_src[rand_num]

            # Target
            if i <= (lens_t-1):
                _, (images_t, labels_t) = list_tar[i]
            else:
                rand_num_t = random.randint(0, lens_t - 1)
                _, (images_t, labels_t) = list_tar[rand_num_t]

            # Health
            if i <= (lens_h-1):
                _, (images_h, labels_h) = list_health[i]
            else:
                rand_num_h = random.randint(0, lens_h - 1)
                _, (images_h, labels_h) = list_health[rand_num_h]       
            images = Variable(images).to(device)
            labels = Variable(labels.float()).to(device)
            images_t = Variable(images_t).to(device)
            labels_t = Variable(labels_t).to(device)
            images_h = Variable(images_h).to(device)
            labels_h = Variable(labels_h).to(device)       
            # Source        
            predict_labels, source_fe, target_fe = Models_net(images, images_t)       
            labels = labels.long()
            C_loss = criterion_classify(predict_labels, labels)
            Loss_fe_S = criterion_S(source_fe, images)
            # Target
            predict_label_t, target_fe, source_fe = Models_net(images_t, images)
            _, Pseudo = torch.max(predict_label_t.data[:, 1:], 1)
            Pseudo = Pseudo + 1
            # Health
            predict_label_h, target_fe_h, _ = Models_net(images_h, images)          
            source_fe = source_fe.view(len(source_fe), -1)
            target_fe = target_fe.view(len(target_fe), -1)
            loss_mmd = mmd_loss(source_fe, predict_labels, target_fe, predict_label_t, factors)
            P_loss = criterion_classify(predict_label_t, Pseudo)
            P_loss_h = criterion_classify(predict_label_h, labels_h.long())
            Loss_fe_T = criterion_T(target_fe, images_t)
            _, my_pre = torch.max(predict_labels.data, 1)
            corrects += (my_pre == labels).sum().item()
            total += labels.size(0)
            random_value = torch.rand(1).item() * 0.5 # 1 #
            loss = P_loss + C_loss + P_loss_h + (Loss_fe_S + Loss_fe_T + loss_mmd)* random_value
            loss.backward()
            all_loss += loss
            lr = optimizer.param_groups[0]['lr']
            optimizer.step()
            print('train_batch========>is', i + 1)

        all_loss = all_loss / (i + 1)
        print("epoch:", epoch + 1, "loss:", all_loss.item())
        print('Train accuracy is============> {:.4f}%'.format(corrects / total * 100))
        LOSS.append(all_loss.item())

#  ----------------Test--------------
        Models_net.eval()

        correct = 0
        T_data = []
        P_data = []
        for i, (v_images, v_labels) in enumerate(val_dataloder):

            v_images = Variable(v_images).to(device)
            v_labels = Variable(v_labels.float()).to(device)
            predict_labels_v, _, _ = Models_net(v_images, v_images)
            T_data.extend(v_labels.cpu().detach().numpy())
            P_data.extend(predict_labels_v.cpu().detach().numpy())

            T_pre = predict_labels_v.data.max(1)[1]

            correct += T_pre.eq(v_labels).sum()
        accuracy = float(correct) / len(val_dataloder.dataset)
        print('validation accuracy=============>is{}%'.format(accuracy * 100))

    torch.save(Models_net, save_model_path) 

    df = pd.DataFrame(LOSS, columns=['loss'])
    df.to_csv(save_loss_path)
    print("save last model")
    return Models_net

def Test_model(testdata, model):

    model.eval()
    model.to(device)
    correct = 0
    Pre_L = []
    T_data = []
    P_data = []
    total = 0

    for i, (T_images, T_labels) in enumerate(testdata):
        v_images = Variable(T_images).to(device)
        v_labels = Variable(T_labels.float()).to(device)
        predict_labels_v, _, _ = model(v_images, v_images)
        T_data.extend(v_labels.cpu().detach().numpy())
        P_data.extend(predict_labels_v.cpu().detach().numpy())
        _, my_pre = torch.max(predict_labels_v.data, 1)
        Pre_L.extend(my_pre.cpu().detach().numpy())
        correct += (my_pre == v_labels).sum().item()
        total += v_images.size(0)

    print('test accuracy is============> {:.4f}%'.format(correct / total * 100))
    Pre_L = pd.DataFrame(Pre_L)
    return T_data, P_data, Pre_L

# initial weights
def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)


if __name__ == "__main__":
    # ------------------read data------------------#
    root_path = 'G:/'
    predict_root_path = 'G:/'
    source_name = 'Bearing1_2'
    Bearing_names = 'Bearing1_1'
    First_point = 910
    # ========resource========

    path_s = root_path + 'data_fe/' + source_name + '.mat'  #
    stdpath = './degradation/STD/' + source_name + 'B_std_abs.m'
    my_step = 1
  
    train_x_s = read_data_fe(path_s, step=my_step, pathsnames=stdpath)
    train_x_s = train_x_s.reshape(len(train_x_s), 1, -1)
    My_labels = pd.read_excel(root_path + '/Label/' + source_name + '_label.xlsx')

    My_label = My_labels['Labels'].iloc[0:,]
    start_index = My_label[My_label == 1].index[0]
    train_x_s = train_x_s 
    train_y_s = My_label.values 

    # ===========Target===========

    path_t = predict_root_path + '/data_fe/' + Bearing_names + '.mat'  
    stdpath_t = './degradation/STD/' + Bearing_names + 'std_t.m'
    degenerate_path = './degradation/results/' + Bearing_names + '.xlsx'
    train_x_t = read_data_fe(path_t, step=my_step, pathsnames=stdpath_t)
    train_x_t = train_x_t.reshape(len(train_x_t), 1, -1)
    train_y_t = np.concatenate([np.zeros(First_point), np.ones(len(train_x_t) - First_point)],
                               axis=0)
    # --------save path-------------------
    save_model_path = './degradation/models/' + Bearing_names+'predict.pkl'  
    save_loss_path = './degradation/loss/' + Bearing_names+'predict.csv'
    # parameter setting
    learning_rate = 0.001
    epochs = 100
    batch_size = 60
    batch_size_t = batch_size
    My_shapes = train_x_s.shape
    classfication = 4

    Models_net = Classification_Dann(output_size=classfication, step = my_step)
    Models_net.apply(weights_init)

    criterion_classify = nn.CrossEntropyLoss(reduction='mean')
    criterion_S = nn.MSELoss(reduction='mean')
    criterion_T = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(Models_net.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=30)


    # ------------Source----------

    train_X = Variable(torch.Tensor(train_x_s))
    train_Y = Variable(torch.Tensor(train_y_s))
    Torch_Traindata = Data.TensorDataset(train_X, train_Y)
    Torch_Traindata = Data.DataLoader(dataset=Torch_Traindata, batch_size=batch_size, shuffle=True, pin_memory=False,
                                      num_workers=0, drop_last=False)
    ##------------ validate  ----------
    val_X = Variable(torch.Tensor(train_x_s))
    val_Y = Variable(torch.Tensor(train_y_s))
    Torch_Valdata = Data.TensorDataset(val_X, val_Y)
    Torch_Valdata = Data.DataLoader(dataset=Torch_Valdata, batch_size=batch_size, shuffle=False, pin_memory=False,
                                    num_workers=0, drop_last=False)

    #            ===Target===
    # ---------------------health-----------------------
    T_X_health = Variable(torch.Tensor(train_x_t[:First_point,:, :]))
    T_Y_health = Variable(torch.Tensor(train_y_t[:First_point]))
    Target_health = Data.TensorDataset(T_X_health, T_Y_health)
    Target_health = Data.DataLoader(dataset=Target_health, batch_size=batch_size_t, shuffle=False, pin_memory=False,
                                  num_workers=0, drop_last=False)
    # degradation assessment
    Target_X = Variable(torch.Tensor(train_x_t[First_point:,:, :]))
    Target_Y = Variable(torch.Tensor(train_y_t[First_point:]))
    Target_data = Data.TensorDataset(Target_X, Target_Y)
    Target_data = Data.DataLoader(dataset=Target_data, batch_size=batch_size_t, shuffle=False, pin_memory=False,
                                      num_workers=0, drop_last=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    Target_X = Variable(torch.Tensor(train_x_t))
    Target_Y = Variable(torch.Tensor(train_y_t))
    All_Target_data = Data.TensorDataset(Target_X, Target_Y)
    All_Target_data = Data.DataLoader(dataset=All_Target_data, batch_size=1, shuffle=False, pin_memory=False,
                                  num_workers=0, drop_last=False)



    # # =================》training《==============
    factors = 1.0

    for nums in range(1, 7):
        degenerate_path = './degradation/results/' + Bearing_names + '_' + str(nums) + '_' + str(factors) + 'predict.xlsx'

        modle = train_val(Torch_Traindata, Target_data, Target_health, Torch_Valdata, num_epochs= epochs, factors=factors)

        # ------------Load model----------
        model_2 = torch.load(save_model_path)
        total = sum([param.nelement() for param in model_2.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        # test model
        T_data, P_data, Degenerates = Test_model(All_Target_data, model_2)
        Degenerates.to_excel(degenerate_path)
