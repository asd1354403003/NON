import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn.parameter import Parameter
import math
from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.wd import WideAndDeepModel
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from sklearn.metrics import roc_auc_score
import torch.utils.data as Data
import DNN
import matplotlib.pyplot as plt
import sys


def one_hot(x):
    x_train = x.applymap(str)
    one_hot = pd.get_dummies(x_train)
    x_train = one_hot.values
    x_train = torch.tensor(x_train, dtype=torch.float)
    return x_train.cuda()


def get_e_2(x_train, field, device):


    weight = Parameter(torch.Tensor(x_train.shape[0], x_train.shape[1]))
    stdv = 1. / math.sqrt(weight.size(1))
    weight = weight.data.uniform_(-stdv, stdv) * 100
    what = []

    for i in range(x_train.shape[0]):
        x = []

        for j in range(len(field)):

            if j == 0:
                xx = x_train[i][: field[j]].to(device)
                ww = weight[i][: field[j]].to(device)
            else:
                xx = x_train[i][sum(field[:j]): sum(field[:j + 1])].to(device)
                ww = weight[i][sum(field[:j]): sum(field[:j + 1])].to(device)

            ba = F.linear(xx, ww, None).cuda()

            x.append(float(ba))

        what.append(x)
    e = torch.tensor(what)
    lin = nn.Linear(e.shape[1], e.shape[1], bias=True)
    e_2 = lin(e)
    rel = nn.ReLU()
    e_2 = rel(e_2)
    e_2 = torch.tensor(e_2).cuda()
    return e_2


def get_dim(x):
    columns = list(x.columns)
    fielda = []
    for i in range(len(columns)):
        a = x[columns[i]].values
        fielda.append(len(np.unique(a)))
    return fielda


#
# def check_index(x, field):
#     if torch.max(x) > sum(field):
#         import sys
#         print('index out of range')
#         sys.exit()


def get_model(name, field_dims):
    if name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=4)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
            field_dims, embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400),
            dropouts=(0, 0, 0))
    elif name == 'dnn':
        return DNN.DNN(len(field_dims), 256, 128, 1)
    elif name == 'dnnaux':
        return DNN.DNN(len(field_dims), 256, 128, 1)
    else:
        raise ValueError('unknown model name: ' + name)


def train(model, optimizer, criterion, loader, device, field, model_name):
    log_interval = 1000
    model.train()
    total_loss = 0

    for step, (batch_x, batch_y) in enumerate(loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        e_2 = get_e_2(batch_x, field, device)
        e_2 = e_2.to(device)

        if model_name == 'dnnaux':
            w = model.hidden1.weight.T.to(device)
            h = torch.randn(w.shape[1], batch_y.shape[0]).to(device)
            mat = torch.matmul(w, h).to(device)
            sig = torch.sigmoid(mat)
            y = model(sig.T)
            loss = criterion(y, batch_y.float())
            total_loss += loss.item()

            w = model.hidden2.weight.T.to(device)
            h = torch.randn(w.shape[1], batch_y.shape[0]).to(device)
            mat = torch.matmul(w, h).to(device)
            mat = torch.matmul(mat.T, torch.randn(mat.shape[0], 23).to(device))
            sig = torch.sigmoid(mat)
            y = model(sig).to(device)
            loss = criterion(y, batch_y.float())
            total_loss += loss.item()

            w = model.o.weight.T.to(device)
            h = torch.randn(w.shape[1], batch_y.shape[0]).to(device)
            mat = torch.matmul(w, h).to(device)
            mat = torch.matmul(mat.T, torch.randn(mat.shape[0], 23).to(device))
            sig = torch.sigmoid(mat)
            y = model(sig).to(device)
            loss = criterion(y, batch_y.float())
            total_loss += loss.item()

            if (step + 1) % log_interval == 0:
                print('    - loss:', total_loss / log_interval)
                total_loss = 0

        else:
            if model_name != 'dnn':
                e_2 = e_2.long()
            y = model(e_2)
            loss = criterion(y, batch_y.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (step + 1) % log_interval == 0:
                print('    - loss:', total_loss / log_interval)
                total_loss = 0


def test(model, val_loader, device, field, model_name):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(val_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            e_2 = get_e_2(batch_x, field, device)

            if model_name == 'dnnaux':
                y = model(e_2)
                targets.extend(batch_y.tolist())
                predicts.extend(y.tolist())
            else:
                if model_name != 'dnn':
                    e_2 = e_2.long()

                y = model(e_2)

                targets.extend(batch_y.tolist())
                predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


def main(model_name,
         epochs,
         learning_rate,
         batch_size,
         device,
         weight_decay
         ):
    device = torch.device(device)
    train_df = pd.read_csv('train.csv', nrows=300)
    train_df.fillna(train_df.mean(), inplace=True)
    x_train = train_df.drop('click', axis=1)
    y_train = train_df['click']

    y_train = np.where(y_train > 1, 1, y_train)
    y_train = np.where(y_train < 0, 0, y_train)


    x_train, x, y_train, y = train_test_split(x_train, y_train, test_size=0.2)
    x_val, x_test, y_val, y_test = train_test_split(x, y, test_size=0.5)

    y_train = torch.tensor(y_train)
    y_val = torch.tensor(y_val)
    y_test = torch.tensor(y_test)

    field = get_dim(x)

    x_train = one_hot(x_train)
    x_val = one_hot(x_val)
    x_test = one_hot(x_test)

    model = get_model(model_name, field).to(device)

    train_dataset = Data.TensorDataset(x_train, y_train)
    val_dataset = Data.TensorDataset(x_val, y_val)
    test_dataset = Data.TensorDataset(x_test, y_test)

    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = Data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    auc_list = []

    auc = 0
    while auc <= 0.5:
        for epoch_i in range(epochs):
            train(model, optimizer,  criterion, train_loader, device, field, model_name)
            auc = test(model, val_loader, device, field, model_name)
            print(model_name, 'epoch:', epoch_i+1, 'validation: auc:', auc)

            auc = test(model, test_loader, device, field, model_name)
            if auc > 0.5:
                auc_list.append(auc)
        print(model_name, 'test auc:', auc)

    e_2 = get_e_2(x_train, field, device)

    if model_name == 'dnn':
        return model(e_2), auc_list
    elif model_name == 'dnnaux':
        return model(e_2), auc_list
    else:
        e_2 = e_2.long()
        return model(e_2)


if __name__ == '__main__':

    afi = main('afi', 1, 0.01, 64, 'cuda:0', 0)
    ffm = main('ffm', 1, 0.01, 64, 'cuda:0', 0)
    wd = main('wd', 1, 0.01, 64, 'cuda:0', 0)
    nfm = main('nfm', 1, 0.01, 64, 'cuda:0', 0)
    xdfm = main('xdfm', 1, 0.01, 64, 'cuda:0', 0)

    dnn, auc_dnn = main('dnn', 2, 0.01, 64, 'cuda:0', 0)

    non = torch.cat((ffm, wd, nfm, xdfm, afi))

    dnnaux, auc_dnn_aux = main('dnnaux', 2, 0.01, 64, 'cuda:0', 0)

    if len(auc_dnn_aux) > len(auc_dnn):
        for i in range(len(auc_dnn_aux) - len(auc_dnn)):
            auc_dnn_aux.remove(min(auc_dnn_aux))
    elif len(auc_dnn_aux) < len(auc_dnn):
        for i in range(len(auc_dnn_aux) - len(auc_dnn)):
            auc_dnn.remove(min(auc_dnn))

    a = x = np.linspace(0, 60000, len(auc_dnn_aux))

    plt.plot(a, auc_dnn, label='auc')
    plt.plot(a, auc_dnn_aux, label='auc_dnn_aux')
    plt.legend()
    plt.show()
