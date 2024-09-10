import numpy as np

import torch
from torch import nn

def xavier_uniform_weights_init(model):
    for layer in model:
        if type(layer) is torch.nn.modules.linear.Linear:
                gain = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(layer.weight, gain=gain)
                torch.nn.init.uniform_(layer.bias)
                
                
def xavier_normal_weights_init(model):
    for layer in model:
        if type(layer) is torch.nn.modules.linear.Linear:
                gain = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_normal_(layer.weight, gain=gain)
                torch.nn.init.normal_(layer.bias)
                
    
def he_uniform_weights_init(model):
    for layer in model:
        if type(layer) is torch.nn.modules.linear.Linear:
                torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.uniform_(layer.bias)
                
                
def he_normal_weights_init(model):
    for layer in model:
        if type(layer) is torch.nn.modules.linear.Linear:
                torch.nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.normal_(layer.bias)
                
                
def generate_model(D_in, D_hidden, weight_init=False, inner_f='relu', out_f='sigmoid', dropout_p=0.0):
    
    if inner_f == 'relu':
        inner_function = nn.ReLU
    elif inner_f == 'tanh':
        inner_function = nn.Tanh
    
    if out_f == 'sigmoid':
        out_function = nn.Sigmoid
    elif out_f == 'tanh':
        out_function = nn.Tanh
    
    elif dropout_p:
        
        model = nn.Sequential(
            nn.Linear(D_in, 2*D_hidden),
            nn.Dropout(p=dropout_p),
            inner_function(),

            nn.Linear(2*D_hidden, D_hidden),
            nn.Dropout(p=dropout_p),
            inner_function(),

            nn.Linear(D_hidden, 2*D_hidden),
            nn.Dropout(p=dropout_p),
            inner_function(),

            nn.Linear(2*D_hidden, D_in),
            out_function()
        )
    else:
        
        model = nn.Sequential(
            nn.Linear(D_in, 2*D_hidden),
            inner_function(),

            nn.Linear(2*D_hidden, D_hidden),
            inner_function(),

            nn.Linear(D_hidden, 2*D_hidden),
            inner_function(),

            nn.Linear(2*D_hidden, D_in),
            out_function()
        )       
    
    if weight_init == 'xavier_uniform':
        xavier_uniform_weights_init(model)
        
    if weight_init == 'xavier_normal':
        xavier_normal_weights_init(model)
        
    if weight_init == 'he_uniform':
        he_uniform_weights_init(model)
        
    if weight_init == 'he_normal':
        he_normal_weights_init(model)
        
    return model


def train_models(X_train, y_train, X_valid, y_valid, epochs, model_lst, optimizer_lst):
    
    print("training models")
    
    criterion = nn.MSELoss()
    
    for model in model_lst:
        model.double()
    
    loss_array = np.zeros((len(model_lst), 2, epochs)) # (num of models, [0==train;1==valid], num of epochs)
    valid_min_lst = [np.Inf for _ in range(len(model_lst))]
    
    for epoch in range(epochs):

        # Shuffle training set
        num_train = len(X_train)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        # Shuffle validation set
        num_valid = len(X_valid)
        indices = list(range(num_valid))
        np.random.shuffle(indices)
        X_valid = X_valid[indices]
        y_valid = y_valid[indices]

        train_loss_lst = [0.0 for _ in range(len(model_lst))]

        for x, y in zip(X_train, y_train):
            for i, (model, optimizer) in enumerate(zip(model_lst, optimizer_lst)):
                optimizer.zero_grad()
                y_ = model(x)
                loss = criterion(y, y_)
                train_loss_lst[i] += loss.item()
                loss.backward()
                optimizer.step()

        else:
            valid_loss_lst = [0.0 for _ in range(len(model_lst))]
            with torch.no_grad():
                for x, y in zip(X_valid, y_valid):
                    for i, model in enumerate(model_lst):   
                        y_ = model(x)
                        loss = criterion(y, y_)
                        valid_loss_lst[i] += loss.item()

        for i, _ in enumerate(model_lst):
            loss_array[i][0][epoch] = train_loss_lst[i]/len(X_train)
            loss_array[i][1][epoch] = valid_loss_lst[i]/len(X_valid)

        for i, _ in enumerate(model_lst):      
            if loss_array[i][1][epoch] <= valid_min_lst[i]:
                torch.save(model_lst[i].state_dict(), f'model_{i}.pt')
                valid_min_lst[i] = loss_array[i][1][epoch]
    
    return model_lst


def evaluate_models(X_test, y_test, model_lst):
    
    print("evaluating models")
    
    criterion = nn.MSELoss()
    
    X_corrected = np.zeros((len(model_lst), X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    before_loss = np.zeros((len(model_lst), X_test.shape[0], X_test.shape[1]))
    after_loss  = np.zeros((len(model_lst), X_test.shape[0], X_test.shape[1]))
    mu_before   = np.zeros(len(model_lst))
    std_before  = np.zeros(len(model_lst))
    mu_after    = np.zeros(len(model_lst))
    std_after   = np.zeros(len(model_lst))
    
    for i, model in enumerate(model_lst):
        # i  - model number
        with torch.no_grad():
            X_corrected[i] = model(X_test)
        for n in range(X_test.shape[0]):
            # n - initial state number
            for t in range(X_test.shape[1]):
                # t - time point number
                before_loss[i][n][t] = criterion(X_test[n][t], y_test[n][t])
                after_loss[i][n][t]  = criterion(torch.from_numpy(X_corrected[i][n][t]), y_test[n][t])

        mu_before[i]  = np.mean(before_loss[i])
        std_before[i] = np.std(before_loss[i], ddof=1)
        mu_after[i]   = np.mean(after_loss[i])
        std_after[i]  = np.std(after_loss[i], ddof=1)
    
    return (mu_before, std_before, mu_after, std_after)


def generate_simple_model(D_in, D_hidden, inner_f='LReLU', out_f='sigmoid'):
    
    """
    Generate a neural network of a form in - hid - hid - in
    """
    
    inner_function = nn.LeakyReLU
    if out_f == 'sigmoid':
        out_function   = nn.Sigmoid
    elif out_f == 'tanh':
        out_function   = nn.Tanh
    
    model = nn.Sequential(
        nn.Linear(D_in, D_hidden),
        inner_function(),

        nn.Linear(D_hidden, D_hidden),
        inner_function(),

        nn.Linear(D_hidden, D_in),
        out_function()
    )
        
    he_normal_weights_init(model)
        
    return model


def generate_onelayer_model(D_in, D_hidden, inner_f='LReLU', out_f='sigmoid'):
    
    """
    Generate a neural network of a form in - hid - in
    """
    
    inner_function = nn.LeakyReLU
    if out_f == 'sigmoid':
        out_function   = nn.Sigmoid
    elif out_f == 'tanh':
        out_function   = nn.Tanh
    
    model = nn.Sequential(
        nn.Linear(D_in, D_hidden),
        inner_function(),

        nn.Linear(D_hidden, D_in),
        out_function()
    )
        
    he_normal_weights_init(model)
        
    return model


def generate_onelayer_clones(D_in, D_hidden_lst=[1,], inner_f='LReLU', out_f='sigmoid', N_clones=2):

    """
    Generate a matrix of neural networks of a form in - hid - in. 
    Matrix form: (hidden layer width, clones)
    """    
    model_matrix = []

    for D_hidden in D_hidden_lst:
        models = []

        parent_model = generate_onelayer_model(D_in, D_hidden, inner_f=inner_f, out_f=out_f)
        models.append(parent_model)
        for _ in range(N_clones-1):
            model = generate_onelayer_model(D_in, D_hidden, inner_f=inner_f, out_f=out_f)
            model.load_state_dict(models[0].state_dict())
            models.append(model)
        model_matrix.append(models)

    return model_matrix