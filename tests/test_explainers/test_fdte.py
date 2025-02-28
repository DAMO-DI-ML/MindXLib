import numpy as np
from mindxlib.explainers.timeseries import FDTempExplainer
import pandas as pd
import torch
import sys
import torch.nn as nn
# 测试代码
class LSTM_new_1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,num_layers=1,task='classification',ReVIn=False):
        super(LSTM_new_1, self).__init__()
        self.hidden_dim = hidden_dim
        self.task = task
        self.ReVIn = ReVIn

        # LSTM以word_embeddings作为输入, 输出维度为 hidden_dim 的隐藏状态值
        self.lstm = nn.LSTM(input_dim, hidden_dim,num_layers, batch_first=True, dropout=0.4)

        # 线性层将隐藏状态空间映射到标注空间
        self.hidden2tag = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 一开始并没有隐藏状态所以我们要先初始化一个
        # 关于维度为什么这么设计请参考Pytoch相关文档
        # 各个维度的含义是 (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 256, self.hidden_dim),
                torch.zeros(1, 256, self.hidden_dim))

    def forward(self, x): 
        # input: batch_size X #features X seq_len
        # output: batch_size X #features X seq_len
        x = x.transpose(1,2)
        if self.ReVIn:
            mean = torch.mean(x,dim=1,keepdim=True)
            std = torch.std(x,dim=1,keepdim=True)
            x = (x - mean)/(std+0.001)
        lstm_out, _ = self.lstm(x)
        tag_space = self.hidden2tag(lstm_out)
        if self.task == 'classification':
            tag_scores = torch.softmax(tag_space,axis=-1)
        elif self.ReVIn:
            tag_scores = tag_space*(std+0.001) + mean
        else:
            tag_scores = tag_space
        return tag_scores.transpose(1,2) 

# 用户自己实现
    def predict(self, x, only_last=False):
        return self.forward(x)[:,:,-1]
    
    def train_model(self,train_dataset, test_dataset=None,lr=0.01,batch_size=320,n_epochs=500):
        train_loader=Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader=Data.DataLoader(dataset=test_dataset, batch_size=batch_size*100, shuffle=False)
        optimizer =torch.optim.Adam(self.parameters(), lr=lr, betas=(0.8, 0.9),weight_decay=1e-6)
#         scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
        scheduler = CosineAnnealingLR(optimizer, T_max=300)
        self.train()
        eps = 1e-10
        for epoch in range(n_epochs):
            train_loss = 0.0
            eps = eps*0.9
            # temperature = (1-1/n_epochs)*temperature
            self.train()
            N_train = 0
            for step,(X_batch,target_batch) in enumerate(train_loader):
                y_hat = self.forward(X_batch)
                dim_output = target_batch.shape[-1]
                if self.task == 'classification':
                    loss = torch.mean(torch.mean(-target_batch * torch.log(y_hat[:,:,-1]+1e-1) - (1-target_batch) * torch.log(1-y_hat[:,:,-1]+1e-1),dim=0),dim=0)
                else:
                    loss = torch.mean(torch.mean((target_batch-y_hat[:,-dim_output:])**2,dim=0),dim=0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*X_batch.shape[0]
                N_train += X_batch.shape[0]

            train_loss = train_loss / N_train
            scheduler.step()

            self.eval()
            with torch.no_grad():
                epoch_test_loss = 0.0
                N_test = 0
                for step, (X_batch, y_batch) in enumerate(test_loader):
                    output = self.forward(X_batch)[:,:,-1].argmax(dim=-1)
                    loss = torch.sum(1.0*(y_batch.argmax(dim=-1)==output),dim=0)
                    epoch_test_loss += loss.detach().cpu().numpy()
                    N_test += X_batch.shape[0]
                test_loss = epoch_test_loss/N_test
                if epoch % 10 == 0:
                    print(f'Epoch: {epoch+1} Train mse = {np.round(train_loss,4)}, test acc = {np.round(test_loss,4)}')

    
        print(f'Epoch: {epoch+1} Train mse = {np.round(train_loss,4)}, test mse = {np.round(test_loss,4)}')

        

if __name__ == "__main__":
    
    print("Running tests for FDTempExplainer...")
    print("sys.path",sys.path)

    # 将数据移动到 GPU（如果可用）
    device = "cuda:1" 
    # 模拟时间序列数据
    data = np.random.rand(5, 50, 2)  # 100 个样本，每个样本 50 个时间步，10 个特征

    # data = torch.from_numpy(data)

    lstm = LSTM_new_1(input_dim=1, hidden_dim=120, output_dim=4,num_layers=3,task='classification').to(device)
    lstm.load_state_dict(torch.load('/mnt/workspace/workgroup/workgroup/yitian/MindXLib/tests/test_explainers/test_model/lstm-LKA.pt'))


    # 实例化 FDTempExplainer
    explainer = FDTempExplainer(
        model=lstm,
        data=data
    )


    # 调用 explain 方法进行解释
    explainer.explain(
        data,
        patch_size = 5,
        test_size=0.2,
        random_state=42,
        device=device
    )


    # 打印训练集和测试集的形状
    print("Train data shape:", explainer.data.shape)

    # 获取主效应和交互效应
    print("Main effects shape:", explainer.attribution_results['main_effect'].shape)
    print("Interaction effects shape:", explainer.attribution_results['interaction_effect'].shape)

    print("Tests completed successfully!")