### 我李健立自己思考解決、絕無抄襲 ###
import numpy as np
import matplotlib.pyplot as plt 
np.random.seed(543)
np.set_printoptions(linewidth=300)   #讓矩陣輸出時不要換行
np.set_printoptions(suppress = True)   #不顯示科學記號

from mnist import load_mnist

def sigmoid(x):
    return 1.0 / (1+np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
 
#def softmax(x):
    #s = np.zeros_like(x , dtype = float)
    #exp_x = np.zeros_like(x , dtype = float)
    #for i in range(x.shape[0]):
        #c = np.max(x[i])
        #exp_x[i] = np.exp(x[i]-c)   #boardcast   #防溢位
        #s[i] = exp_x[i] / np.sum(exp_x[i])
    #return s

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    
def L(s):
    B = len(s)
    return -np.sum(t_B*np.log(s)) / B

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def G_S(s):
    gs = -(t_B/s) / B
    return gs

def S_a2(s):
    ga2 = np.zeros((s.size,s.size) , dtype = float)
    for i in range(s.size):
        ga2[i,i] = s[i]*(np.sum(s)-s[i])
    for i in range(1,s.size):
        for j in range(i,s.size):
            ga2[i-1,j] = -s[i-1]*s[j]
            ga2[j,i-1] = ga2[i-1,j]
    return ga2

def La2(a2):
    La2 = np.zeros_like(a2 , dtype = float)
    for i in range(B):
        La2[i] = np.dot( L_s[i] , S_a2(s[i]) )
    return La2

def Z_a1(z):
    ga1 = np.zeros((z.size,z.size) , dtype = float)
    for i in range(z.size):
        ga1[i,i] = z[i]*(1-z[i])
    return ga1

def acc(x,t):
    a1 = np.dot(x,W1) + b1   #B*50
    z = sigmoid(a1)   #B*50
    a2 = np.dot(z,W2) + b2   #B*10
    s1 = softmax(a2)   #B*10
    s_idx = np.argmax(s1,axis = 1)   #每列最大值
    t_idx = np.argmax(t,axis = 1)
    acc = np.sum(s_idx == t_idx) / float(x.shape[0])
    return acc

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
n1,n2,n3 = 784,50,10   
W1 = 0.01*np.random.randn(n1,n2)   #784*50
b1 = np.zeros(n2)   #(50,)
W2 = 0.01*np.random.randn(n2,n3)   #50*10
b2 = np.zeros(n3)   #(10,)
iters_num = 10000
train_size = x_train.shape[0]
B = 100
lr = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / B , 1) 

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, B)
    x_B = x_train[batch_mask]   #B*784
    t_B = t_train[batch_mask]   #B*10
    
    #####Forward#####
    a1 = np.dot(x_B,W1) + b1   #B*50
    z = sigmoid(a1)   #B*50
    a2 = np.dot(z,W2) + b2   #B*10
    s = softmax(a2)   #B*10

    #####Backward#####
    #L_s = G_S(s)   #B*10 
    #s_a2 = S_a2(s[0])   #10*10 的對稱矩陣
    #print(s_a2.round(3))
    #L_a2 = La2(a2)   #B*10
    L_a2 = (s - t_B) / B   #(簡化後)
    L_b2 = np.dot( np.ones(B) , L_a2 )  #1*10
    L_W2 = np.dot(z.T , L_a2)  #50*10
    L_z = np.dot(L_a2 , W2.T)  #B*50
    #z_a1 =  Z_a1(z[])  #50*50 的對角矩陣
    L_a1 = sigmoid_grad(a1) * L_z  #B*50
    L_b1 = np.dot( np.ones(B) , L_a1 )  #1*50
    L_W1 = np.dot(x_B.T , L_a1)  #784*50
    
    #####SGD更新參數#####
    W2 -= lr*L_W2
    b2 -= lr*L_b2
    W1 -= lr*L_W1
    b1 -= lr*L_b1
    
    #####更新後#####
    a1 = np.dot(x_B,W1) + b1   #B*50
    z = sigmoid(a1)   #B*50
    a2 = np.dot(z,W2) + b2   #B*10
    s = softmax(a2)   #B*10
    
    #####紀錄loss function#####
    loss = cross_entropy_error(s, t_B)
    train_loss_list.append(loss)

    #####計算 1 epoch 的準確率#####
    if i % iter_per_epoch == 0 :
        train_acc = acc(x_train,t_train)
        test_acc = acc(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train_acc | test_acc = " , str(train_acc) , "|" , str(test_acc))

print('準確率 =' , (train_acc*100).round(2) , '%' )
print('loss 從' , train_loss_list[0] , '下降到' , loss)







    

