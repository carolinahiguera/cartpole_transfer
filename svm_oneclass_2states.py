import pickle
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt 

# path_task = f'./output/'


# with open(f'{path_task}/seq_states_T0.pkl', 'rb') as f:
#     seq_states_t0 = pickle.load(f)

# with open(f'{path_task}/seq_states_T1.pkl', 'rb') as f:
#     seq_states_t1 = pickle.load(f)

seq_lenght = 2

dataset_task0 = np.empty((0,4*seq_lenght))
dataset_task1 = np.empty((0,4*seq_lenght))

def get_datasets():
    global dataset_task0, dataset_task1
    print('Tarea 0')
    for i in range( len(seq_states_t0)):
        print(f'Tarea 0 - Episodio {i}')
        for j in range(1,len(seq_states_t0[i])):
            seq = [seq_states_t0[i][j-1][0], seq_states_t0[i][j-1][1],  seq_states_t0[i][j-1][2],  seq_states_t0[i][j-1][3],
                   seq_states_t0[i][j][0], seq_states_t0[i][j][1], seq_states_t0[i][j][2], seq_states_t0[i][j][3]]
            dataset_task0 = np.append(dataset_task0, np.array([seq]), axis=0)
    print('Tarea 1')
    for i in range( len(seq_states_t1)):
        print(f'Tarea 1 - Episodio {i}')
        for j in range(1,len(seq_states_t1[i])):
            seq = [seq_states_t1[i][j-1][0], seq_states_t1[i][j-1][1], seq_states_t1[i][j-1][2], seq_states_t1[i][j-1][3],
                   seq_states_t1[i][j][0], seq_states_t1[i][j][1], seq_states_t1[i][j][2], seq_states_t1[i][j][3]]
            dataset_task1 = np.append(dataset_task1, np.array([seq]), axis=0)
    

def compute_confusion_matrix(clfs):
    conf_matrix = np.zeros((4,4))
    for t in range(4):
        for j in range(4):
            y_pred = clfs[t].predict(Xtest[j])
            if(j==t):
                n_error = y_pred[y_pred == -1].size / Xtest[j].shape[0]
            else:
                n_error = y_pred[y_pred == 1].size / Xtest[j].shape[0]
            conf_matrix[t][j] = n_error
    return conf_matrix

def compute_small_confusion_matrix(clfs):
    tasks = [0,1]
    conf_matrix = np.zeros((len(tasks),len(tasks)))
    for t,task_0 in enumerate(tasks):
        for j,task_1 in enumerate(tasks):
            y_pred = clfs[t].predict(Xtest[task_1])
            if(j==t):
                n_error = y_pred[y_pred == -1].size / Xtest[task_1].shape[0]
            else:
                n_error = y_pred[y_pred == 1].size / Xtest[task_1].shape[0]
            conf_matrix[t][j] = n_error
    return conf_matrix


# print('Extrayendo las secuencias')
# get_datasets()
# print('Fin secuencias')


path = f'./output/svm_2states/'
X = np.load(f'{path}/Xtrain.npy')
Y = np.load(f'{path}/Ytrain.npy')

dataset_task0 = shuffle(X[Y==0])
y_t0 = Y[Y==0]
dataset_task1 = shuffle(X[Y==1])
y_t1 = Y[Y==1]

X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(dataset_task0, y_t0, test_size=0.9)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(dataset_task1, y_t1, test_size=0.9)




# nu = [10e-3, 10e-2,  0.2, 0.5, 0.7, 0.9]
# nu = [0.2, 0.5, 0.7, 0.9]
# kernel = ['rbf']
# degree = [2]
# gamma = [10e-4, 10e-3, 10e-2, 10e-1, 10e-0]

nu = [0.1]#0.1
kernel = ['poly']#'rbf'
degree = [2]
gamma = [0.01]#0.005

Xtrain = []
Xtest = []
clfs = []

Xtrain.append(X_train_0)
Xtrain.append(X_train_1)
Xtest.append(X_test_0)
Xtest.append(X_test_1)




tasks = [0,1]

for param_kernel in kernel:
    save_path = f'{path}/{param_kernel}/'
    for param_nu in nu:
        for param_gamma in gamma:            
            if(param_kernel!='poly'):
                print(f'kernel={param_kernel} - gamma={param_gamma} - nu={param_nu}')
                clfs = []
                for i,task in enumerate(tasks):
                    clfs.append(svm.OneClassSVM(nu=param_nu, kernel=param_kernel, gamma=param_gamma, verbose=True))
                    clfs[i].fit(Xtrain[task])
                    pkl_filename = f'{save_path}/svm_model_T{i}.pkl'
                    with open(pkl_filename, 'wb') as file:
                        pickle.dump(clfs[i], file)
                conf_matrix = compute_small_confusion_matrix(clfs)
                fig = plt.gcf()
                ax = plt.subplot()
                sns.heatmap(conf_matrix, annot=True, ax = ax, cmap="YlGnBu");
                ax.xaxis.set_ticklabels(['T0', 'T1']); ax.yaxis.set_ticklabels(['T0', 'T1']);
                plt.savefig(f'{save_path}/FINAL_T0T1_n{param_nu}_r{param_gamma}.png')
                plt.clf()                
            else:
                for param_degree in degree:
                    print(f'kernel={param_kernel} - gamma={param_gamma} - nu={param_nu} - d={param_degree}')
                    clfs = []
                    for i,task in enumerate(tasks):
                        clfs.append(svm.OneClassSVM(nu=param_nu, kernel=param_kernel, gamma=param_gamma, degree=param_degree, verbose=True))
                        clfs[i].fit(Xtrain[i])
                        pkl_filename = f'{save_path}/svm_model_T{i}.pkl'
                        with open(pkl_filename, 'wb') as file:
                            pickle.dump(clfs[i], file)
                    conf_matrix = compute_small_confusion_matrix(clfs)
                    fig = plt.gcf()
                    ax = plt.subplot()
                    sns.heatmap(conf_matrix, annot=True, ax = ax, cmap="YlGnBu");
                    ax.xaxis.set_ticklabels(['T0', 'T1']); ax.yaxis.set_ticklabels(['T0', 'T1']);
                    plt.savefig(f'{save_path}/FINAL_n{param_nu}_r{param_gamma}_d{param_degree}.png')
                    plt.clf()



