import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import statsmodels.api as sm
import random
import datetime
import pickle

myseed = 69
np.random.seed(myseed)

def draw_scatter_pic(data):
    plt.figure(figsize=(10,8))

    plt.subplot(321) 
    plt.title('Sepal Length v.s Sepal Width')
    plt.scatter(data[:,0], data[:,1], c='#1f77b4')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')

    plt.subplot(322) 
    plt.title('Sepal Length v.s Petal Length')
    plt.scatter(data[:,0], data[:,2])
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Length')

    plt.subplot(323) 
    plt.title('Sepal Length v.s Petal Width')
    plt.scatter(data[:,0], data[:,3])
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Width')

    plt.subplot(324) 
    plt.title('Sepal Width v.s Petal Length')
    plt.scatter(data[:,1], data[:,2])
    plt.xlabel('Sepal Width')
    plt.ylabel('Petal Length')

    plt.subplot(325) 
    plt.title('Sepal Width v.s Petal Width')
    plt.scatter(data[:,1], data[:,3])
    plt.xlabel('Sepal Width')
    plt.ylabel('Petal Width')

    plt.subplot(326) 
    plt.title('Petal Length v.s Petal Width')
    plt.scatter(data[:,2], data[:,3])
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')

    plt.tight_layout()
    plt.show()

def draw_error_plot(setosa_correct, versicolor_correct, virginica_correct, all_setosa, all_versicolor, all_virginica):
    '''
    0 - Setosa、 1 - Versicolor、2 - Virginica ，畫 error bar
    '''
    setosa_correct = np.array(setosa_correct)
    setosa_mean = np.mean(setosa_correct, axis = 0)
    setosa_std = np.std(setosa_correct, axis = 0)

    versicolor_correct = np.array(versicolor_correct)
    versicolor_mean = np.mean(versicolor_correct, axis = 0)
    versicolor_std = np.std(versicolor_correct, axis = 0)

    virginica_correct = np.array(virginica_correct)
    virginica_mean = np.mean(virginica_correct, axis = 0)
    virginica_std = np.std(virginica_correct, axis = 0)

    labels = ['setosa', 'versicolor', 'virginica']
    std = [setosa_std, versicolor_std, virginica_std]
    y = [setosa_mean, versicolor_mean, virginica_mean]
    x = np.arange(len(labels))

    plt.title('setosa : {:.3f}/{}   versicolor : {:.3f}/{}   virginica : {:.3f}/{}'
              .format(setosa_mean, all_setosa, versicolor_mean, all_versicolor, virginica_mean, all_virginica))
    plt.ylabel('pred correct mean')
    plt.bar(x, y, color=['blue','orange','green'], yerr=std, tick_label=labels)
    plt.show()

iris = datasets.load_iris()
data = iris.data
target = iris.target
# draw_scatter_pic(data)

Min_max_normalization = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_normalization = Min_max_normalization.fit_transform(data)

items = 1
final = []
setosa_correct = []
versicolor_correct = []
virginica_correct = []
all_setosa = 0
all_versicolor = 0
all_virginica = 0
while True:
    now = datetime.datetime.now()
    print('{:0>3d}/100 Experiments: Start at {:}'.format(items, now))
    num = random.randint(0, 1000)
    data_normalization_const = sm.add_constant(data_normalization)
    train_data, test_data, train_label, test_label = train_test_split(data_normalization_const, target, 
                                                    test_size=0.2, random_state = num, shuffle = True)
    model = sm.OLS(train_label, train_data)
    ols_fit_line = model.fit()
    # print(ols_fit_line.summary())
    pred = ols_fit_line.predict(test_data)

    pred_ans = []
    pred_setosa = 0
    pred_versicolor = 0 
    pred_virginica = 0
    for i in range(pred.shape[0]):
        if pred[i] <= 0.5:
            pred_ans.append(0)
            pred_setosa += 1
        elif 0.5 < pred[i] < 1.5 :
            pred_ans.append(1)
            pred_versicolor += 1
        elif 1.5 <= pred[i] :
            pred_ans.append(2)
            pred_virginica += 1
                  
    all_setosa += pred_setosa
    all_versicolor += pred_versicolor
    all_virginica += pred_virginica
    
    count = 0
    pred_setosa_correct = 0
    pred_versicolor_correct = 0 
    pred_virginica_correct = 0
    for idx in range(len(pred_ans)):       
        if test_label[idx] == pred_ans[idx]: # 檢查預測跟原先標籤是否一樣
            count += 1
            if test_label[idx] == 0:
                pred_setosa_correct += 1
            elif test_label[idx] == 1:
                pred_versicolor_correct += 1
            elif test_label[idx] == 2:
                pred_virginica_correct += 1
        else:
            continue
    final.append(count/len(pred_ans)) # 測試集有多少比例預測正確，一一添加至列表
    setosa_correct.append(pred_setosa_correct/pred_setosa)
    versicolor_correct.append(pred_versicolor_correct/pred_versicolor)
    virginica_correct.append(pred_virginica_correct/pred_virginica)
    if items == 100:
        break
    else:
        items += 1
# print(final)

## 算 100 回的 mean、std
final = np.array(final)
ols_correct_mean = np.mean(final, axis=0)
ols_correct_std = np.std(final, axis=0)

## 畫 error bar
draw_error_plot(setosa_correct, versicolor_correct, virginica_correct, all_setosa, all_versicolor, all_virginica)

## 儲存
combine = {'data': data, 'target': target, 'ols_correct_mean': ols_correct_mean, 'ols_correct_std': ols_correct_std}
with open('summer_training_Q2.pickle', 'wb') as f:
    pickle.dump(combine, f)