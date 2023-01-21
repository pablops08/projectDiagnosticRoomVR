!pip install --upgrade pip
!pip install pandas as pd
!pip install numpy as np
!pip install keras
!pip install tensorflow

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import keras



#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)

data=pd.read_csv('C:\\Users\\pablo\\OneDrive\\Documentos\\PGEE\\TFC\\RNA Final\\BaseDados\\Parâmetros_de_todos_para_validacao_1_ciclos_RGB.csv')
dataset=data[['Velocidade média (m/s)','Comprimento médio passada','Largura da passada','Comprimento médio de meio passo em metros','Ângulo médio de flexão do joelho esquerdo','Ângulo médio de flexão do joelho direito','Simetria do comprimento de passo','Ângulo extensão do quadril (°)','Ângulo médio de abertura das pernas durante a caminhada (°)','Cadência','Movimento']]
datasheet=dataset.values


import random
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.metrics import confusion_matrix

shuffle(datasheet)


valores_linhas=datasheet[:,:-1]
valores_coluna_saida=datasheet[:,-1:]

!pip install imblearn
!pip install seaborn

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
nr = NearMiss() #Undersampling
smt = SMOTE() #Oversampling
valores_linhas, valores_coluna_saida = smt.fit_sample(valores_linhas, valores_coluna_saida)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(valores_linhas, valores_coluna_saida, test_size=0.3, shuffle=True) # test_size é o percentual do dataset que vai ser usado nas amostras


##Salvar Base de Dados

pd.DataFrame(X_train).to_csv('csv_X_train.csv')
pd.DataFrame(X_test).to_csv('csv_X_test.csv')
pd.DataFrame(Y_train).to_csv('csv_Y_train.csv')
pd.DataFrame(Y_test).to_csv('csv_Y_test.csv')

##Usar Base de Dados Anterior


# AUC per-fold
auc = []

# Indices para treinamento e validação per-fold
train_Cross_1 = []
train_Cross_2 = []
train_Cross_3 = []
train_Cross_4 = []
train_Cross_5 = []
val_Cross_1 = []
val_Cross_2 = []
val_Cross_3 = []
val_Cross_4 = []
val_Cross_5 = []




# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Merge inputs and targets
inputs = X_train
targets = Y_train

from sklearn.metrics import roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from keras.optimizers import RMSprop
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
num_folds=5
class_weight = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 1.0,
}

kfold = KFold(n_splits=num_folds, shuffle=True)
es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=0.0001, patience=10)

fold_no = 1
for train, val in kfold.split(inputs, targets):  
    
    model = Sequential([
    keras.layers.Flatten(input_shape=(10,)),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dropout(.3),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dropout(.3),
    keras.layers.Dense(6, activation='softmax')
    ])  
   
    #Normalizar Train e Validation e Test---------------
    !pip3 install sklearn 
    import sklearn as sk
    from sklearn import preprocessing
    from joblib import dump
    min_max_escalar=preprocessing.StandardScaler()
    X_train_NORM=min_max_escalar.fit_transform(inputs[train])    
    
    dump(min_max_escalar, 'scaler_filename.joblib')
    from joblib import load
    scaler = load('scaler_filename.joblib')

    X_val_NORM=scaler.fit_transform(inputs[val])
    X_test_NORM=scaler.fit_transform(X_test)
    #-----------------------------------------------------   
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['mae','accuracy'])    
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', verbose=0.0001, patience=10)    
    historico = model.fit(X_train_NORM, targets[train],class_weight=class_weight,shuffle=True,batch_size=16, epochs=100,validation_data=(X_val_NORM, targets[val]),callbacks=[es])
    scores = model.evaluate(X_val_NORM, targets[val], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[2]} of {scores[2]*100}%')
    acc_per_fold.append(scores[2] * 100)
    loss_per_fold.append(scores[0])
    model.save('C:\\Users\\pablo\\OneDrive\\Documentos\\PGEE\\TFC\\RNA Final\\CurrentModel\\ModeloSalvoCrossValidation_{}'.format(fold_no), save_format='tf')
    load_model = tf.keras.models.load_model('C:\\Users\\pablo\\OneDrive\\Documentos\\PGEE\\TFC\\RNA Final\\CurrentModel\\ModeloSalvoCrossValidation_{}'.format(fold_no))  
    y_pred = load_model.predict(X_test_NORM)
    y_pred=(np.around(y_pred))
    y_Pred=[]
    for i in range(0,len(y_pred)):
        #print(np.argmax(y_pred[i]))
        y_Pred.append(np.argmax(y_pred[i]))
    print(classification_report(Y_test,y_Pred,digits=5))
    auc.append(roc_auc_score(Y_test, load_model.predict_proba(X_test_NORM), multi_class='ovr'))
    # Salvar Base de treinamento e validação
    pd.DataFrame(train).to_csv('csv_train_Cross_{}.csv'.format(fold_no))
    pd.DataFrame(val).to_csv('csv_val_Cross_{}.csv'.format(fold_no))
    #-------
    print(auc[-1])
    fold_no = fold_no + 1



    CATEGORIAS=["Time Up and Go","Em circulos", "Marcha em linha reta", "Elevação excessiva do calcanhar"," Assimetria de passo", "Circundação do pé"]


    # == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

##Carregar o melhor modelo
load_model_1 = tf.keras.models.load_model('C:\\Users\\pablo\\OneDrive\\Documentos\\PGEE\\TFC\\RNA Final\\CurrentModel\\ModeloSalvoCrossValidation_{}'.format(1))
load_model_2 = tf.keras.models.load_model('C:\\Users\\pablo\\OneDrive\\Documentos\\PGEE\\TFC\\RNA Final\\CurrentModel\\ModeloSalvoCrossValidation_{}'.format(2))
load_model_3 = tf.keras.models.load_model('C:\\Users\\pablo\\OneDrive\\Documentos\\PGEE\\TFC\\RNA Final\\CurrentModel\\ModeloSalvoCrossValidation_{}'.format(3))
load_model_4 = tf.keras.models.load_model('C:\\Users\\pablo\\OneDrive\\Documentos\\PGEE\\TFC\\RNA Final\\CurrentModel\\ModeloSalvoCrossValidation_{}'.format(4))
load_model_5 = tf.keras.models.load_model('C:\\Users\\pablo\\OneDrive\\Documentos\\PGEE\\TFC\\RNA Final\\CurrentModel\\ModeloSalvoCrossValidation_{}'.format(5))

##CROSS VALIDATION - END##

import matplotlib.pyplot as plt
import numpy as np

X_test_NORM=scaler.fit_transform(X_test)

y_pred_1 = load_model_1.predict(X_test_NORM)  #cross_val_predict(model, X_train, Y_train, cv = cv)
y_pred_2 = load_model_2.predict(X_test_NORM)  #cross_val_predict(model, X_train, Y_train, cv = cv)
y_pred_3 = load_model_3.predict(X_test_NORM)  #cross_val_predict(model, X_train, Y_train, cv = cv)
y_pred_4 = load_model_4.predict(X_test_NORM)  #cross_val_predict(model, X_train, Y_train, cv = cv)
y_pred_5 = load_model_5.predict(X_test_NORM)  #cross_val_predict(model, X_train, Y_train, cv = cv)

y_pred = y_pred_1
for i in range(0,len(y_pred_1)):
    for j in range(0,6):
        y_pred[i][j] = (y_pred_1[i][j] + y_pred_2[i][j] + y_pred_3[i][j] + y_pred_4[i][j] + y_pred_5[i][j])/5
y_pred

#print(y_pred)
y_pred=(np.around(y_pred))
#print(y_pred)

y_Pred=[]
for i in range(0,len(y_pred)):
    #print(np.argmax(y_pred[i]))
    y_Pred.append(np.argmax(y_pred[i]))
#print(y_Pred)


import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, figsize=(24,24)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() /2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=18)

    plt.ylabel('True label',fontsize=18)
    plt.xlabel('Predicted label',fontsize=18)
    plt.tight_layout()

# Plot normalized confusion matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, y_Pred)
np.set_printoptions(precision=2)

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, y_Pred)
np.set_printoptions(precision=2)

plt.figure(figsize=(15, 10))
plot_confusion_matrix(cnf_matrix, classes=CATEGORIAS, normalize=True,
                      title='Normalized confusion matrix - CROSS VALIDATION - RGB - 1 CICLO- 1/1')
plt.savefig("Matrix Confusao Normalizada CROSS VALIDATION - 2Dense500 + Dropout 0.3'.png")
plt.show()

model.save('modelSaveH5.h5')

# Save Model in ONNX
!pip install keras2onnx
!pip install onnx
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
import onnx
import keras2onnx
onnx_model_name = 'modelSavedONNX.onnx'
model = load_model('modelSaveH5.h5')
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, onnx_model_name)

import numpy as np

MaxMin_Train = [[np.amax(X_train[:,0]),np.nanmin(X_train[:,0])]]
for i in range(1, X_train.shape[1]):
    MaxMin_Train.append([np.amax(X_train[:,i]),np.nanmin(X_train[:,i])])

    
