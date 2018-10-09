
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Flatten, Dropout
from keras.utils import np_utils
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold
import numpy as np
import hog
import matplotlib.pyplot as plt

# Load the data set
X, Y = hog.create_grad_arrays_from_file("data upgraded.csv")
X = np.array(X)
Y = np.array(Y)

seed = 7
#np.random.seed(seed)

# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
model = ""
for train, test in kfold.split(X, Y):
    model = Sequential()
    #model.add(Conv2D(15, kernel_size=(2, 2), strides=(1, 1),
                    #activation='relu',
                    #input_shape=(5, 4, 2),
                    #data_format="channels_last"))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    #model.add(Dropout(0.3))
    #model.add(Flatten())
    model.add(Dense(60, activation='relu', input_shape=(5, 4, 2)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

	# Fit the model
    history = model.fit(X[train], Y[train], epochs=150, batch_size=30, verbose=1, validation_data=(X[test], Y[test]))
	# evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print(scores)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# Loss Curves
plt.figure(figsize=[10,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[10,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

plt.show()

''' fig, ax = plt.subplots(1, 2, figsize = (12, 6), dpi = tdpi)
  ax = ax.ravel()
  
  for i in range(len(imgs)):
    img = imgs[i]
    ax[i].imshow(img)
  
  for i in range(per_row * per_col):
    ax[i].axis('off')

    plt.show()

plt.show()

t = np.arange(0.0, 2.0, 0.01)
s1 = np.sin(2*np.pi*t)
s2 = np.sin(4*np.pi*t)

plt.figure(1)
plt.subplot(211)
plt.plot(t, s1)
plt.subplot(212)
plt.plot(t, 2*s1)

plt.figure(2)
plt.plot(t, s2)

# now switch back to figure 1 and make some changes
plt.figure(1)
plt.subplot(211)
plt.plot(t, s2, 's')
ax = plt.gca()
ax.set_xticklabels([])

plt.show()
 '''
