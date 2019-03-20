#small project to recognise a number
import inline as inline
import matplotlib.pyplot as plt
#matplotlib inline
from sklearn.datasets import load_digits
digits = load_digits()
import random

import pylab as pl
'''

pl.gray()
pl.matshow(digits.images[0])
pl.show()
'''
#print(digits.images[0])
'''
images_and_labels = list(zip(digits.images,digits.target))
plt.figure(figsize=(5,5))

for index, (image,label) in enumerate(images_and_labels[:15]):
    plt.subplot(3,5,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('%i'%label)
'''
import random
from sklearn import ensemble,neighbors

#define variables
n_samples = len(digits.images)
x = digits.images.reshape((n_samples, -1))
y = digits.target

#seed
random.seed(1)

#Create random indices
sample_index = random.sample(range(len(x)),len(x)//5,)
valid_index = [i for i in range(len(x)) if i not in sample_index]

#Sample and validation images
sample_images = [x[i] for i in sample_index]
valid_images = [x[i] for i in valid_index]

#Sample and validation targets
sample_targets = [y[i] for i in sample_index]
valid_targets = [y[i] for i in valid_index]

#Using random tree classifier
classifier = ensemble.RandomForestClassifier(random_state=1)

#Using Knearest neighbors
clf = neighbors.KNeighborsClassifier()

#Fit model with sample data
classifier.fit(sample_images,sample_targets)

from sklearn import metrics

#Attempt to predict validation targets
#score = classifier.score(valid_images,valid_targets)
pred_target = classifier.predict(valid_images)
print('Random Tree Classifier:\n')
#print('Score\t'+str(score))
print("Accuracy: ",metrics.accuracy_score(valid_targets,pred_target))
'''
i=100

pl.gray()
pl.matshow(digits.images[i])
pl.show()
print(digits.target[i])
'''
#359 in sample  1438 in validation
print(sample_index)
print(valid_index)
print(len(pred_target))
k=0
error = []
total = len(valid_index)
error_count = 0
for i in valid_index:
    print(i)
    print("Actual: "+str(digits.target[i])+"Prediction :"+str(pred_target[k]))
    if(digits.target[i]!=pred_target[k]):
        error.append(digits.target[i])
        error_count = error_count + 1
    k = k + 1

print("Accuracy :" + str((total - error_count)/total))
print("0 :"+str(error.count(0))+",1 :"+str(error.count(1))+",2 :"+str(error.count(2))+",3 :"+str(error.count(3))+
      ",4 :"+str(error.count(4))+",5 :"+str(error.count(5))+",6 :"+str(error.count(6))+",7 :"+str(error.count(7))+
      ",8 :"+str(error.count(8))+",9 :"+str(error.count(9)))
#classifier.predict(y[[i]])

'''
k=0
for i in valid_index  :
    print(i)
  #  pl.gray()
  #  pl.matshow(valid_images)
  #  pl.show()
    print("Prediction: "+str(valid_targets[k])+" Actual: "+str(digits.target[i]))
    if(valid_targets[k]!=digits.target[i]):
        break
    k = k + 1
'''
