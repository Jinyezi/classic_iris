import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import cross_validation as cval
from sklearn import grid_search
from sklearn import preprocessing


#Throw up a basic container class to hold data
class dataset():
  '''
  Container object for iris data
  '''
  def __init__(self):
    self.features = []
    self.classes = []

  def load_data(self,fpath):
    '''
    Given data file, reads information into this object
    '''
    with open(fpath,'r') as f:
      lines = f.readlines()
      for line in lines:
        line = line.strip()
        line = line.split(',')
        self.features.append([float(item) for item in line[:4]])
        self.classes.append(line[-1])

    #Scale data
    scale = preprocessing.StandardScaler()
    self.features = scale.fit_transform(self.features)


def testing_cycle(data,classes,gamma=0,C=0,loops=50):
  '''
  Takes data, does a stratified split (70/30) and trains/tests the provided
  classifier $loops times
  '''

  #Create random (but balanced) test/train groups
  splits = cval.StratifiedShuffleSplit(classes,n_iter=loops,test_size=0.3)

  scores = []
  for train_indices,test_indices in splits:

    classifier = svm.SVC(gamma=gamma,C=C,kernel='rbf')

    train_data    = [data[i] for i in train_indices]
    train_classes = [classes[i] for i in train_indices]
    test_data     = [data[i] for i in  test_indices]
    test_classes = [classes[i] for i in  test_indices]

    #train_data = np.array(train_data)
    #Train model
    classifier.fit(train_data,train_classes)
    #Get predictions
    predictions = classifier.predict(test_data)
    score = sum([1 for i,t in zip(predictions,test_classes) if i==t])/float(len(predictions))
    scores.append(score)

  print 'Average score: ', np.mean(scores)
  print 'Best score: ', max(scores)
  print '\n\n\n'
  plt.hist(scores)
  plt.show()


#Load in data
iris = dataset()
iris.load_data('.\\dataset\\iris.data')

#I want to approach this with a basic RBF SVM classifier
#Need to find ideal parameters for C,gamma in the classifier
#Perform a gridsearch to find good values
gammas = np.logspace(-9, 3, 13)
cvals  = np.logspace(-2, 10, 13)
param_grid = dict(gamma=gammas,C=cvals)
splits = cval.StratifiedShuffleSplit(iris.classes, n_iter=10, test_size=0.3)
grid = grid_search.GridSearchCV(svm.SVC(), param_grid=param_grid, cv=splits)
grid.fit(iris.features,iris.classes)

#Make a picture of the grid search
scores = [x[1] for x in grid.grid_scores_]
scores = np.array(scores).reshape(len(gammas), len(cvals))
plt.imshow(scores,interpolation='none')
plt.xticks(np.arange(len(gammas)), gammas, rotation=45)
plt.yticks(np.arange(len(cvals)), cvals)
plt.xlabel('gamma')
plt.ylabel('C')
plt.title('Model Performance')
plt.colorbar()
plt.show()

#Grab best model parameters
gamma,C = grid.best_params_['gamma'],grid.best_params_['C']

#Feed into train function
testing_cycle(iris.features,iris.classes,gamma=gamma,C=C)
