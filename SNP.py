
'''
Shubham Gulia
Project(Machine Learning)
'''

import random
from sklearn import svm
import os
import sys

"""
First split the data 70:30 and then perform feature selection
"""

#### Chi-Square for feature selection ####


def chi_square(data, n_features):

    label = [row[-1] for row in data]
    rows = len(data)
    cols = len(data[0]) - 1
    T = []
    for j in range(0, cols):
        ct = [[1, 1], [1, 1], [1, 1]]

        for i in range(0, rows):
            if label[i] == 0:
                if data[i][j] == 0:
                    ct[0][0] += 1
                elif data[i][j] == 1:
                    ct[1][0] += 1
                elif data[i][j] == 2:
                    ct[2][0] += 1
            elif label[i] == 1:
                if data[i][j] == 0:
                    ct[0][1] += 1
                elif data[i][j] == 1:
                    ct[1][1] += 1
                elif data[i][j] == 2:
                    ct[2][1] += 1

        col_totals = [sum(x) for x in ct]
        row_totals = [sum(x) for x in zip(*ct)]
        total = sum(col_totals)
        exp_value = [[(row * col) / total for row in row_totals] for col in col_totals]
        sqr_value = [[((ct[i][j] - exp_value[i][j]) ** 2) / exp_value[i][j] for j in range(0, len(exp_value[0]))] for i in range(0, len(exp_value))]
        chi_2 = sum([sum(x) for x in zip(*sqr_value)])
        T.append(chi_2)
    indices = sorted(range(len(T)), key=T.__getitem__, reverse=True)
    idx = indices[:n_features]
    return idx




##### Extracting top 15 features ######


def feature_extraction(X, cols):
    V = []
    columns = list(zip(*X))
    for i in cols:
        V.append(columns[i])
    V = list(zip(*V))
    return V


#### Create a random subsample from the dataset with replacement


def subsample(dataset, labels, ratio):
    sampleData = []
    sampleLabel = []
    n_sample = round(len(dataset) * ratio)

    row_index = [random.randint(0, n_sample - 1) for _ in range(0, n_sample)]

    # while len(sample) < n_sample:
    #     index = random.randrange(len(dataset))
    #     sample.append(dataset[index])
    #     sampleLabel.append(labels[index])
    # return sample, sampleLabel

    for i in row_index:
        sampleData.append(dataset[i])
        sampleLabel.append(labels[i])
    return sampleData, sampleLabel


#### Build SVM model #####


def buildSVM(sampleData, sampleLabel):
    model = svm.SVC(kernel='linear', C=1)
    model.fit(sampleData, sampleLabel)
    model.score(sampleData, sampleLabel)
    return model


#### Make a prediction with a list of bagged SVM models ####


def bagging(model, row):
    predictions = [list(m.predict([row])) for m in model]
    pred = [i[0] for i in predictions]
    return max(set(pred), key=pred.count)


#### Calculate accuracy ####


def accuracy(actual, predicted):
    count = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            count += 1
    return count / float(len(actual)) * 100.0


########################################
######## Read Train Data
########################################

print('Reading data...........')

datafile = sys.argv[1]
f = open(datafile, 'r')
data = []
i = 0
l = f.readline()
while (l != ''):
    a = l.split(",")
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    data.append(l2)
    l = f.readline()

print('Training data:', len(data))

######################################
### Read Train labels
######################################

datafile = sys.argv[2]
f = open(datafile, 'r')
label = []
i = 0
l = f.readline()
while (l != ''):
    a = l.split(",")
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    label.append(l2)
    l = f.readline()

print('Training labels:', len(label))

########################################
######## Read Test Data
########################################

datafile = sys.argv[3]
f = open(datafile, 'r')
realtest = []
i = 0
l = f.readline()
while (l != ''):
    a = l.split(",")
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    realtest.append(l2)
    l = f.readline()

print('Test data:', len(realtest))


trainLabels, index = zip(*label)
trainLabels = list(trainLabels)

### Combine trainData and trainLabels ###

for i in range(0, len(data)):
    data[i].append(trainLabels[i])

del (index, i, label)
print('Data loaded')

########################################################
###### Split Data = 70% training data and 30% test data
########################################################

print('Splitting Data...........')

ratio = 0.70
dataLength = len(data)
size = int(dataLength * ratio)

index_train = random.sample(range(dataLength), size)

train_subset = []
test_subset = []

for i in range(len(data)):
    if i in index_train:
        train_subset.append(data[i])
    else:
        test_subset.append(data[i])

'''
train_subset contains 70% of the data and
test_subset contains 30% of the data
'''

print('Splitting done!!! 70% training data and 30% test data')

#### Feature selection ####

print('Feature Selection ................')

featureCol = chi_square(train_subset, 15)
realTestData = feature_extraction(realtest, featureCol)
featureCol.append(len(train_subset[0]) - 1)
newTrainingData = feature_extraction(train_subset, featureCol)
newTestData = feature_extraction(test_subset, featureCol)

print('Feature Selection done!')

###### SVM Algorithm ######

newTrainingData = [list(elem) for elem in newTrainingData]
newTestData = [list(elem) for elem in newTestData]
realTestData = [list(elem) for elem in realTestData]

newTrainingLabel = [row[-1] for row in newTrainingData]
for row in newTrainingData:
    del (row[-1])

newTestLabel = [row[-1] for row in newTestData]
for row in newTestData:
    del (row[-1])

### Bootstrap aggregating (Bagging) with SVM ####

print('Bagging.........')
bags = 50
models = [] * bags

for _ in range(0, bags):
    sampleData, sampleLabel = subsample(newTrainingData, newTrainingLabel, 1)
    # SVM linear Model
    m = buildSVM(sampleData, sampleLabel)
    models.append(m)

print("Number of SVM model bags created =", len(models))

predictions = []
for row in newTestData:
    predictions.append(bagging(models, row))

model_accuracy = accuracy(newTestLabel, predictions)

print('\nACCURACY OF THE MODEL IS', model_accuracy, '%\n')

'''
Predicting labels for the given test data
'''
print('Predicting labels for given test dataset.......')

realTestDataPredict = [bagging(models, row) for row in realTestData]

file_path = os.path.dirname(os.path.abspath('__file__'))

#### The predicted labels are saved in file called testlabels #####

path = file_path + '/testlabels'

### Print the predicted labels ####

f = open(path, "a")
w = 0
for i in realTestDataPredict:
    f.write(' '.join(map(str, [int(i)])) + ' ' + str(w) + "\n")
    w += 1
f.close()

del (featureCol[-1])
print('Number of features =', str(len(featureCol)))
print('Feature columns =', str(featureCol))

score = model_accuracy / (100 * len(featureCol))
print('Score of the output =', str(score))
print('Predicted labels file Saved in: ', path)

