import numpy as np
import pandas as pd
import sklearn.svm
from sklearn.model_selection import train_test_split



def fitness(sol, total_features, label, split = 0.2):
  feature =  reduce_features(sol, total_features)
  xtrain, xtest, ytrain, ytest = train_test_split(feature, label, test_size = split, random_state = 4 )
  SVM_classifier = sklearn.svm.SVC(kernel='rbf',gamma='scale',C=5)
  SVM_classifier.fit(xtrain, ytrain)
  predictions = SVM_classifier.predict(xtest)
  clf_value = classification_accuracy(ytest, predictions)

  val = 1-clf_value

  #in case of multi objective  []

  set_cnt=sum(sol)
  set_cnt=set_cnt/np.shape(sol)[0]
  val=omega*val+(1-omega)*set_cnt
  return val


binarised_vector = np.random.randint(low=0, high=2, size=pop_shape)

def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features


def SOPF(sol_vector, features):
  binarised_vector = sol_vector
  for i in range(len(binarised_vector)):
    temp_vector = binarised_vector
    if (binarised_vector[i] == 1):
      binarised_vector[i] = 0
    else:
      binarised_vector[i] = 1
    if (fitness(temp_vector, features, label)<fitness(binarised_vector, features, label)):
      resultant_vector = temp_vector
    else:
      resultant_vector = binarised_vector

    return resultant_vector


def AWCM(population):
  weighted_solution = []
  for i in range(len(population)):
    temp_weighted_solution = []
    for j in range(len(population[i])):
      candidate_solution = population[i][j]
      Acc_d = fitness(candidate_solution, data_inputs, data_outputs, 0.5)
      temp_goodness = []
      for k in range(len(population[i][j])):
        temp_goodness.append(population[i][j][k]*Acc_d)
      temp_weighted_solution.append(temp_goodness)
    weighted_solution.append(temp_weighted_solution)
  
  summed_solution = []
  for i in range(len(population)):
    for j in range(len(weighted_solution[i])):
      if (j==0):
        sum_list = weighted_solution[i][j]
      else:
        sum_list = [(a + b) for a, b in zip(sum_list, weighted_solution[i][j])]
    summed_solution.append(sum_list)

  mean = []
  for i in range(len(summed_solution)):
    sum = 0
    for j in range(len(summed_solution[i])):
      sum += summed_solution[i][j]/len(summed_solution[i])
    mean.append(sum)

  final_population = []
  for i in range(len(mean)):
    final_solution = []
    for j in range(len(summed_solution[i])):
      if (summed_solution[i][j] >= mean[i]):
        final_solution.append(1)
      else:
        final_solution.append(0)
    final_population.append(final_solution)

  return final_population
