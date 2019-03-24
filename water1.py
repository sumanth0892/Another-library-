import os
import h2o
h2o.init()
import h2o.grid
data_path = ""
file = ""
fname = os.path.join(data_path,file)
data = h2o.import_file(data)
factorsList = ['X6','X8']
data[factorsList] = data[factorsList].asfactor() #Convert into enumerated factors
train,test = data.split_frame([0.8])
train.describe()
test.describe()
x = [] #Features to use for training data
y = "" #Feature to predict using the tree

#Create a Random forest model with default settings and 10 fold validation split
m = h2o.estimators.H2ORandomForestEstimator(model_id='RF_default',nfolds=10)
m.train(x,y,train)
m.model_performance(test)

#Grid search to test several variants of the model
g = h2o.grid.H2OGridSearch(h2o.estimators.H2ORandomForestEstimator(nfolds=10)
                           hyper_params = {})
#The hyper-params are filled such as number of trees and minimum rows
g.train(x,y,train)
g.model_performance(test)

