#Training the mnist digit set using h2o library
import os
import h2o
h2o.init(nthreads = -1,max_mem_size = 3G)
import h2o.grid

data_path = ""
fileTrain = ""
fileTest = ""
train = h2o.import_file(os.path.join(data_path,fileTrain))
test = h2o.import_file(os.path.join(data_path,fileTest))
x = range(0,784)
y = 784
train[[y]] = train[[y]].asfactor() #Labels of training data
test[[y]] = test[[y]].asfactor() #Labels of testing data

valid,train = train.split_frame([1.0/6.0])
m = h2o.estimators.H2ORandomForestEstimator(model_id = 'RF_default')
m.train(x,y,train,validation_frame = valid)

