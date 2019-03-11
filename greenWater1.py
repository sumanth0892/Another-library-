import os
import h2o
h2o.init()
path = "/Users/sumanthpareekshit/Downloads/Coding_practice"
fname = os.path.join(path,"/Users/sumanthpareekshit/Downloads/Coding_practice/greenWater.xlsx")
data = h2o.import_file(fname)
factors = ["X6","X8"]
data[factors] = data[factors].asfactor()
train,test = data.split_frame([0.8])
x = ['X1','X2','X3','X4','X5','X6','X7','X8']
y = "Y2"
train.describe(); test.describe()
train.cor().round(2)
test.cor().round(2)
m = h2o.estimators.H2ORandomForestEstimator(model_id=RF_defaults,nfolds=10)
m.train(x,y,train)
