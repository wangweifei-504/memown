import pickle
name="machine-1-1_test.pkl"
name="MSL_train.pkl"
f=open(name,'rb')
content=pickle.load(f)
print(content)
