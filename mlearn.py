#import
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

# prediction / training function
def predict(age,gender):
	music_data = pd.read_csv('music.csv')
	X = music_data.drop(columns=['genre'])
	y = music_data['genre']
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8)
	model = DecisionTreeClassifier()
	model.fit(X_train,y_train)
	# output model
	joblib.dump(model,'music-predict.joblib')
	# load model instead of training everytime
	#model = joblib.load('music-predict.joblib')
	# output visualization
	tree.export_graphviz(model,out_file='music-predict.dot',feature_names=['age','gender'],class_names=sorted(y.unique()),label='all',rounded=True,filled=True)
	pred = model.predict(X_test)
	score = accuracy_score(y_test,pred) # AI accuracy score
	return model.predict([[str(age),str(gender)]])

# input/output function
def take():
	f=['FEMALE','F','FEM','GIRL','WOMAN']
	m=['MALE','M','MASC','GUY','MAN']
	age = input("Age? ")
	gen = input("Gender? ")
	if gen.upper() in f:
		gender = 0
	elif gen.upper() in m:
		gender = 1
	print("If I know people well i think you like",''.join(predict(age,gender)))

if __name__=="__main__":
	take()