# Import
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

# Prediction / Training function
def predict(year):
	movie_data = pd.read_csv('data/best_movies_netflix.csv')
	X = movie_data.drop(columns=['title','number_of_votes','duration','main_genre','main_production'])
	y = movie_data['main_genre']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	model = DecisionTreeClassifier()
	model.fit(X_train, y_train)
    # output model
	joblib.dump(model, 'model/movie-predict.joblib')
    # load model instead of training everytime
    # model = joblib.load('model/music-predict.joblib')
    # output visualization
	tree.export_graphviz(model, out_file='visualization/movie-predict.dot', feature_names=['release_year','score'], class_names=sorted(y.unique()), label='all', rounded=True, filled=True)
	pred = model.predict(X_test)
	score = accuracy_score(y_test, pred) # AI accuracy score
	return model.predict([[str(year),movie_data['score'].max()]])

# I/O function
if __name__=="__main__":
	year = input("Year?")
	print("Well if I'm right",''.join(predict(year)).title(),"is the most successful genre of",year)