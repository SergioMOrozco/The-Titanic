from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib
from scipy.stats import randint
import TitanicPipeline
import pandas as pd

titanic_data = pd.read_csv("train.csv")

pipeline = TitanicPipeline.TitanicPipeline()
preprocessed_titanic_data = pipeline.fit_transform(titanic_data)

preprocessed_titanic_data_df = pd.DataFrame(preprocessed_titanic_data,columns=pipeline.categories)

X_train = preprocessed_titanic_data_df.drop(['Survived'], axis=1)
y_train = preprocessed_titanic_data_df["Survived"]

knn_clf = KNeighborsClassifier()

grid_param = {
    'weights' : ['uniform','distance'],
    'algorithm' : ['ball_tree','kd_tree','brute','auto'],
    'n_neighbors' : randint(1,100)
}

gd_knn = RandomizedSearchCV(knn_clf,grid_param,cv=10,n_jobs=8,n_iter=50, scoring='accuracy')
gd_knn.fit(X_train,y_train)
print(gd_knn.best_params_)
print(gd_knn.best_score_)

## Best : 81.93%
joblib.dump(gd_knn.best_estimator_, "knn_classifier.pkl")