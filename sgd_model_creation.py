from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib
from scipy.stats import expon, reciprocal
import TitanicPipeline
import pandas as pd
import numpy as np

titanic_data = pd.read_csv("train.csv")

pipeline = TitanicPipeline.TitanicPipeline()
preprocessed_titanic_data = pipeline.fit_transform(titanic_data)

preprocessed_titanic_data_df = pd.DataFrame(preprocessed_titanic_data,columns=pipeline.categories)

X_train = preprocessed_titanic_data_df.drop(['Survived'], axis=1)
y_train = preprocessed_titanic_data_df["Survived"]

sgd_clf = SGDClassifier(random_state=42,max_iter=5000)

grid_param = {
    'loss' : ['hinge','modified_huber','log','squared_hinge','perceptron'],
    'penalty' : ['l1','l2','elasticnet'],
    'learning_rate' : ['constant','optimal','invscaling','adaptive'],
    'eta0' : expon(scale=4)
}

gd_sgd = RandomizedSearchCV(sgd_clf,grid_param,scoring='accuracy',cv=10,n_jobs=8,n_iter=50)

gd_sgd.fit(X_train,y_train)

print(gd_sgd.best_params_)
print(gd_sgd.best_score_)

best_sgd = SGDClassifier()
best_sgd.set_params(**gd_sgd.best_params_)
best_sgd.fit(X_train,y_train)

## Best : 81.37%
joblib.dump(best_sgd, "sgd_classifier.pkl")

