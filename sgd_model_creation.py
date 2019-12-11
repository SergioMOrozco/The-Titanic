from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import TitanicPipeline
import pandas as pd
import numpy as np

titanic_data = pd.read_csv("train.csv")

pipeline = TitanicPipeline.TitanicPipeline()
preprocessed_titanic_data = pipeline.fit_transform(titanic_data)

preprocessed_titanic_data_df = pd.DataFrame(preprocessed_titanic_data,columns=pipeline.categories)

X_train = preprocessed_titanic_data_df.drop(['Survived'], axis=1)
y_train = preprocessed_titanic_data_df["Survived"]

sgd_clf = SGDClassifier(random_state=42,)

grid_param = {
    'loss' : ['hinge','modified_huber','log'],
    'penalty' : ['l1','l2','elasticnet']
}

gd_sr = GridSearchCV(sgd_clf,grid_param,scoring='accuracy',cv=5,n_jobs=8)

gd_sr.fit(X_train,y_train)

print(gd_sr.best_params_)
print(gd_sr.best_score_)

