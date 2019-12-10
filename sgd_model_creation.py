from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
import TitanicPipeline
import pandas as pd
import numpy as np

titanic_data = pd.read_csv("train.csv")

pipeline = TitanicPipeline.TitanicPipeline()
preprocessed_titanic_data = pipeline.fit_transform(titanic_data)

preprocessed_titanic_data_df = pd.DataFrame(preprocessed_titanic_data,columns=pipeline.categories)

X_train = preprocessed_titanic_data_df.drop(['Survived'], axis=1)
y_train = preprocessed_titanic_data_df["Survived"]
sgd_clf = SGDClassifier(random_state=42)
print(cross_val_score(sgd_clf,X_train,y_train,scoring="accuracy",cv=3))
