from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import TitanicPipeline
import pandas as pd

titanic_data = pd.read_csv("train.csv")


pipeline = TitanicPipeline.TitanicPipeline()
preprocessed_titanic_data = pipeline.fit_transform(titanic_data)

preprocessed_titanic_data_df = pd.DataFrame(preprocessed_titanic_data,columns=pipeline.categories)

X_train = preprocessed_titanic_data_df.drop(['Survived'], axis=1)
y_train = preprocessed_titanic_data_df["Survived"]

nb_clf = BernoulliNB()

nb_clf.fit(X_train,y_train)

## Averages around 77%
joblib.dump(nb_clf, "naive_bayes_classifier.pkl")