from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal,randint
from sklearn.externals import joblib
import pandas as pd
import TitanicPipeline
titanic_data = pd.read_csv("train.csv")

pipeline = TitanicPipeline.TitanicPipeline()
preprocessed_titanic_data = pipeline.fit_transform(titanic_data)

preprocessed_titanic_data_df = pd.DataFrame(preprocessed_titanic_data,columns=pipeline.categories)

X_train = preprocessed_titanic_data_df.drop(['Survived'], axis=1)
y_train = preprocessed_titanic_data_df["Survived"]

svm_clf = SVC()

grid_param = {
    'kernel' : ['linear','rbf'],
    'degree' : randint(1,100),
    'C' : [1,10,100],
    'gamma' : [1,0.1,0,0.01,0.001]
}

gd_svm = RandomizedSearchCV(svm_clf,grid_param,cv=10,n_jobs=8,n_iter=50, verbose=1)

gd_svm.fit(X_train,y_train)

print (gd_svm.best_params_)
print (gd_svm.best_score_)

## Best : 81.82%
joblib.dump(gd_svm.best_estimator_, "svm_classifier.pkl")