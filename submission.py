from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import TitanicPipeline

submission_data = pd.read_csv("test.csv")

ids = submission_data["PassengerId"]
ids = ids.astype(int)

pipeline = TitanicPipeline.TitanicPipeline(is_testing=True)
preprocessed_submission_data = pipeline.fit_transform(submission_data)

X_test = preprocessed_submission_data

knn = joblib.load("knn_classifier.pkl")

predictions = pd.DataFrame(knn.predict(X_test)).astype(int)

results = pd.concat([ids,predictions],axis=1)

print(results)

results.to_csv('submission.csv',index=False,header=["PassengerId","Survived"])