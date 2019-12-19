import category_encoders as ce
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer

class AgeSplitter(BaseEstimator,TransformerMixin):
    def __init__(self, age_threshold=15,strategy="most_frequent"):
        self.age_threshold = age_threshold
        self.strategy = strategy
        self.age_categories_ = ['Age > ' + str(self.age_threshold)]
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        x_copy = np.array(X).flatten()
        imputer = SimpleImputer(strategy=self.strategy, missing_values=np.nan)
        ret_x = imputer.fit_transform(x_copy.reshape((-1,1)))
        ret_x = np.array([int(i > self.age_threshold) for i in ret_x.flatten()]).reshape((-1,1))

        return ret_x