import category_encoders as ce
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class NameSplitter(BaseEstimator,TransformerMixin):
    def __init__(self):
        ## Used HashingEncoder due to high cardinality.
        ## Needed to add max_process=1 in order for the transform to work. Seen here: https://github.com/scikit-learn-contrib/categorical-encoding/issues/215
        self._title_hashing_encoder = ce.HashingEncoder(n_components=5) 
        self._last_name_hashing_encoder = ce.HashingEncoder() 
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        last_name_np = np.empty(shape=(len(X)), dtype=object)
        title_np = np.empty(shape=(len(X)), dtype=object)

        x_copy = np.array(X).flatten()

        for i in range(len(X)):
            last_name_np[i], rest_of_name = x_copy[i].split(',')
            last_name_np[i] = last_name_np[i].strip()
            title_np[i] = rest_of_name.split('.')[0].strip()

        last_name_hashed = self._last_name_hashing_encoder.fit_transform(last_name_np.reshape((-1,1)))
        title_hashed = self._title_hashing_encoder.fit_transform(title_np.reshape((-1,1)))

        self._create_title_categories(len(np.array(title_hashed)[0]))
        self._create_last_name_categories(len(np.array(last_name_hashed)[0]))

        last_and_title = np.c_[last_name_hashed, title_hashed]

        return last_and_title

    def _create_title_categories(self,length):
        self.title_categories_ = ["Title_" + str(i) for i in range(length)]
    def _create_last_name_categories(self,length):
        self.last_name_categories_ = ["LastName_" + str(i) for i in range(length)]