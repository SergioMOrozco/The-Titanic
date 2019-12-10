import numpy as np
import category_encoders as ce
import NameSplitter
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class TitanicPipeline(BaseEstimator,TransformerMixin):

    def __init__(self, using_neural_net=False):

        self._num_list = ['Pclass','Age','SibSp','Parch','Fare']

        self._embarked_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent", missing_values=np.nan)), ## gets most frequently used value and replaces nan's with that value
        ('one_hot', OneHotEncoder()), ## one hot encodes this feature
        ])

        self._cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent", missing_values=np.nan)), ## gets most frequently used value and replaces nan's with that value
        ('ordinal_encoder', OrdinalEncoder()), ## Replaces each string with an integer [0,n_categories-1]
        ('feature_scaler', MinMaxScaler())
        ])

        self._num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean", missing_values=np.nan)), ## gets most frequently used value and replaces nan's with that value
        ('feature_scaler', MinMaxScaler()), ## Replaces each string with an integer [0,n_categories-1]
        ])

        self._preprocessor = ColumnTransformer([
            ("numerical",self._num_pipeline, self._num_list ),
            ("embarked", self._embarked_pipeline, ['Embarked'] ),
            ("name",NameSplitter.NameSplitter(),['Name']),
            ("cat", self._cat_pipeline, ['Ticket','Sex','Survived']),
        ])

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        transformed_data = self._preprocessor.fit_transform(X,y)

        self.categories = self._num_list
        self.categories.extend(self._preprocessor.transformers_[1][1][1].categories_[0].tolist())
        self.categories.extend(self._preprocessor.transformers_[2][1].last_name_categories_)  
        self.categories.extend(self._preprocessor.transformers_[2][1].title_categories_)  
        self.categories.extend(['Ticket','Sex','Survived'])
        return transformed_data