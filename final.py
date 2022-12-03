#Griffen Agnello, Alexis Brown    CS483   Final Project   Natural Language Processing
import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class Preprocessor( BaseEstimator, TransformerMixin ):    
    def fit( self, xs, ys = None ): return self    
    def transform( self, xs ):        
        def drop_newline( t ): return re.sub( '\n', ' ', t )
        #Remove Hyperlinks and the single case of "http" but no link
        def drop_hyperlinks( t ): return re.sub(r"http?://[^ ]*|https?://[^ ]*|http", '', t)        
        #Remove hyphens, quotations
        def drop_extra_punctuation( t ): return re.sub( r"\'|\-", '', t )        
        def spacify_non_letter_or_digit( t ): return re.sub( '\W', ' ', t )        
        def combine_spaces( t ): return re.sub( '\s+', ' ', t )
        def combine_numbers( t ): return re.sub( r'[0-0] [0-9]', '', t )
        transformed = xs.str.lower()         
        transformed = transformed.apply(drop_hyperlinks)
        transformed = transformed.apply( drop_newline )  
        transformed = transformed.apply( drop_extra_punctuation )        
        transformed = transformed.apply( spacify_non_letter_or_digit )        
        transformed = transformed.apply( combine_spaces ) #optional
        transformed = transformed.apply( combine_numbers )
        return transformed

class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, xs, xy, **params):
        return self

    # actually perform the selection
    def transform(self, xs):
        return xs[self.columns]




def main():
    data=pd.read_csv("./csv/train.csv")    
    ys=data['disaster']
    xs=data.drop(columns = ['disaster'])

    # use this instead of 'english' when testing if certain SWs improve performance
    ourStopWords = ['the', 'a', 'an', 'that']

    
    gridSelect = {
        'column_select__columns': [['keyword', 'location', 'text']]
    }

    # tree => n_estimators=num trees in forest (100 def), max_depth of each tree (none by def)
    gridForestParameters = {
        "max_depth" : [2, 10, 20, 50],
        "max_features" : ["log2", "sqrt"], # num fs to consider when looking for best split. Def sqrt
        "n_estimators" : [50, 100]
    }


    gridVectorParameters = {
        "strip_accents" : [None, "unicode", "ascii"],
        "stop_words" : ["english", ourStopWords],
        "ngram_range" : [(1, 1), (1, 3), (1, 4)]
    }

    params = [gridSelect, gridVectorParameters, gridForestParameters]

    steps=[
        ('column_select', SelectColumns('column_select__columns')),
        ('preprocess', Preprocessor()),
        ('vectorize', CountVectorizer()),
        ('forest', RandomForestClassifier())
    ]

    print("\nFiltered\n")
    pipe=Pipeline(steps)
    search=GridSearchCV(pipe, params, n_jobs=-1)
    pipe.fit(xs,ys)
    #print(search.best_score_)
    #print(search.best_estimator_)
    with pd.option_context('display.max_rows', None,
        'display.max_columns', None,
        'display.width', None):
        # print(data['text'].to_csv())
        print(pipe.transform(data['text']).to_csv())
    
main()












