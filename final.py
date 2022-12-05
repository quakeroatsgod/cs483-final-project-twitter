#Griffen Agnello, Alexis Brown    CS483   Final Project   Natural Language Processing
import pandas as pd
import numpy as np
import re
import sys
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

#Class that cleans up and prepares the tweet text in the pipeline for the count vectorizer
class Preprocessor( BaseEstimator, TransformerMixin ):    
    def fit( self, xs, ys = None ): return self    
    def transform( self, xs ):        
        def drop_newline( t ): return re.sub( '\n', ' ', t )
        #Remove Hyperlinks and the single case of "http" but no link
        def drop_hyperlinks( t ): return re.sub(r"http?://[^ ]*|https?://[^ ]*|http", '', t)        
        #Removes hyphens, double quotations, commas, pound signs, periods and turns them into spaces
        def drop_extra_punctuation( t ): return re.sub( r"\"|\-|\,|\#|\.|\;[^;()]|\:[^:()]", ' ', t )   
        #Removes single quotations. Words like "I'm" will become "im"
        def drop_quotations(t): return re.sub(r"\'","",t)
        #Removes $, <, >, =, +, etc.
        def drop_special_characters(t): return re.sub(r"[\<\>\=\_\{\}\[\]\'\*\^\&\%\$\|\/\\]",' ',t)
        #Remove @username
        def drop_mentions(t): return re.sub(r"@[^ ]*", '',t)
        #Adds a space before ?
        def spacify_question_mark( t ): return re.sub('\?', ' ?', t )
        #Adds a space before !
        def spacify_exclamation_point( t ): return re.sub('\!', ' !', t )        
        def combine_spaces( t ): return re.sub( '\s+', ' ', t )
        #Cut down any repeating characters of 3 or more (like aaaaa) is reduced to length 2 (aa)
        def reduce_extra_characters( t ): return re.sub( r"(.)(\1{1})(\1+)", r"\1\2", t )
        # def reduce_extra_characters(t): return re.sub(r"",'',t)
        transformed = xs.str.lower()              \
            .apply(drop_hyperlinks)               \
            .apply(drop_mentions)                 \
            .apply(drop_newline)                  \
            .apply(reduce_extra_characters)       \
            .apply(drop_extra_punctuation)        \
            .apply(drop_quotations)               \
            .apply(drop_special_characters)       \
            .apply(spacify_question_mark)         \
            .apply(spacify_exclamation_point)     \
            .apply(combine_spaces)                \
            .str.strip()
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
    testData=pd.read_csv("./csv/test.csv")
    submission_df=pd.read_csv("./csv/sample_submission.csv")
    testData.drop(columns = ['keyword', 'location'])     
    ys=data['disaster']
    xs=data.drop(columns = ['disaster'])

    # use this instead of 'english' when testing if certain SWs improve performance
    ourStopWords = ['the', 'a', 'an', 'that']
    '''
    gridSelect = {
        'column_select__columns': [
            ['keyword', 'location', 'text'],
        ]
    }
    '''

    grid_params = {
        'vectorize__strip_accents' : ["unicode"],
        'vectorize__stop_words' : ["english"],
        'vectorize__ngram_range' : [(1,1),(1,2),(1,3),(1,4),(1,5)],
        'vectorize__max_features': [20000,50000,75000,100000],
        'vectorize__max_df': [0.5,0.6,0.7,.8, 1],
        'nb__alpha': [4.0,5.0,6.0,10.0],
    }

    steps=[
        ('column_select', SelectColumns('text')),
        ('preprocess', Preprocessor()),
        ('vectorize', CountVectorizer()),
        ('nb', MultinomialNB())
    ]

    print("Processing...",file=sys.stderr)
    pipe=Pipeline(steps)
    search=GridSearchCV(pipe, grid_params, scoring='accuracy', n_jobs=-1, cv=5)
    search.fit(xs,ys)
    print(search.best_score_)
    print(search.best_estimator_)
    print(search.best_params_)
    #Prepare predictions for kaggle submission
    submission_df['target']=search.predict(testData).tolist()
    print(submission_df.drop(columns=["Target"]).to_csv(index=False))
    print("Done!",file=sys.stderr)
    '''
    with pd.option_context('display.max_rows', None,
        'display.max_columns', None,
        'display.width', None):
        # print(data['text'].to_csv())
        print(pipe.transform(data['text']).to_csv())
    '''
main()












