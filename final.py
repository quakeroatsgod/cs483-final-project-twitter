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

def main():
    data=pd.read_csv("./csv/train.csv")    
    ys=data['disaster']
    xs=data.drop(columns = ['disaster'])
    steps=[
        ('preprocess',Preprocessor()),
    ]
    print("\nFiltered\n")
    pipe=Pipeline(steps)
    pipe.fit(xs,ys)
    with pd.option_context('display.max_rows', None,
        'display.max_columns', None,
        'display.width', None):
        # print(data['text'].to_csv())
        print(pipe.transform(data['text']).to_csv())
main()