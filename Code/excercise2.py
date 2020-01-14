import re
try:
    import nltk
except ImportError:
    from pip._internal import main as pip
    pip(['install', '--user', 'nltk'])
    import nltk

try:
    import sklearn
except ImportError:
    from pip._internal import main as pip
    pip(['install', '--user', 'sklearn'])
    import sklearn


# constants
fname_train_pos = '../data/IMDb/train/imdb_train_pos.txt'
fname_train_neg = '../data/IMDb/train/imdb_train_neg.txt'
fname_test_pos = '../data/IMDb/test/imdb_test_pos.txt'
fname_test_neg = '../data/IMDb/test/imdb_test_neg.txt'

# functions
REGX_PUNCT = re.compile("[.;:!\'?,\"()\[\]]")
REGEX_HTML = re.compile(r'<[^>]+>')


def clean_text(reviews):
    reviews = [REGX_PUNCT.sub("", line.lower()) for line in reviews]
    reviews = [REGEX_HTML.sub(" ", line) for line in reviews]    
    return reviews

STOP_WORDS = ['in','of','at','a','the', 'i', 'we', 'was', 'is']

def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in STOP_WORDS])
        )
    return removed_stop_words

def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

def get_stemmed_text(corpus):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

#--------------------------------------------------------------------

reviews_train_pos = []
for line in open(fname_train_pos, 'r', encoding='utf-8'):
    reviews_train_pos.append(line.strip())

reviews_train_neg = []
for line in open(fname_train_neg, 'r', encoding='utf-8'):
    reviews_train_neg.append(line.strip())

reviews_test_pos = []
for line in open(fname_test_pos, 'r', encoding='utf-8'):
    reviews_test_pos.append(line.strip())

reviews_test_neg = []
for line in open(fname_test_neg, 'r', encoding='utf-8'):
    reviews_test_neg.append(line.strip())

reviews_train = reviews_train_pos + reviews_train_neg
reviews_test = reviews_test_pos + reviews_test_neg

# Step 1 - pre-process text
print('Pre-processing')
print(' 1. convert to lowercase, clean text of punctuation and HTML...')
reviews_train = clean_text(reviews_train)
reviews_test = clean_text(reviews_test)
print('    done')

print(' 2. remove stop words...')
reviews_train = remove_stop_words(reviews_train)
reviews_test = remove_stop_words(reviews_test)
print('    done')

print(' 3. stemming text...')
reviews_train = get_stemmed_text(reviews_train)
print('.', end = '')
reviews_test = get_stemmed_text(reviews_test)
print('    done')

# X_train and X_test are lists of strings, each representing one document
X_train = reviews_train
X_test = reviews_test

# y_train and y_test are lists of labels
y_train = ([1] * len(reviews_train_pos)) + ([0] * len(reviews_train_neg))
y_test = ([1] * len(reviews_test_pos)) + ([0] * len(reviews_test_neg))


from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from nltk.corpus import stopwords

# this calculates a vector of term frequencies for each document
wc_vectorizer = CountVectorizer(binary=False) 

# this calculates a vector of two and three word n-grams that occur within each docuemnt 
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(2, 3))

# this normalizes each term frequency by the number of documents having that term
tfidf = TfidfTransformer()

# feature selection
k_best = SelectKBest(score_func=f_classif, k=500)

# this is a linear SVM classifier
clf = LinearSVC(C=1)

from sklearn.pipeline import FeatureUnion
pipeline = Pipeline(steps=[
    ("feature_union", FeatureUnion([
        ("wc_vectorizer",wc_vectorizer),
        ("ngram_vectorizer", ngram_vectorizer)
    ])),
    ('tfidf',tfidf),
    ('k_best',k_best),
    ('clf',clf)
])


# call fit as you would on any classifier
print('')
print('Training the model...')
pipeline.fit(X_train,y_train)
print('done')

# predict test instances
print('')
print('Predicting the model...')
y_pred = pipeline.predict(X_test)
print('done')

# calculate f1
print('')
print(classification_report(y_test, y_pred))

