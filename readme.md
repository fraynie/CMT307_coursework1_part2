# CMT307 Applied Machine Learning Coursework 1 (Part2)


## Running the code 
Clone or download the code and training and test IMDb data sets to your PC and navigate to the `CMT307_coursework1_part2-master\Code` folder. The solution is contained within a single python script called `excercise2.py`. To run the the script in Linux:
 - Type chmod a+x hello.py to tell Linux that it is an executable program
 - Type ./excercise2.py to run the script

## Extracting the data to Python lists
The positive and negative movie reviews contained in the IMDb are imported to Python lists using the following code:

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

## Text pre-processing
The first stage of text pre-processing is to convert to lower case, remove punctuation and HTML tags via using the following function which uses regex expressions to identify the text for removal:

    def clean_text(reviews):
        reviews = [REGX_PUNCT.sub("", line.lower()) for line in reviews]
        reviews = [REGEX_HTML.sub(" ", line) for line in reviews]    
        return reviews
Stopwords are then removed via the following function:

    STOP_WORDS = ['in','of','at','a','the', 'i', 'we', 'was', 'is']
    
    def remove_stop_words(corpus):
        removed_stop_words = []
        for review in corpus:
            removed_stop_words.append(
                ' '.join([word for word in review.split() 
                          if word not in STOP_WORDS])
            )
        return removed_stop_words

Finally stemming is performed using the NLTK PorterStemmer:

    def get_stemmed_text(corpus):
        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()
        return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

## Training labels
Lists of labels for the data are also created - the value '1' represents a positive review and '0' indicates a negative review.

    y_train = ([1] * len(reviews_train_pos)) + ([0] * len(reviews_train_neg))
    y_test = ([1] * len(reviews_test_pos)) + ([0] * len(reviews_test_neg))

## Model pipeline
The model pipeline includes a word count  and n-gram vectorizer, which are combined using the SKLearn FeatureUnion function, together with a TF-IDF vectorizer. 
    
    # this calculates a vector of term frequencies for each document
    wc_vectorizer = CountVectorizer(binary=False) 
    
    # this calculates a vector of two and three word n-grams that occur within each docuemnt 
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(2, 3))
    
    # this normalizes each term frequency by the number of documents having that term
    tfidf = TfidfTransformer()

Feature extraction is performed using univariate feature selection which works by selecting the best features based on univariate statistical tests - we use sklearn’s SelectKBest to  features to keep the 500 best features.

    # feature selection
    k_best = SelectKBest(score_func=f_classif, k=500)
   

 Finally we configure an linear support vector classifier.
    
    # this is a linear SVM classifier
    clf = LinearSVC(C=1)
 

   The various components are added to the pipeline as follows:

    pipeline = Pipeline(steps=[
        ("feature_union", FeatureUnion([
            ("wc_vectorizer",wc_vectorizer),
            ("ngram_vectorizer", ngram_vectorizer)
        ])),
        ('tfidf',tfidf),
        ('k_best',k_best),
        ('clf',clf)
    ])

## Training the predicting the model
We fit the pipeline to the training data with the fit function...

    # call fit as you would on any classifier
    pipeline.fit(X_train,y_train)

... followed by predict.

    # predict test instances
    y_pred = pipeline.predict(X_test)`

## Print out the performance 
Overall performance (precision, recall, f-measure and accuracy) of the trained model against the test set is available via the classification_report

    print(classification_report(y_test, y_pred))

Results from the model:
|              |precision    |recall  |f1-score   |support
|--------------|-------------|--------|-----------|-------
|           0  |    0.88     |0.85    | 0.86      |2501
|           1  |    0.86     |0.88    | 0.87      |2499
| | | | |
|    accuracy  |             |        | 0.87      |5000
|   macro avg  |    0.87     |0.87    | 0.87      |5000
|weighted avg  |    0.87     |0.87    | 0.87      |5000

