import numpy as np
import pandas as pd
import xml.etree.ElementTree as et
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tag import CRFTagger
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import string
import random
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import hamming_loss, accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV

# Parsing human_annotated_labeled_data
human_annotated_labeled_data = et.parse("human_annotated_labeled_data.xml")

ids = []
titles = []
contents = []
labels = []

for id in human_annotated_labeled_data.iter('ID'):
    ids.append(id.text)

for title in human_annotated_labeled_data.iter('JUDUL'):
    titles.append(title.text)

for content in human_annotated_labeled_data.iter('ISI'):
    contents.append(content.text)

for _class in human_annotated_labeled_data.iter('CLASS'):
    label_class = []
    for label in _class.iter('LABEL'):
        label_class.append(label.text)
    labels.append(label_class)

human_annotated_labeled_data_df = pd.DataFrame(list(zip(id, title, content, label)), columns=['id', 'title', 'content', 'label'])

# Parsing machine_annotated_labeled_data
machine_annotated_labeled_data = et.parse("machine_annotated_labeled_data_v1.xml")

ids = []
titles = []
contents = []
labels = []

for id in machine_annotated_labeled_data.iter('ID'):
    ids.append(id.text)

for title in machine_annotated_labeled_data.iter('JUDUL'):
    titles.append(title.text)

for content in machine_annotated_labeled_data.iter('ISI'):
    contents.append(content.text)

for _class in machine_annotated_labeled_data.iter('CLASS'):
    label_class = []
    for label in _class.iter('LABEL'):
        label_class.append(label.text)
    labels.append(label_class)

machine_annotated_labeled_data_df = pd.DataFrame(list(zip(id, title, content, label)), columns=['id', 'title', 'content', 'label'])

# Combining dataframes
human_machine_df = pd.concat([human_annotated_labeled_data_df, machine_annotated_labeled_data_df])

# Check shape
human_machine_df.shape

# Check for null values
human_machine_df.isnull().sum()

# Drop null values
human_machine_df.dropna(inplace=True)

# Remove duplicated rows based on 'title' and 'content'
human_machine_df = human_machine_df[~human_machine_df[['title', 'content']].duplicated()]

# Initialize CRFTagger
ct = CRFTagger()
ct.set_model_file('all_indo_man_tag_corpus_model.crf.tagger')

# Define pos_filter function
def pos_filter(lst_type, text):
    res = []
    pos_tag = ct.tag_sents([text.split()])
    for tag in pos_tag[0]:
        if tag[1] in lst_type:
            res.append(tag[0])
    return " ".join(res) 

# Define preprocess_post_filter function
def preprocess_post_filter(text):
    lst_type = ["NN", "NNS", "NNP", "NNPS"]
    filtered_text = pos_filter(lst_type, text)
    return filtered_text

# Preprocess 'content' column
human_machine_df['content_cleaned'] = human_machine_df['content'].apply(lambda x: x.lower())
human_machine_df['content_cleaned'] = human_machine_df['content_cleaned'].apply(lambda x: re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", x))
human_machine_df['content_cleaned'] = human_machine_df['content_cleaned'].apply(lambda x: re.sub('\w*\d\w*',' ', x))
human_machine_df['content_cleaned'] = human_machine_df['content_cleaned'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))
human_machine_df['content_cleaned'] = human_machine_df['content_cleaned'].apply(lambda x: re.sub(' +',' ',x))

# Initialize Stopword Remover Factory
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# Remove stop words
human_machine_df['content_cleaned'] = human_machine_df['content_cleaned'].apply(lambda x: stopword.remove(x))

# Initialize Stemmer Factory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Apply stemming
human_machine_df['content_cleaned'] = human_machine_df['content_cleaned'].apply(lambda x: stemmer.stem(x))

# Check cleaned 'content' column
human_machine_df['content_cleaned']

# Define a function to convert text to lowercase and strip whitespace
def lower_and_strip(x):
    lst = []
    for i in x:
        lst.append(i.lower().strip())
    return lst

# Apply the function to the 'labels' column
human_machine_df['labels'] = human_machine_df['labels'].apply(lambda x: None if x == [None] else x)
human_machine_df.dropna(inplace=True)
human_machine_df['labels'] = human_machine_df['labels'].apply(lower_and_strip)

# Define the output categories
output_category = [
    "Kebidanan dan Kandungan",
    "Penyakit Dalam",
    "Kesehatan Anak",
    "Kesehatan Kulit dan Kelamin",
    "Kesehatan Gizi",
    "Kesehatan Telinga, Hidung dan Tenggorokan (THT)",
    "Gigi",
    "Kesehatan Mata",
    "Bedah",
    "Kesehatan Jiwa",
    "Ortopedi (Tulang)",
    "Jantung dan Pembuluh Darah",
    "Urologi",
    "Saraf",
    "Pulmonologi (Paru)",
    "Umum",
]

# Convert output categories to lowercase
lower_category = [i.lower() for i in output_category]

# Define a function to extract labels
def extract_label(df):
    mlb = MultiLabelBinarizer(classes=lower_category)
    extracted_label_array = mlb.fit_transform(df['labels']) 
    extracted_label_df = pd.DataFrame(extracted_label_array, columns=mlb.classes_)
    return df.drop('labels', axis=1), extracted_label_df, mlb

# Extract labels using the defined function
X, y, mlb = extract_label(human_machine_df)

# Select the 'content_cleaned' column from X
X = X[['content_cleaned']]

# Split the dataset into training and testing sets
X_train, y_train, X_test, y_test = iterative_train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2)

# Create a TfidfVectorizer
vectorizer = TfidfVectorizer(token_pattern='[^\s]{3,}')

# Fit and transform the training data
train_vec = vectorizer.fit_transform(pd.DataFrame(X_train)[0].astype(str))

# Transform the testing data
test_vec = vectorizer.transform(pd.DataFrame(X_test)[0].astype(str))

# Create a BinaryRelevance classifier with LinearSVC
classifierSVM = BinaryRelevance(
    classifier=LinearSVC(),
    require_dense=[False, True]
)

# Fit the classifier to the training data
classifierSVM.fit(train_vec, y_train)

# Import necessary metrics and perform evaluation
pred = classifierSVM.predict(test_vec)
print("Accuracy score:", accuracy_score(y_test, pred))
print("Hamming loss:", hamming_loss(y_test, pred))
print(classification_report(y_test, pred, target_names=lower_category))

# Concatenate dataframes
human_machine_df = pd.concat([human_annotated_labeled_data_df, machine_annotated_labeled_data_df])

# Create a new column by combining 'title' and 'ISI'
human_machine_df['title_and_content'] = human_machine_df['title'] + " " + human_machine_df['ISI']

# Check for rows with missing values in both 'title' and 'ISI'
human_machine_df[human_machine_df['title'].isna() & human_machine_df['ISI'].isna()]

# Check for missing values in the dataframe
human_machine_df.isnull().sum()

# Fill missing values with empty strings
human_machine_df.fillna("", inplace=True)
human_machine_df.isnull().sum()

# Apply text preprocessing steps
human_machine_df['title_and_content_cleaned'] = human_machine_df['title_and_content'].apply(lambda x: x.lower())
human_machine_df['title_and_content_cleaned'] = human_machine_df['title_and_content_cleaned'].apply(lambda x: re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", x))
human_machine_df['title_and_content_cleaned'] = human_machine_df['title_and_content_cleaned'].apply(lambda x: re.sub('\w*\d\w*',' ', x))
human_machine_df['title_and_content_cleaned'] = human_machine_df['title_and_content_cleaned'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))
human_machine_df['title_and_content_cleaned'] = human_machine_df['title_and_content_cleaned'].apply(lambda x: re.sub(' +',' ',x))

# Remove stopwords
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
human_machine_df['title_and_content_cleaned'] = human_machine_df['title_and_content_cleaned'].apply(lambda x: stopword.remove(x))

# Stem the text
factory = StemmerFactory()
stemmer = factory.create_stemmer()
human_machine_df['title_and_content_cleaned'] = human_machine_df['title_and_content_cleaned'].apply(lambda x: stemmer.stem(x))

# Apply text preprocessing to labels
human_machine_df['labels'] = human_machine_df['labels'].apply(lambda x: None if x == [None] else x)
human_machine_df.dropna(inplace=True)
human_machine_df['labels'] = human_machine_df['labels'].apply(lower_and_strip)

# Extract labels
X, y, mlb = extract_label(human_machine_df)

# Select the 'title_and_content_cleaned' column from X
X = X[['title_and_content_cleaned']]

# Split the dataset into training and testing sets using iterative_train_test_split
X_train, y_train, X_test, y_test = iterative_train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2)

# Create a TfidfVectorizer for feature extraction
vectorizer = TfidfVectorizer(token_pattern='[^\s]{3,}')

# Fit and transform the training data
train_vec = vectorizer.fit_transform(pd.DataFrame(X_train)[0].astype(str))

# Transform the testing data
test_vec = vectorizer.transform(pd.DataFrame(X_test)[0].astype(str))

# Create a BinaryRelevance classifier with SVC (SVM with linear kernel)
classifierSVM = BinaryRelevance(
    classifier=SVC(kernel='linear', probability=True),
    require_dense=[False, True]
)

# Fit the classifier to the training data
classifierSVM.fit(train_vec, y_train)

# Import necessary metrics and perform evaluation
pred = classifierSVM.predict(test_vec)
print("accuracy score", accuracy_score(y_test, pred))
print("hamming loss", hamming_loss(y_test, pred))
print(classification_report(y_test, pred, target_names=lower_category))

# Use XGBoost classifier
classifierXGBoost = BinaryRelevance(
    classifier=XGBClassifier(),
    require_dense=[False, True]
)

# Fit the XGBoost classifier to the training data
classifierXGBoost.fit(train_vec, y_train)

# Import necessary metrics and perform evaluation
pred = classifierXGBoost.predict(test_vec)
print("Accuracy score:", accuracy_score(y_test, pred))
print("Hamming loss:", hamming_loss(y_test, pred))
print(classification_report(y_test, pred, target_names=lower_category))

# Oversampling

def create_dataset():

    ''' 
    Create a unevenly distributed sample data set multilabel  
    classification using make_classification function
    
    args
    nsample: int, Number of sample to be created
    
    return
    X: pandas.DataFrame, feature vector dataframe with 10 features 
    y: pandas.DataFrame, target vector dataframe with 5 labels
    '''
    
    X, y = make_classification(n_classes=5, class_sep=2, 
                           weights=[0.1, 0.025, 0.205, 0.008, 0.9], n_informative=3, n_redundant=1, flip_y=0,
                           n_features=10, n_clusters_per_class=1, n_samples=1000, random_state=10)
    y = pd.get_dummies(y, prefix='class')
    return pd.DataFrame(X), y

def get_tail_label(df):
    columns = df.columns
    n = len(columns)
    irpl = np.zeros(n)
    for column in range(n):
        irpl[column] = df[columns[column]].value_counts()[1]
    irpl = max(irpl) / irpl
    mir = np.average(irpl)
    tail_label = []
    for i in range(n):
        if irpl[i] > mir:
            tail_label.append(columns[i])
    return tail_label

def get_index(df):
    tail_labels = get_tail_label(df)
    index = set()
    for tail_label in tail_labels:
        sub_index = set(df[df[tail_label] == 1].index)
        index = index.union(sub_index)
    return list(index)

def get_minority_instace(X, y):
    index = get_index(y)
    X_sub = X[X.index.isin(index)].reset_index(drop=True)
    y_sub = y[y.index.isin(index)].reset_index(drop=True)
    return X_sub, y_sub

def nearest_neighbour(X):
    nbs = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='kd_tree').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices

def MLSMOTE(X, y, n_sample):
    indices2 = nearest_neighbour(X)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n - 1)
        neighbour = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis=0, skipna=True)
        target[i] = np.array([1 if val > 2 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference, :] - X.loc[neighbour, :]
        new_X[i] = np.array(X.loc[reference, :] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    new_X = pd.concat([X, new_X], axis=0)
    target = pd.concat([y, target], axis=0)
    return new_X, target

X, y, mlb = extract_label(human_machine_df)
X = X[['title_and_content_cleaned']]

X_train, y_train, X_test, y_test = iterative_train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2)

vectorizer = TfidfVectorizer(token_pattern='[^\s]{3,}')
train_vec = vectorizer.fit_transform(pd.DataFrame(X_train)[0].astype(str))
test_vec = vectorizer.transform(pd.DataFrame(X_test)[0].astype(str))

X_sub, y_sub = get_minority_instace(pd.DataFrame(train_vec.toarray()), pd.DataFrame(y_train))

X_res, y_res = MLSMOTE(X_sub, y_sub, 100)

new_y_train = np.concatenate((y_train, y_res.astype(int).values))
new_x_train = np.concatenate((train_vec.toarray(), X_res.values))

classifierSVM = BinaryRelevance(
    classifier=LinearSVC(),
    require_dense=[False, True]
)

classifierSVM.fit(new_x_train, new_y_train)

pred = classifierSVM.predict(test_vec)
print("Accuracy score:", accuracy_score(y_test, pred))
print("Hamming loss:", hamming_loss(y_test, pred))
print(classification_report(y_test, pred, target_names=lower_category))

# Using Unlabeled Data

unlabeled_data = et.parse("unlabeled_data_v1.xml")

ids = []
titles = []
contents = []

for id in unlabeled_data.iter('ID'):
    ids.append(id.text)

for title in unlabeled_data.iter('JUDUL'):
    titles.append(title.text)

for content in unlabeled_data.iter('ISI'):
    contents.append(content.text)

unlabeled_data_df = pd.DataFrame(list(zip(ids, titles, contents)), columns=['id', 'title', 'content'])

unlabeled_data_df.head(5)

# Check the shape of the DataFrame
print(unlabeled_data_df.shape)

# Check for missing values
print(unlabeled_data_df.isnull().sum())

# Drop rows with missing values
unlabeled_data_df.dropna(inplace=True)

# Check for missing values again
print(unlabeled_data_df.isnull().sum())

# Combine 'title' and 'content' into 'title_and_content'
unlabeled_data_df['title_and_content'] = unlabeled_data_df['title'] + " " + unlabeled_data_df['content']

# Lowercase the 'title_and_content_cleaned' column
unlabeled_data_df['title_and_content_cleaned'] = unlabeled_data_df['title_and_content'].apply(lambda x: x.lower())

# Remove email, digits, punctuation, and extra whitespace
unlabeled_data_df['title_and_content_cleaned'] = unlabeled_data_df['title_and_content_cleaned'].apply(
    lambda x: re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", x)
)
unlabeled_data_df['title_and_content_cleaned'] = unlabeled_data_df['title_and_content_cleaned'].apply(
    lambda x: re.sub('\w*\d\w*',' ', x)
)
unlabeled_data_df['title_and_content_cleaned'] = unlabeled_data_df['title_and_content_cleaned'].apply(
    lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x)
)
unlabeled_data_df['title_and_content_cleaned'] = unlabeled_data_df['title_and_content_cleaned'].apply(
    lambda x: re.sub(' +',' ',x)
)

# Remove stopwords
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
unlabeled_data_df['title_and_content_cleaned'] = unlabeled_data_df['title_and_content_cleaned'].apply(lambda x: stopword.remove(x))

# Stem the text
factory = StemmerFactory()
stemmer = factory.create_stemmer()
unlabeled_data_df['title_and_content_cleaned'] = unlabeled_data_df['title_and_content_cleaned'].apply(lambda x: stemmer.stem(x))

# Check the preprocessed DataFrame
print(unlabeled_data_df)

# Check the shape of the 'human_machine_df' DataFrame (if needed)
# print(human_machine_df.shape)

# Vectorize the cleaned text
unlabeled_data_vec = vectorizer.transform(unlabeled_data_df['title_and_content_cleaned'].to_numpy())

# Predict using the trained classifier
unlabeled_predict_vec = classifierSVM.predict_proba(unlabeled_data_vec)

# Create a DataFrame with prediction probabilities
unlabeled_predict_df = pd.DataFrame(unlabeled_predict_vec.toarray())
unlabeled_predict_df.index = unlabeled_data_df.index

# Check the prediction DataFrame
print(unlabeled_predict_df)

# Rename columns in unlabeled_predict_df
unlabeled_predict_df.columns = lower_category

# Define a function to convert probabilities to binary labels
def convertToOne(x):
    if x < 0.90:
        return 0
    else:
        return 1

# Apply the conversion function to unlabeled_predict_df
unlabeled_predict_df_selected = unlabeled_predict_df.copy()
for category in lower_category:
    unlabeled_predict_df_selected[category] = unlabeled_predict_df_selected[category].apply(convertToOne)

# Filter rows with at least one confident prediction
confidence_unlabeled_data = unlabeled_predict_df_selected.loc[~(unlabeled_predict_df_selected==0).all(axis=1)]

# Combine unlabeled_data_df with confident predictions
unlabeled_data_annotate = pd.concat([unlabeled_data_df, confidence_unlabeled_data], axis=1).dropna()

# Extract labels from human_machine_df
X, y, mlb = extract_label(human_machine_df)
y.index = X.index

# Combine labeled and unlabeled data
joined_train_data = pd.concat([X, y], axis=1)
joined_train_data = pd.concat([joined_train_data, unlabeled_data_annotate])

# Reset index and select features and labels
joined_train_data.reset_index(drop=True, inplace=True)
X = joined_train_data[['title_and_content_cleaned']]
y = joined_train_data[lower_category]
y = y.astype(int)

# Split the data into train and test sets
X_train, y_train, X_test, y_test = iterative_train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2)

# Create a TF-IDF vectorizer
vectorizer_two = TfidfVectorizer(token_pattern='[^\s]{3,}')
train_vec = vectorizer_two.fit_transform(pd.DataFrame(X_train)[0].astype(str))
test_vec = vectorizer_two.transform(pd.DataFrame(X_test)[0].astype(str))

# Initialize and train the classifier
classifierSVMJoinedData = BinaryRelevance(
    classifier=LinearSVC(),
    require_dense=[False, True]
)
classifierSVMJoinedData.fit(train_vec, y_train)

# Evaluate the classifier
pred = classifierSVMJoinedData.predict(test_vec)
print("Accuracy score:", accuracy_score(y_test, pred))
print("Hamming loss:", hamming_loss(y_test, pred))
print(classification_report(y_test, pred, target_names=lower_category))

# Hyperparameter tuning using GridSearchCV
gridSearchClf = GridSearchCV(classifierSVMJoinedData, {"classifier__C": [0.1, 1, 2, 5]}, cv=3, scoring='f1_macro', verbose=2)
gridSearchClf.fit(train_vec, y_train)

# Evaluate the tuned classifier
pred = gridSearchClf.predict(test_vec)
print("Accuracy score:", accuracy_score(y_test, pred))
print("Hamming loss:", hamming_loss(y_test, pred))
print(classification_report(y_test, pred, target_names=lower_category))

# Get the best estimator from GridSearchCV
best_classifier = gridSearchClf.best_estimator_

def cleaning_test_data(df):
    # Lower
    df['title_and_content_cleaned'] = df['title_and_content'].apply(lambda x: x.lower())

    # Remove email
    df['title_and_content_cleaned'] = df['title_and_content_cleaned'].apply(lambda x: re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", x))

    # Remove Digit
    df['title_and_content_cleaned'] = df['title_and_content_cleaned'].apply(lambda x: re.sub('\w*\d\w*', ' ', x))

    # Remove Punctuation
    df['title_and_content_cleaned'] = df['title_and_content_cleaned'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))

    # Remove Whitespace
    df['title_and_content_cleaned'] = df['title_and_content_cleaned'].apply(lambda x: re.sub(' +', ' ', x))

    ## Stopword
    df['title_and_content_cleaned'] = df['title_and_content_cleaned'].apply(lambda x: stopword.remove(x))

    ## Stemming
    df['title_and_content_cleaned'] = df['title_and_content_cleaned'].apply(lambda x: stemmer.stem(x))

    return df

# Parsing testing_data
testing_data = et.parse("testing_data_v1.xml")

ids = []
titles = []
contents = []

for id in testing_data.iter('ID'):
    ids.append(id.text)

for title in testing_data.iter('JUDUL'):
    titles.append(title.text)

for content in testing_data.iter('ISI'):
    contents.append(content.text)

testing_data_df = pd.DataFrame(list(zip(ids, titles, contents)), columns=['id', 'title', 'content'])

# Check data shape and handle missing values
testing_data_df.fillna("", inplace=True)
testing_data_df['title_and_content'] = testing_data_df['title'] + " " + testing_data_df['content']

# Clean the testing data using the cleaning_test_data function
testing_data_df = cleaning_test_data(testing_data_df)

# Transform testing data using the vectorizer and predict labels
testing_data_vec = vectorizer_two.transform(testing_data_df['title_and_content_cleaned'].to_numpy())
testing_data_predict_vec = gridSearchClf.predict(testing_data_vec).toarray()

# Create a DataFrame with the predicted labels
testing_data_predict_df = pd.DataFrame(testing_data_predict_vec)
testing_data_predict_df.insert(0, 'id', testing_data_df['id'].values)

# Save the results to a CSV file
testing_data_predict_df.to_csv("results/br_SVC_kernel_linear_semisupervised_gridsearch.csv", header=False, index=False)
