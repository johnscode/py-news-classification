
import pandas as pd
import json
import matplotlib.pyplot as plt
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")


def remove_row_with_empty_column(df, col_name):
    df.replace("", pd.NA, inplace = True)
    df.dropna(subset=[col_name], inplace=True)
    return df


def scrub_news_cat_data(dataframe):
    dataframe = remove_row_with_empty_column(dataframe, "headline")
    dataframe = remove_row_with_empty_column(dataframe, "short_description")
    return dataframe


def get_categories(dataframe):
    category_counts = pd.DataFrame(dataframe.groupby(["category"]).count()["headline"].sort_values(ascending=False),columns=["headline"])
    category_list = category_counts.loc[category_counts["headline"] >= 2000].index.to_list()
    return category_list


def retrieve_news_cat_data():
    data = []
    path = 'data/News_Category_Dataset_v3.json'
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    raw_data_frame = pd.DataFrame(data)

    scrubbed_data_frame = scrub_news_cat_data(raw_data_frame)
    category_list = get_categories(scrubbed_data_frame)
    return scrubbed_data_frame, category_list


def prepare_feature_data(data_frame, category_list):
    trainlist = []
    for cat in category_list:
        trainlist.append(data_frame.loc[data_frame["category"] == cat,['headline', 'category', 'short_description']].head(1000))
    train_data_frame = pd.concat(trainlist, ignore_index=True)
    train_data_frame["news"] = train_data_frame["headline"] +' '+ train_data_frame["short_description"]

    train_data_frame = remove_empty_row(train_data_frame, "news")
    train_data_frame = text_cleanup(train_data_frame,"news")
    return train_data_frame


# remove numbers, punctuation and stop words
def text_cleanup(df,col_name):
    all_doc = []
    for index,row in df.iterrows():
        doc = nlp(row[col_name])
        token_text = []
        for token in doc:
            if not(token.like_num or token.is_punct or token.is_stop):
                token_text.append(str(token.lemma_).lower())
        token_text = " ".join(token_text)
        all_doc.append(token_text)
    df.loc[:,col_name] = all_doc
    return df


def remove_empty_row(df, col_name):
    df.replace([" ","","_","__"], pd.NA, inplace = True)
    df.dropna(subset=[col_name], inplace=True)
    return df


def vectorize(df,col_name,target_col_name):
    vectorizer = TfidfVectorizer()
    feature = vectorizer.fit_transform(df.loc[:,col_name])
    tdf_df = pd.DataFrame(feature.toarray(),columns=vectorizer.get_feature_names_out())
    X_train, X_test , y_train, y_test = train_test_split(feature.toarray(),df.loc[:,target_col_name],random_state = 42)
    return X_train, X_test,y_train,y_test


# use this function with scrubbed data frame
def plot_category_dist(data_frame):
    count_df = pd.DataFrame(data_frame.groupby(["category"]).count()["headline"].sort_values(ascending=False),columns=["headline"])
    plt.figure(figsize=(30,10))
    plt.xlabel("categories")
    plt.ylabel("count")
    plt.bar(count_df.index,count_df["headline"])
    plt.plot(count_df.index,count_df["headline"],'ro')
    return

