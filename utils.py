from sklearn.model_selection import train_test_split
import pandas as pd


# Split training data by sentiment
def split_sentiment(df):

    df_pos = df[df["sentiment"] == 1]
    df_neg = df[df["sentiment"] == 0]

    return df_pos, df_neg


# Split training and testing data
def split_train_test(df_pos, df_neg, random_state=None):

    df_train_pos, df_test_pos = train_test_split(df_pos, test_size=0.2, random_state=random_state)
    df_train_neg, df_test_neg = train_test_split(df_neg, test_size=0.2, random_state=random_state)

    df_train = pd.concat([df_train_pos, df_train_neg])
    df_test = pd.concat([df_test_pos, df_test_neg])

    return df_train, df_test