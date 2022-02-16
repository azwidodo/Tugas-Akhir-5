from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Split training data by sentiment


def split_sentiment(df):

    df_pos = df[df["sentiment"] == 1]
    df_neg = df[df["sentiment"] == 0]

    return df_pos, df_neg


# Split training and testing data
def split_train_test(df_pos, df_neg, random_state=None):

    df_train_pos, df_test_pos = train_test_split(
        df_pos, test_size=0.2, random_state=random_state)
    df_train_neg, df_test_neg = train_test_split(
        df_neg, test_size=0.2, random_state=random_state)

    df_train = pd.concat([df_train_pos, df_train_neg])
    df_test = pd.concat([df_test_pos, df_test_neg])

    return df_train, df_test


# Plot accuracy and loss of neural network model
def plot_history(history):
    # Plotting accuracy on training data
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Plotting loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
