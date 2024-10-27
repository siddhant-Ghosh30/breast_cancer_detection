import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# export model and scaler to a binary file to be used in the application so that we don't have to train the model again as that will take time
import pickle


def create_model(data):
    x = data.drop(columns=['diagnosis'])
    y = data['diagnosis']

    # scale the data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)

    # train
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # test model
    y_pred = model.predict(x_test)
    print(f"Accuracy of our model is:{accuracy_score(y_test, y_pred)}")
    print(f"Classification report:\n {classification_report(y_test, y_pred)}")

    return model, scaler
 

def get_clean_data():
    data = pd.read_csv("data.csv")
    data = data.drop(columns=['Unnamed: 32', 'id']) # or data.drop(['Unnamed: 32', 'id'], axis=1)

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data


def main():

    data = get_clean_data()
    # print(data.info())
    model, scaler = create_model(data)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    



if __name__ == "__main__":
    main()

