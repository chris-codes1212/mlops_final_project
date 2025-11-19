import json
import requests
from sklearn.metrics import accuracy_score

# get data from data file as a list of json objects to be used as payloads
with open('test_data.json', 'r') as data_file:
    test_data = json.load(data_file) 

# set url of our application and endpoint that we are sending requests to
url = "http://18.233.170.49:8000/predict" 

# create empty lists to append to in following for loop
true_labels = []
predicted_labels = []

# loop through all json payloads in the list of json items in test_data
for payload in test_data:

    try:
        # make call to /predict endpoint
        response = requests.post(url, json=payload)

        # get actual label from payload
        true_label = payload["true_label"]

        # get predicted label from response
        predicted_label = json.loads(response._content.decode("utf-8"))

        # append true and predicted labels to their respective lists
        true_labels.append(true_label)
        predicted_labels.append(predicted_label["Sentiment"])

        # output status code of response
        print(f'Status Code:{response.status_code}')

    # print error message on exception
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# calculate and output final accuracy of all predictions on test_data
accuracy = accuracy_score(y_true=true_labels, y_pred=predicted_labels)
print(f'Final Accuracy: {accuracy}')