import pandas as pd
import numpy as np
import os

dataset = pd.read_csv("Dataset/fraudTrain.csv",nrows=100000)
dataset['trans_date_trans_time'] = pd.to_datetime(dataset['trans_date_trans_time'])

X = []

def getFrequency(frequency, card, date_time):
    count = 0
    for i in range(len(frequency)):
        if str(frequency[i,0]) == str(date_time) and str(frequency[i,1]) == card:
            count = frequency[i,2]
            break
    return count
card = np.unique(dataset['cc_num'])
for i in range(len(card)):
    data = dataset.loc[dataset['cc_num'] == card[i]]
    frequency = data.groupby(['trans_date_trans_time', 'cc_num']).size()
    frequency = frequency.to_frame(name = 'frequency').reset_index()
    frequency = frequency.values
    average_time = data['trans_date_trans_time'].mean().to_datetime64().astype('M8[D]').astype('O')
    average_amount = data['amt'].mean()
    amounts = data['amt'].ravel()
    locations = data['state'].ravel()
    data = data.values
    last_transaction_date = data[len(data)-1,1].to_datetime64().astype('M8[D]').astype('O')
    for j in range(len(data)):
        days_difference = average_time - data[j,1].to_datetime64().astype('M8[D]').astype('O') #average transaction time - current time
        days_difference = days_difference.days
        amount_difference = average_amount - amounts[j] #average transaction amount - current transaction amount
        location = locations[j] #finding location  inside or outside toronto and canada
        interval = last_transaction_date  - data[j,1].to_datetime64().astype('M8[D]').astype('O') #differnce between last transaction date and current transaction date
        interval = interval.days
        freq = getFrequency(frequency, card[i], data[j,1])
        temp = []
        temp.append(card[i])
        if days_difference < 4:
            temp.append(0)
        if days_difference >= 4 and days_difference < 7:
            temp.append(1)
        if days_difference >= 7:
            temp.append(2)
        if amount_difference < 10:
            temp.append(0)
        if amount_difference >= 10 and amount_difference < 50:
            temp.append(1)
        if amount_difference >= 50:
            temp.append(2)
        if location == 'TN':
            temp.append(0)
        if location == 'CA':
            temp.append(1)
        if location != 'TN' and location != 'CA':
            temp.append(2)
        if interval < 30:
            temp.append(0)
        if interval >= 30 and interval < 90:
            temp.append(1)
        if interval >= 90:
            temp.append(2)
        if freq < 3:
            temp.append(0)
        if freq >= 3 and freq < 5:
            temp.append(1)
        if freq >= 5:
            temp.append(2)
        X.append(temp)    
            
output = pd.DataFrame(X, columns=['Card_No', 'Time_Difference', 'Amount_Difference', 'Location', 'Interval', 'Frequency'])
#output.to_csv("Data.csv", index=False)

#dataset = pd.read_csv("Data.csv")
temp = output.drop(['Card_No'], axis = 1)
temp = temp.values
labels = []

for i in range(len(temp)):
    unique, count = np.unique(temp[i], return_counts=True)
    index = np.argmax(count)
    if index == 0:
        labels.append("Legal")
    if index == 1:
        labels.append("Suspicious")
    if index == 2:
        labels.append("Fraud")
        
output['label'] = labels

output.to_csv("Data.csv", index= False)
dataset = pd.read_csv("Data.csv")
fraud = dataset.loc[dataset['label'] == "Fraud"]
suspicious = dataset.loc[dataset['label'] == "Suspicious"]
legal = dataset.loc[dataset['label'] == "Legal"]

fraud.to_csv("fraud.csv", index=False)
suspicious.to_csv("suspicious.csv", index=False)
legal.to_csv("legal.csv", index=False)

fraud = pd.read_csv("fraud.csv", nrows=9000)
suspicious = pd.read_csv("suspicious.csv", nrows=10000)
legal = pd.read_csv("legal.csv", nrows=10000)

dataset = [fraud, suspicious, legal]
dataset = pd.concat(dataset)
dataset.to_csv("Data.csv", index=False)            
os.remove("fraud.csv")       
os.remove("suspicious.csv")       
os.remove("legal.csv")
