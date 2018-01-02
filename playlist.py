
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
#np.random.seed(9001)
import os
from sklearn.metrics import mean_squared_error




#Splitting data into training and development set
def df_to_array(df):
    return df.iloc[:,0:len(df.columns)-1],df.iloc[:,-1]


def split(data):
    #np.random.seed(9001)
    dev_data=data.sample(frac=0.2)
    train_data=data.drop(dev_data.index)
    train_data.reset_index(inplace=True, drop= True)
    dev_data.reset_index(inplace=True, drop= True)

    train_x, train_y= df_to_array(train_data)
    dev_x,dev_y= df_to_array(dev_data)
    
    return train_x,train_y,dev_x,dev_y


def train_without_trans(train_x,train_y,dev_x,dev_y):
    lm=LinearRegression()
    lm
    #Train the model without any transformation
    mod=lm.fit(train_x,train_y)

    # print("Estimated intercept coefficients", lm.intercept_)
    # print("Number of Coefficients", lm.coef_)
    predicted_dev=lm.predict(dev_x)
    plt.figure(figsize=(20,20))
    plt.subplot(2,1,1)
    plt.scatter(dev_y,lm.predict(dev_x))
    plt.plot()
    #Residual plots
    plt.figure(figsize=(20,20))
    plt.subplot(2,1,2)
    plt.scatter(predicted_dev,dev_y-predicted_dev)
    plt.show()
    y_predicted=lm.predict(dev_x)

    print("Mean Squared Error without Transformation", mean_squared_error(dev_y, y_predicted))


def train_with_trans(train_x,train_y,dev_x,dev_y):
    lm=LinearRegression()
    lm
    #Train with transformation of the response variable
    mod=lm.fit(train_x,np.log10(train_y))

    predicted_dev=lm.predict(dev_x)
    plt.figure(figsize=(20,20))
    plt.subplot(2,1,1)
    plt.scatter(np.log10(dev_y),lm.predict(dev_x))
    plt.plot()
    #Residual plots
    plt.figure(figsize=(20,20))
    plt.subplot(2,1,2)
    plt.scatter(predicted_dev,np.log10(dev_y)-predicted_dev)
    plt.show()
    #Predict on development data
    predicted_dev=lm.predict(dev_x)
    predicted_dev
    #Actual prediction is inverse of log10

    pred=10**(predicted_dev)
    
    print("Mean Squared Error with Transformation", mean_squared_error(dev_y, pred))

def main():
    #Getting the dat directory
    current_dir=os.getcwd()
    data_dir = os.path.join(current_dir, 'train/')
    #Wrting the CSV file
    master_file = open('data.csv', 'w+')
    #Reading the text file
    with open('train-x') as f:
        for line in f:
            with open(os.path.join(data_dir, line.rstrip()), "r") as f1:
                master_file.write(f1.readlines()[0]+"\n")
    
    #Assigning column names
    data=pd.read_csv('data.csv',header= None)
    data.columns=['acousticness','danceability','energy','instrumentalness','liveness','speechiness','tempo','valence']


    #Analyzing if correlations exist within the dataset
    corr=data.corr()
    print(corr)


    #Reading the taget class file
    with open('train-y') as f:
        temp = [line.strip() for line in f]


    #Cleaning target file
    import re
    y=[]
    for i in range(0,len(temp)):
        if i==0:
            s = temp[0]
            y.append(int((re.findall('\d+', s))[0]))
        else:
            y.append(int(temp[i]))
    data['y']=y[0:len(data)]
    train_x,train_y,dev_x,dev_y=split(data)
  
    train_without_trans(train_x,train_y,dev_x,dev_y)
    train_with_trans(train_x,train_y,dev_x,dev_y)




if __name__=="__main__":
    main()





