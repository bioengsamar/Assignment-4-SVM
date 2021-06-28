from sklearn.svm import SVC
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split


# Load a txt file
def load_txt(filename):
    data = pd.read_excel(filename, header=None)
    return data

def split_data(data):
    x=data.iloc[:,0:24]
    y=data.iloc[:,24]
    x_data=[]
    y_data=[]
    chunk_size=int(data.shape[0]/10)
    for i in range(10):
        start=chunk_size*i
        features=x.loc[start:start+chunk_size]
        lables=y.loc[start:start+chunk_size]
        x_data.append(features)
        y_data.append(lables)
        
    return x_data, y_data, x, y

def model (x_data, y_data):
    scores=[]
    for i in range(10):
        x_test=x_data[i:4+i]
        y_test=y_data[i:4+i]
        x_train=x_data[:i]+x_data[4+i:]
        y_train=y_data[:i]+y_data[4+i:]
        x1_frames = [l for l in x_train]
        y1_frames = [m for m in y_train]
        x2_frames = [l for l in x_test]
        y2_frames = [m for m in y_test]
        x1=pd.concat(x1_frames, ignore_index=True)
        y1=pd.concat(y1_frames, ignore_index=True)
        x2=pd.concat(x2_frames, ignore_index=True)
        y2=pd.concat(y2_frames, ignore_index=True)
        model=SVC(kernel='linear')
        model.fit(x1, y1)
        score=model.score(x2,y2)
        scores.append(score)
    return np.mean(scores)

if __name__ == "__main__":
    path="data.xlsx"
    data=load_txt(path)
    x_data,y_data, x, y=split_data(data)
    print(model(x_data, y_data)) #0.7573391606363946
    #print(x_data[0])
    
    #calculation accuracy 10 times using method train_test_split without making it from scratch
    scores=[]
    for i in range(10):
        #print(i)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4,random_state=i)
        model=SVC(kernel='linear')
        model.fit(X_train, y_train)
        score=model.score(X_test,y_test)
        #print(score)
        scores.append(score)
    print(scores)
    print(np.mean(scores)) #0.7456249999999999