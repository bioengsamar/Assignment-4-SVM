import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


class LinearSVMUsingSoftMargin:
    def __init__(self, C=1.0):
        self._support_vectors = None
        self.C = C
        self.beta = None
        self.b = None
        
    def decision_function(self, X):
        return X.dot(self.beta) + self.b
 
    def cost(self, margin):
        return (1 / 2) * self.beta.dot(self.beta) + self.C * np.sum(np.maximum(0, 1 - margin))
 
    def margin(self, X, y):
        return y * self.decision_function(X)

    def fit(self, X, y, lr=0.001, n_iters=500):
        # Initialize Beta and b
        self.beta = np.random.randn(X.shape[1])
        self.b = 0
        loss_array = []
        for i in range(n_iters):
            margin = self.margin(X, y)
            loss = self.cost(margin)
            loss_array.append(loss)
            #print(loss)
            misclassified_pts_idx = np.where(margin < 1)[0] #get the indices of the data points where (yi(βTxi+1))<1
            #print(misclassified_pts_idx)
            delta_beta = self.beta - self.C * y[misclassified_pts_idx].dot(X[misclassified_pts_idx])
            self.beta = self.beta - lr * delta_beta  #update beta
            
            delta_b = - self.C * np.sum(y[misclassified_pts_idx])
            self.b = self.b - lr * delta_b  #update b
        
        self._support_vectors = np.where(self.margin(X, y) <= 1)[0] # then finally get the index of the support vectors

    def predict(self, X):
        return np.sign(self.decision_function(X))
 
    def score(self, X, y):
        P = self.predict(X)
        print(y == P)
        return np.mean(y == P)
 
def load_data(cols):
    global iris
    iris = sns.load_dataset("iris")
    iris = iris.tail(100)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(iris["species"])
    x= iris
    x= x[cols]
    return x.values, y
def plot (Y):
    features=[iris['petal_length'], iris['petal_width']]
    plt.scatter(features[0], features[1], alpha=0.8,
            s=100*features[1], c=Y, cmap='Spectral')
    plt.xlabel('petal_length')
    plt.ylabel('petal_width')
    plt.show()

if __name__ == '__main__':
    # make the targets are (-1, +1)
    cols = ["petal_length", "petal_width"]
    X, Y = load_data(cols)
    Y[Y == 0] = -1
    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    #split data into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
    model = LinearSVMUsingSoftMargin(C=15.0)
    #train model
    model.fit(X_train, y_train)
    #test model
    print("test score:", model.score(X_test, y_test))
    #plot
    plot(Y)
    
 
    