from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import cvxpy as cp


class LogisticRegression:
    def __init__(self,iterations=1000,learning_rate=0.001):
        self.iterations=iterations
        self.lr=learning_rate
        self.feature_names=[]
        self.target_names=[]
        self.num_samples=0
        self.num_features=0
        self.x=0
        self.y=0
        self.tn=0
        self.t0=0
        
    def fit(self,x,y):
        if isinstance(x,pd.DataFrame): 
            self.feature_names = [f for f in x.columns]
            self.x=np.array(x)
        else:
            self.x=x 
        if isinstance(y,pd.DataFrame): 
            self.target_names = [f for f in y.columns]
            self.y=np.array(y).reshape(-1)
        else:
            self.y=y 
        self.num_samples=len(x)
        self.num_features=x.shape[1]
        
        self.tn=np.zeros(self.num_features)
        
        for i in range(self.iterations):
            LR=np.dot(self.x, self.tn) + self.t0
            ht=self._sigmoid_function(LR)
            gtn,gt0=self._updations(ht,self.tn,self.t0)
            self.tn-=self.lr*gtn
            self.t0-=self.lr*gt0
    
    def _updations(self,ht,tn,t0):
        gtn=np.dot(self.x.T, (ht-self.y))
        gtn=(1/self.num_samples)*gtn
        gt0=np.sum(ht-self.y)
        gt0=(1/self.num_samples)*gt0
        return gtn,gt0
        
    def _sigmoid_function(self,x):
        output=1/(1+(np.exp(-x)))
        return output
    
    def _prediction(self,x):
        l=np.dot(x, self.tn) + self.t0
        h=self._sigmoid_function(l)
        if h>0.5:
            return 1
        elif h<0.5:
            return 0
    def predict(self,x):
        predicted_values=[]
        x=np.array(x)
        for i in x:
            predicted_values.append(self._prediction(i))
        return predicted_values

xtrain3,xtest3,ytrain3,ytest3=train_test_split(X,y1,test_size=0.2,random_state=72)
lr_model=LogisticRegression(iterations=1500)
lr_model.fit(xtrain3,ytrain3)
yp3=lr_model.predict(xtest3)

acc3=accuracy_score(ytest3,yp3)
print("Accuracy_score : {}".format(acc3))
print("\nClassification_report : \n")
print(classification_report(ytest3,yp3))
print("Confusion_Matrix : \n")
print(confusion_matrix(ytest3,yp3))