from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import cvxpy as cp

def distance (x,p): 
        l=list(map(lambda x,y:(x-y)**2,x,p))
        return np.sqrt(sum(l))

class KNearestNeighbours : 
    def __init__(self,n_neighbors=5): 
        self.n_neighbors=n_neighbors 
        self.feature_names=[]
        self.target_names=[]
        self.num_sampels=0
        self.x=0
        self.y=0
        self.predicted_values=0

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
        self.num_sampels=len(x)
        
    def predict(self, p): 
        predict_values=[] 
        p=np.array(p) 
        for i in p: 
            p = dict(sorted(self._predictions(i).items())[0:self.n_neighbors]) 
            predict_values.append(stats.mode(p.values())) 
        self.predicted_values=predict_values
        return predict_values
    def _predictions(self,p):
        d=dict()
        for x,y in zip(self.x,self.y):
            dis=distance(x,p)
            d.update({dis:y})
        return d
    
xtrain2,xtest2,ytrain2,ytest2=train_test_split(X,y,test_size=0.2,random_state=90)
knn_model=KNearestNeighbours(3)
knn_model.fit(xtrain2,ytrain2)

yp2=knn_model.predict(xtest2)
acc2=accuracy_score(ytest2,yp2)
print("accuracy_score : ",acc2,"\n")
print("\nClassification_report : \n")
print(classification_report(ytest2,yp2))
print("Confusion_Matrix : \n")
print(confusion_matrix(ytest2,yp2))
