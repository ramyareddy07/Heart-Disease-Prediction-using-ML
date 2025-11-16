from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import cvxpy as cp
class SVM:
    def __init__(self):
        self.x=None
        self.y=None
        self.feature_names=None
        self.target_name=None
        self.w=None
        self.r=None
        self.psi=None
        
    def fit(self,X,y):
        if isinstance(X,pd.DataFrame): 
            self.feature_names = [f for f in X.columns]
            self.x=np.array(X)
        else:
            self.x=X
        if isinstance(y,pd.DataFrame): 
            self.target_names = [f for f in y.columns]
            self.y=np.array(y).reshape(-1,1)
        else:
            self.y=y
        self.w,self.r,self.psi=self._optimization(self.x,self.y,0.1)
            
    def _optimization(self,x,y,c):
        xrows=len(x)
        xcols=x.shape[1]
        D=np.diag(y)
        E=np.ones(xrows)
        W=cp.Variable(xcols)
        r=cp.Variable(1)
        psi=cp.Variable(xrows)
        obj=cp.Minimize(0.5*cp.norm(W)**2 + c*cp.sum(psi))
        constraints=[cp.matmul(D, x*W-r*E)+psi >=E , psi >= 0]
        opt=cp.Problem(obj, constraints)
        opt.solve()
        W=np.array(W.value)
        r=r.value
        psi=np.array(psi.value)
        return W,r,psi
    
    def _prediction(self,x):
        x=x.reshape(x.shape[0],1)
        p=(np.matmul(self.w,x) - self.r)
        if p>0:
            return 1
        elif p<0:
            return -1
    
    def predict(self,x):
        ypred=[]
        x=np.array(x)
        for i in range(0,x.shape[0]):
            ypred.append(self._prediction(x[i,:]))
        return ypred
    

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=72)
svm_model=SVM()
svm_model.fit(xtrain,ytrain)
yp=svm_model.predict(xtest)

acc1=accuracy_score(ytest,yp)
print("Accuracy_score : {}".format(acc1))
print("\nClassification_report : \n")
print(classification_report(ytest,yp))
print("Confusion_Matrix : \n")
print(confusion_matrix(ytest,yp))