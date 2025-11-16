import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler as RS

Df=pd.read_csv("heart.csv")
Df.head(5)

X=Df
X['Heart Disease'].replace('Presence',1,inplace=True)
X['Heart Disease'].replace('Absence',-1,inplace=True)

y=Df['Heart Disease']
y1=y.copy(deep=True)
X=X.drop(columns='Heart Disease')

features_names=[i for i in X.columns]
target_name='Heart Disease'

y1.replace(-1,0,inplace=True)


scaler=RS()
X=scaler.fit_transform(X)
X=pd.DataFrame(data=X, columns=features_names)
X.head()