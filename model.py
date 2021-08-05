import pandas as pd
import pickle
import numpy as np
df=pd.read_excel('Final_data-crop_yield.xlsx')
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Crop']=le.fit_transform(df['Crop'])
X=df.drop(columns=['Domain','Area','Unit','Yield','RelativeHumidity'])
y=df['Yield']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)
model=LinearRegression()
model.fit(X_train,y_train)
lreg_yhat = model.predict(X_test)
# Linear Regression
mse1 = np.mean((lreg_yhat - y_test)**2)
rmse1 = np.sqrt(mse1)
print(rmse1)
# Save the model
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))
