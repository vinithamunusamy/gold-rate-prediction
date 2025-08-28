# project 1
xa = np.array([2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]).reshape(-1,1)
yb = np.array([1110.200,1292.400,1469.200,1557.500,1648.000,1760.600,1865.570,1911.400,1900.000,2799.000])
print("total data")
print(xa)
print(yb)
#step 3
x_train,x_test,y_train,y_test = train_test_split(xa,yb,test_size=0.2)
print('trainig data')
print(x_train)
print(y_train)

print('testing data')
print(x_test)
print(y_test)
#step 4
lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)
#step 5 
print("prediction input")
print(x_test)
y_pred = lin_reg.predict(x_test)
print(y_pred)
#step 6
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error:",rmse)
#step 7 
plt.scatter(xa,yb,color ='blue',label ='Data points')
plt.plot(xa,lin_reg.predict(xa),color = 'red',label='Regression Line')
