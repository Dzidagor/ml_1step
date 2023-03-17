import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_excel("house_prices/price1.xlsx")
pred = pd.read_excel("house_prices/prediction_price.xlsx")
# Создание модели линейной регресси
reg = linear_model.LinearRegression()
# Обучение моедли на данных
reg.fit(df[["area"]], df.price)
plt.scatter(df.area, df.price)
plt.scatter(pred.area, reg.predict(pred[["area"]]), color="red")
# plt.plot(df.area, reg.predict(df[["area"]]))
plt.xlabel("площадь, m^2")
plt.ylabel("стоимость, млн. руб.")
plt.show()
print(reg.predict(pred[["area"]]))