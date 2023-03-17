import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_excel("linear_regression_ht_data/gdprussia.xlsx")
model_by_one_dependence = linear_model.LinearRegression()
model_by_two_dependencies = linear_model.LinearRegression()
model_by_one_dependence.fit(data[["oilprice"]], data.gdp)
model_by_two_dependencies.fit(data[["year", "oilprice"]], data.gdp)
plt.xlabel("Цена на нефть, $")
plt.ylabel("ВВП")
print(model_by_two_dependencies.predict([[2025,100]]))
plt.scatter(data.oilprice, data.gdp)
plt.plot(data.oilprice, model_by_one_dependence.predict(data[["oilprice"]]), color="red")
plt.plot(data.oilprice, model_by_two_dependencies.predict(data[["year", "oilprice"]]), color="green")
plt.show()

print("")


