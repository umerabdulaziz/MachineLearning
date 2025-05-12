import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("AI-ML.csv")

x = df[['Feature1', 'Feature2', 'Feature3', 'Feature4']]
y = df['Output']

model = LinearRegression()
model.fit(x, y)

print("Slope (m):", model.coef_)
print("Intercept (c):", model.intercept_)

input_data = [[60, 70, 80, 90]]
prediction = model.predict(input_data)
print("Prediction:", prediction[0])

plt.hist(df['Output'], bins=10, edgecolor='black')
plt.title('Distribution of Output Values')
plt.xlabel('Output')
plt.ylabel('Frequency')

plt.show()
