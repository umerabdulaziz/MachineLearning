data = {
    'Age': [22, 25, 47, 52, 46, 56, 23, 56, 55, 60],
    'Salary': [15000, 29000, 48000, 60000, 52000, 80000, 20000, 68000, 62000, 70000],
    'Buy':    [0,     0,     1,     1,     1,     1,     0,     1,     1,     1]
}
# IMPORTS
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df = pd.DataFrame(data)

x = df[['Age', 'Salary']]
y = df['Buy']

model = LogisticRegression()
model.fit(x, y)

prediction = model.predict([[47, 48000], [45, 32000], [28, 22000], [52, 60000]])
print("Will Buy (1 = Yes, 0 = No):", prediction)

plt.plot(prediction)
plt.show()