import pandas as pd
import mord as m
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import numpy as np

data = {
    'WaitTime': [2, 3, 5, 10, 1, 8, 4, 2],
    'ResolutionTime': [5, 10, 15, 20, 3, 18, 12, 7],
    'Satisfaction': ['Excellent', 'Good', 'Average', 'Poor', 'Excellent', 'Poor', 'Average', 'Good']
}

df = pd.DataFrame(data)

ordinal_map = {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3}
df['Satisfaction_encoded'] = df['Satisfaction'].map(ordinal_map)

x = df[['WaitTime', 'ResolutionTime']]
y = df['Satisfaction_encoded']

model = m.LogisticIT()
model.fit(x, y)

prediction = model.predict(np.array([[4, 10]]))

inverse_map = {v: k for k, v in ordinal_map.items()}
print("Predicted Satisfaction Level:", inverse_map[prediction[0]])
plt.plot(prediction)
