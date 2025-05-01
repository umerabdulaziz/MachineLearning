import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = {
    'PythonSkill': [8, 6, 9, 5, 10, 4, 3, 7, 2],
    'Experience':  [5, 4, 6, 3, 7, 2, 1, 5, 1],
    'Role':        ['Data Scientist', 'Data Analyst', 'ML Engineer',
                    'Data Analyst', 'ML Engineer', 'Intern',
                    'Intern', 'Data Scientist', 'Intern']
}

df = pd.DataFrame(data)

le = LabelEncoder()
df['Role_encoded'] = le.fit_transform(df['Role'])

x = df[['PythonSkill', 'Experience']]
y = df['Role_encoded']
print(df['Role_encoded'])
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(x,y)

new_candidates = [[7, 4], [10, 7], [3, 1]]
predicted_roles = model.predict(new_candidates)

decoded_roles = le.inverse_transform(predicted_roles)
plt.plot(predicted_roles)
plt.show()
