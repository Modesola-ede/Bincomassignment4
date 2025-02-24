import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

file_path = r"C:\Users\Modesola\Downloads\advertising_sales.csv"
df = pd.read_csv(file_path)
print(df.head())
#print(df.columns.tolist())

df.rename(columns={
    'TV Spend ($1000s)': 'TV Spend',
    'Sales ($1000s)': 'Sales'
}, inplace=True)

plt.xlabel('TV Spend (US$)')
plt.ylabel('Sales (US$)')
plt.scatter(df['TV Spend'], df['Sales'], color='purple', marker='+')

reg = linear_model.LinearRegression()
reg.fit(df[['TV Spend']], df['Sales'])

plt.plot(df['TV Spend'], reg.predict(df[['TV Spend']]), color='cyan')

plt.show()
