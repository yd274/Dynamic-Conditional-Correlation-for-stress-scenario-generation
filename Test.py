import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from DCC_class import *
from Utilities import *
import matplotlib.pyplot as plt

spx = pd.read_csv('Data/SPX.csv')
spx1 = pd.read_csv('Data/SPX1.csv')
spx['Date'] = pd.to_datetime(spx['Date'])
spx1['Date'] = pd.to_datetime(spx1['Date'])
spx = pd.concat([spx, spx1])
spx = spx.set_index('Date')
spx['Price'] = spx['Price'].str.replace(',', '', regex=True)
spx['Price'] = spx['Price'].astype(float)
spx = spx[['Price']]
spx = spx[::-1]
spx.columns = ['SPX']

ust = pd.read_excel('Data/DGS10.xlsx', sheet_name='Daily')
ust.columns = ['Date', 'UST']
ust = ust.set_index('Date')
ust.index = pd.to_datetime(ust.index)

data = pd.concat([spx, ust], axis=1)
data = data.dropna()
data = data.resample('ME').last()
data['SPX'] = np.log(data['SPX']).diff() * 100
data['UST'] = data['UST'].diff() * 100
data = data.dropna()

reg = OLS(data['SPX'], sm.add_constant(data['UST'])).fit()
print(reg.summary())

dcc = DCC(data, vol_asymmetry=True).fit()
print(dcc.summary())

conditional_corr = dcc.conditional_cov()['Conditional correlation']
corr = []
for i in range(len(conditional_corr)):
    corr.append(conditional_corr[i][1, 0])

corr = pd.DataFrame(corr)
corr.index = data.index

plt.plot(corr)
plt.axhline(np.corrcoef(data.T)[1][0], color='k')
plt.title('Correlation correlation')
plt.show()

partial_views = [[-10]]
partial_names = ['SPX']
start = pd.to_datetime('07/10/31', format='%y/%m/%d')
end = pd.to_datetime('09/12/31', format='%y/%m/%d')
prediction_GFC = SPX_UST(data, partial_views, partial_names, **{'start': start, 'end': end, 'criterion': 'Min'})
start = pd.to_datetime('22/12/31', format='%y/%m/%d')
end = pd.to_datetime('25/12/31', format='%y/%m/%d')
prediction_rates_up = SPX_UST(data, partial_views, partial_names, **{'start': start, 'end': end, 'criterion': 'Max'})