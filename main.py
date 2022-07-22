#sup world
from random import randint
import pandas_datareader as web
import pandas as pd
import  numpy  as np
from sklearn.mixture  import GaussianMixture
from sklearn.preprocessing import scale
import sklearn
import sklearn.mixture


import matplotlib.pyplot as plt



df  =  web.DataReader('BTC-USD', 'yahoo')
df.reset_index(inplace=True)
print(df)
data = df[['Date','High', 'Low', 'Open', 'Adj Close', 'Volume']]

dete  = pd.DataFrame(data)
#yo  watch dis ahhahaha
quotes = []
for row_set in range(0,100000):
    if row_set%2000==0: print(row_set)
    row_quant = randint(10,30)
    row_start = randint(0, len(dete)-row_quant)
    subset = dete.iloc[row_start:row_start+row_quant]

    close_Date = max(subset['Date'])
    if row_set%2000==0: print(close_Date)

    volume_gap = subset['Volume'].pct_change()
    daily_change = (subset['Adj Close'] - subset['Open']) / subset['Open']
    fract_high = (subset['High'] - subset['Open']) / subset['Open']
    fract_low = (subset['Open'] - subset['Low']) / subset['Open']
    forecast_variable = (subset['Open'].shift(-1) - subset['Open'])

    quotes.append(pd.DataFrame({'Sequence_ID':[row_set]*len(subset),
                                'close_date':[close_Date]*len(subset),
                                'volume_gap':volume_gap,
                                'daily_change':daily_change,
                                'fract_high':fract_high,
                                'fract_low':fract_low,
                                'forecast_variable':forecast_variable}))


quotes_df =  pd.concat(quotes)
print(quotes_df)
print(quotes_df.count())
quotes_df =  quotes_df.dropna(how='any')


daily_change  = np.array(quotes_df['daily_change'].values)
fract_high  = np.array(quotes_df['fract_high'].values)
fract_low  = np.array(quotes_df['fract_low'].values)
volume_gap  = np.array(quotes_df['volume_gap'].values)
forecast  = np.array(quotes_df['forecast_variable'].values)


gg  =  np.column_stack([scale(volume_gap),
                       scale(daily_change),
                       scale(fract_high),
                       scale(fract_low),
                       scale(forecast)])


n = np.arange(1, 15)
bic = np.zeros(n.shape)
print(n.shape)
models  = []




plt.plot(n,bic)
plt.show()


model  = GaussianMixture(n_components=2,
                         covariance_type="full",
                         n_init=100,
                         random_state=7)
model.fit(gg)

def print_gmm_results(gmm,gg):
    print('-'*25)
    print(f'means: {gmm.means_.ravel()}')
    print('-'*25)
    print(f'covars: {gmm.covariances_.ravel()}')
    print('-'*25)
    print(f'means: {np.sqrt(gmm.covariances_.ravel())}')
    print('-'*25)
    print(f'aic: {gmm.aic(gg):.5f}')
    print(f'bic: {gmm.bic(gg):.5f}')
    print('-'*25)


hidden_states = model.predict(gg)


print_gmm_results(model,gg)
print(f'data derived mean: {forecast.mean():.7f}')
print(f'data derived std: {forecast.std():.7f}')

for   i  in range(model.n_components):
    print("{0}th  hidden state".format(i))
    print("mean= ",model.means_[i])
    print("var =",np.diag(model.covariances_[i]))
    print()

##5th val in mean is what  the  prediction believes  will happen

