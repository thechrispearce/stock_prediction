import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import data
goog = pd.read_csv("/Users/chrispearce/Documents/Python Mini Projects/Datasets/GOOG_2010.csv")
nflx = pd.read_csv("/Users/chrispearce/Documents/Python Mini Projects/Datasets/NFLX_2010.csv")
aapl = pd.read_csv("/Users/chrispearce/Documents/Python Mini Projects/Datasets/AAPL_2000.csv")
amzn = pd.read_csv("/Users/chrispearce/Documents/Python Mini Projects/Datasets/AMZN_2000.csv")
sp = pd.read_csv("/Users/chrispearce/Documents/Python Mini Projects/Datasets/S&P500_1990.csv")
sp['Date'] = pd.to_datetime(sp.Date)

# buy/sell function
def buy_sell(price_data,short_data,long_data):
    buyprice = []
    sellprice = []
    signal = -1
    for i in range(len(price_data)):
        if short_data[i] > long_data[i]:
            if signal != 1:
                buyprice.append(1)
                sellprice.append(np.nan)
                signal = 1
            else:
                buyprice.append(np.nan)
                sellprice.append(np.nan)
        elif short_data[i] < long_data[i]:
            if signal == 1:
                buyprice.append(np.nan)
                sellprice.append(1)
                signal = 0
            else:
                buyprice.append(np.nan)
                sellprice.append(np.nan)
        else:
            buyprice.append(np.nan)
            sellprice.append(np.nan)

    return (buyprice, sellprice)

#S&P500 dataframe
sp_data = pd.DataFrame()
sp_data['Date'] = sp.Date
sp_data['Close'] = sp['Adj Close']
sp_data['SMA_30'] = sp['Adj Close'].rolling(window = 20).mean()
sp_data['SMA_100'] = sp['Adj Close'].rolling(window = 100).mean()
sp_data['SMA_365'] = sp['Adj Close'].rolling(window = 260).mean()
sp_data['30v100_buy'] = buy_sell(sp_data.Close,sp_data.SMA_30,sp_data.SMA_100)[0]
sp_data['30v100_sell'] = buy_sell(sp_data.Close,sp_data.SMA_30,sp_data.SMA_100)[1]
sp_data['100v365_buy'] = buy_sell(sp_data.Close,sp_data.SMA_100,sp_data.SMA_365)[0]
sp_data['100v365_sell'] = buy_sell(sp_data.Close,sp_data.SMA_100,sp_data.SMA_365)[1]

sp_2010_data = sp_data[sp_data['Date'] >= '01-04-2010'].reset_index()
sp_2000_data = sp_data[sp_data['Date'] >= '01-03-2000'].reset_index()

#Google dataframe
goog_data = pd.DataFrame()
goog_data['Date'] = goog.Date
goog_data['Close'] = goog['Close']
goog_data['SMA_30'] = goog['Close'].rolling(window = 20).mean()
goog_data['SMA_100'] = goog['Close'].rolling(window = 100).mean()
goog_data['SMA_365'] = goog['Close'].rolling(window = 260).mean()
goog_data['100v365_buy'] = buy_sell(goog_data.Close,goog_data.SMA_100,goog_data.SMA_365)[0]
goog_data['100v365_sell'] = buy_sell(goog_data.Close,goog_data.SMA_100,goog_data.SMA_365)[1]
goog_data['100v365_sp_buy'] = sp_2010_data['100v365_buy']
goog_data['100v365_sp_sell'] = sp_2010_data['100v365_sell']

#SMA plots
plt.figure(figsize=(13,6))
ax = plt.subplot()
ax.set_xticks([i*5*52 for i in range(31)])
ax.set_xticklabels([1990+i for i in range(31)], rotation=30)
plt.plot(sp_data['Close'], label='S&P 500', alpha = 0.3)
#plt.plot(sp_data['SMA_30'], label='S&P 500 30 day avg.', alpha = 0.3)
plt.plot(sp_data['SMA_100'], label='S&P 500 100 day avg.', alpha = 0.3)
plt.plot(sp_data['SMA_365'], label='S&P 500 1 year avg.', alpha = 0.3)
plt.scatter(sp_data.index, sp_data['100v365_buy'] * sp_data.Close, label = 'Buy', marker = '^', color = 'green')
plt.scatter(sp_data.index, sp_data['100v365_sell'] * sp_data.Close, label = 'Sell', marker = 'v', color = 'red')
#plt.scatter(sp_data.index, sp_data['30v100_buy'], label = 'Buy', marker = '^', color = 'green')
#plt.scatter(sp_data.index, sp_data['30v100_sell'], label = 'Sell', marker = 'v', color = 'red')
plt.legend(loc = 2)
#plt.show()

plt.figure(figsize=(13,6))
ax = plt.subplot()
ax.set_xticks([i*5*52 for i in range(11)])
ax.set_xticklabels([2010+i for i in range(11)], rotation=30)
plt.plot(goog_data['Close'], label='Google', alpha = 0.3)
plt.plot(goog_data['SMA_100'], label='Google 100 day avg.', alpha = 0.3)
plt.plot(goog_data['SMA_365'], label='Google 1 year avg.', alpha = 0.3)
#plt.scatter(goog_data.index, goog_data['100v365_sp_buy'] * goog_data.Close, label = 'Buy', marker = '^', color = 'green')
#plt.scatter(goog_data.index, goog_data['100v365_sp_sell'] * goog_data.Close, label = 'Sell', marker = 'v', color = 'red')
plt.scatter(goog_data.index, goog_data['100v365_buy'] * goog_data.Close, label = 'Buy', marker = '^', color = 'green')
plt.scatter(goog_data.index, goog_data['100v365_sell'] * goog_data.Close, label = 'Sell', marker = 'v', color = 'red')
plt.legend(loc = 2)
#plt.show()

plt.figure(figsize=(13,6))
plt.subplot(2,2,1)
plt.title('Google v S&P 500 since 2010')
plt.scatter(sp_2010_data['Close'],goog['Close'],s=1)
plt.subplot(2,2,2)
plt.title('Netflix v S&P 500 since 2010')
plt.scatter(sp_2010_data['Close'],nflx['Close'],s=1)
plt.subplot(2,2,3)
plt.title('Amazon v S&P 500 since 2000')
plt.scatter(sp_2000_data['Close'],amzn['Close'],s=1)
plt.subplot(2,2,4)
plt.title('Apple v S&P 500 since 2000')
plt.scatter(sp_2000_data['Close'],aapl['Close'],s=1)
plt.show()

r_sp_goog = np.corrcoef(sp_2010_data['Close'],goog['Close'])[0][1]
r_sp_nflx = np.corrcoef(sp_2010_data['Close'],nflx['Close'])[0][1]
r_sp_amzn = np.corrcoef(sp_2000_data['Close'],amzn['Close'])[0][1]
r_sp_aapl = np.corrcoef(sp_2000_data['Close'],aapl['Close'])[0][1]
r_aapl_amzn = np.corrcoef(aapl['Close'],amzn['Close'])[0][1]
r_goog_nflx = np.corrcoef(goog['Close'],nflx['Close'])[0][1]

print(r_sp_goog,r_sp_nflx,r_sp_amzn,r_sp_aapl,r_aapl_amzn,r_goog_nflx)