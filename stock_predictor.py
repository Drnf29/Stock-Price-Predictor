import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

#Returns data about the stock ie. Open price, High, Low, Close, Volume, etc.
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")
    
#Data Cleaning - Get rid of irrelvent data and columns
del sp500["Dividends"]
del sp500["Stock Splits"]
sp500 = sp500.loc["1995-01-01":].copy()

#Creates a column called tommorrow which shows the closing price of the previous day
sp500["Tomorrow"] = sp500["Close"].shift(-1)

#Predicts if tomorrows price is going up or down
#Binary set as 1 will be yes it went up and 0 no it went down
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    
#Random Forest is good at not overfitting to the training data
#Initialising the model - Min sample splits protects against overfitting
model = RandomForestClassifier(n_estimators = 250, min_samples_split = 50, random_state = 1)

#You don't want data leakage accidentally giving future data to predict past data
#Cross validation isn't a very good choice as it doesn't take into account time periods
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

#Can't give the other columns as it would know the answers in the future so leakage
predictors = ["Close", "Volume", "Open", "High", "Low"]

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    
    #Returns a probability that it will go up or down tomorrow 
    predictions =  model.predict_proba(test[predictors])[:,1]
    
    #Makes the model more confident on its predictions so that when it thinks it'll
    #go up it will definitely go up
    predictions[predictions >=0.6] = 1
    predictions[predictions < 0.6] = 0
    predictions = pd.Series(predictions, index = test.index, name = "Predictions")

    combined = pd.concat([test["Target"], predictions], axis = 1)
    
    return combined

#Backtesting used to see how well the model performs each year
def backtest(data, model, predictors, start = 2500, step = 250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)



predictions = backtest(sp500, model, predictors) 

predictions["Predictions"].value_counts()
precision_score(predictions["Target"],predictions["Predictions"])

#Will use these horizons to find the mean in the last 2 trading days, week, 3 months, year and 4 years
horizons = [2,5,60,250,100]
#Holds new columns
new_predictors = []


#Calculates if it has gone down or up compared to the average to see if it is
#due for a up-swing or down-swing
for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

#Gets rid of the columns that have no data due to the 4 years of data
sp500 = sp500.dropna()    

predictions = backtest(sp500, model, new_predictors) 

predictions["Predictions"].value_counts()
ps = precision_score(predictions["Target"],predictions["Predictions"])

print(ps)


 





