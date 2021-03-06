import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from datetime import datetime

## For Prophet
from fbprophet import Prophet


DATA_PATH = 'AirPassengers.csv'
SEED = 0
XX = 'Month'
YY = '#Passengers'
OUTPUT_FILE = 'AirPassenger_predictions.csv'
H = 12 # Number of observations into the future
PLOT_BOOL = True #Can turn off all plotting
YY_pretty_plot = 'Number of Passengers per Month'


def explore_plot(df, plot=PLOT_BOOL):
	'''
	Plots 2 line plots. One of all data, one by yearly data.

	Parameters:
		data_df (pd dataframe): # of passengers per month
		plot (boolean): control plotting
	'''
	if plot:
		fig, axes = plt.subplots(2,1, figsize=(13,8))
		fig.suptitle('Number of Airline Passengers per Month (1949-1960)')

		df['Year'] = pd.DatetimeIndex(df[XX]).year
		df['_month'] = pd.DatetimeIndex(df[XX]).month
		sns.lineplot(ax=axes[0], data=df, x=XX, y=YY, marker='o')
		sns.lineplot(ax=axes[1], data=df, x='_month', y=YY, hue='Year', 
			marker='o', palette='Spectral')

		axes[0].set(xlabel='Date', ylabel=YY_pretty_plot)
		axes[1].set(xlabel='Month', ylabel=YY_pretty_plot)
		plt.show()


def eval(df):
	'''
	Calculates mse, mape, stdev of error % for given df.

	Parameters:
		df (pd dataframe): # of passengers per month
	'''
	df['residuals'] = df['predict'] - df[YY]
	df['error_pct'] = df['residuals'] / df[YY]

	errorpct_std = df['error_pct'].std()
	mape = df['error_pct'].apply(lambda x: np.abs(x)).mean()
	mse = df['residuals'].apply(lambda x: x**2).mean()
	print("Mean Absolute Percentage Error: ", mape)
	print("Percentage Error Standard Deviation: ", errorpct_std)
	print("Mean Squared Error: ", mse)


def split_dataset(df, test=.2, validate=.1):
	'''
	Deterministically splits df into train, test, and validation sets.

	Parameters:
		df (pd dataframe): # of passengers per month
		test (float): % of samples for test set
		validate (float): % of samples for validation set
	Returns:
		train (pd dataframe): # of passengers per month
		test (pd dataframe): # of passengers per month
		validate (pd dataframe): # of passengers per month
	'''
	train = df.sample(frac=1-test-validate, random_state=SEED) 
	test_tmp = df.drop(train.index)
	test = test_tmp.sample(frac=test/(test+validate), random_state=SEED)
	validate = test_tmp.drop(test.index)
	
	return train, test, validate


def fit_prophet(train, test, plot=PLOT_BOOL):
	'''
	Fit fbprophet on training data.

	Parameters:
		train (pd dataframe): training data
		test (pd dataframe): test data for tuning
	Returns:
		model (fbprophet object): fitted model on training data
		plot (boolean): control plotting
	'''
	#Format for fbprophet
	train_p_format = train.rename(columns={XX:'ds', YY:'y'})
	test_p_format = test.rename(columns={XX:'ds', YY:'y'})

	#Model and params
	model = Prophet(growth="linear", changepoints=None, 
				n_changepoints=25,
				seasonality_mode="multiplicative",
				yearly_seasonality='auto', 
				weekly_seasonality=False, 
				daily_seasonality=False,
				holidays=None)
	
	model.add_seasonality(name="yearly", period=365.25, fourier_order=20)

	#fit on training
	model.fit(train_p_format)

	#predict on test
	predicted = model.predict(test_p_format)

	test_predicted = test_p_format.merge(predicted[['ds','yhat']], 
				how="left").rename(columns={'ds':XX, 'yhat':'predict', 
				'y':YY})

	#Tune with test set
	eval(test_predicted)

	if plot:
		#Plot predictions on train&test sets to visualize
		tt_p_format = train_p_format.append(test_p_format)
		tt_predicted = model.predict(tt_p_format)

		model.plot(tt_predicted)
		fig = plt.gcf()
		fig.set_size_inches(11, 7)
		plt.subplots_adjust(top=0.9)
		plt.title('Fitted Model on Train & Test Sets')
		plt.xlabel('Date')
		plt.ylabel(YY_pretty_plot)
		plt.show()

	return model


def eval_model(model, df):
	'''
	Predict on validate set, get final performance.

	Parameters:
		model (fbprophet object): fitted model on training data
		df (pd dataframe): validation data of # of passengers per month
	'''
	df_p_format = df.rename(columns={XX:'ds', YY:'y'})
	df_predict = model.predict(df_p_format)
	df = df_p_format.merge(df_predict[['ds', 'yhat']], 
				how="left").rename(columns={'ds':XX, 'yhat':'predict', 
				'y':YY})
	eval(df)


def predict(model, df, plot=PLOT_BOOL):
	'''
	Predict on future horizon with H observations, write results.

	Parameters:
		model (fbprophet object): fitted model on training data
		df (pd dataframe): full data of # of passengers per month
		plot (boolean): control plotting
	'''
	#Future ti observations + predictions
	future = model.make_future_dataframe(periods=H+1, include_history=False, 
				freq="MS")
	future = future.iloc[1: , :]
	future_predict = model.predict(future)

	#Predictions for all given data
	df_p_format = df.rename(columns={XX:'ds', YY:'y'})
	df_predict = model.predict(df_p_format)

	#All observations
	all_p_format = df_p_format.append(future)

	#All predictions
	all_predict = df_predict.append(future_predict)

	all_df = all_p_format.merge(all_predict[['ds', 'yhat']], 
				how="left").rename(columns={'ds':XX, 'yhat':'predict', 
				'y':YY})
	
	#Format and write predictions
	all_df = all_df.drop(columns=["_month", "Year"])
	all_df['Month'] = pd.to_datetime(all_df['Month'],format='%Y-%m-%d', errors='coerce').dt.to_period('m')
	all_df = all_df.set_index('Month')
	all_df.to_csv(OUTPUT_FILE)

	if plot:
		model.plot(all_predict)
		fig = plt.gcf()
		fig.set_size_inches(11, 7)
		plt.subplots_adjust(top=0.9)
		plt.xlabel('Date')
		plt.ylabel(YY_pretty_plot)
		plt.title('Predictions on Historical and Future Observations')
		plt.show()


def main():
	'''
	Plots data, splits data into training/test sets, fits model, predicts.
	'''
	custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m")
	df = pd.read_csv(DATA_PATH, engine='python', parse_dates=['Month'], 
		date_parser=custom_date_parser)

	explore_plot(df)
	train, test, validate = split_dataset(df)

	print("Training Model...")
	model = fit_prophet(train, test)

	print("Evaluating on Validation Set...")
	eval_model(model, validate)
	predict(model, df)


if __name__ == "__main__":
	main()

