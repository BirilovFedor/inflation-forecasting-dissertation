from importlib import reload
import pandas as pd
import matplotlib.pyplot as plt
from Data import Dataset
reload(Dataset)
from Data.Dataset import Dataset
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")
from matplotlib.lines import Line2D
from statsmodels.stats.diagnostic import acorr_ljungbox

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class ARIMA_Data:

    def __init__(self, dataset, date, test_size, validation_size, target_name) -> None:
        self.dataset = dataset
        self.date = date
        self.test_size = test_size
        self.validation_size = validation_size
        self.target_name = target_name

    def check_model(self, order, file_name = ""):
        model = ARIMA(self.dataset[:-(self.test_size+self.validation_size)], order=order, enforce_invertibility=True, enforce_stationarity=True)
        results = model.fit()
        print(results.summary())
        plt.style.use('default')  
        fig = results.plot_diagnostics()
        #fig.savefig(file_name, bbox_inches='tight')
        plt.show()

        residuals = results.resid

        ljung_box_results = acorr_ljungbox(residuals, lags=[5, 10, 20, 40, 60], return_df=True)

        print("\nLjung-Box Test Results:")
        print(ljung_box_results)

        print(f"AIC: {results.aic}")
        print(f"BIC: {results.bic}")

    def plot_acf(self, lags, diff_order, file_name):
        data_diff = self.dataset
        for iter in range(diff_order):
            data_diff = data_diff.diff().dropna()
        plt.style.use('default')  
        fig, ax = plt.subplots(figsize=(12,7), dpi=600)

        plot_acf(
            data_diff.iloc[:-(self.test_size + self.validation_size)],
            lags=lags,
            ax=ax
        )

        plt.grid(
            True,
            linestyle='--',
            linewidth=0.7,
            alpha=0.6,
            color='gray'
        )

        plt.title(f"ACF of {self.target_name} with difference order {diff_order}")
        plt.savefig(file_name, bbox_inches='tight')
        plt.show()

    def plot_pacf(self, lags, diff_order, file_name):

        data_diff = self.dataset
        for iter in range(diff_order):
            data_diff = data_diff.diff().dropna()

        plt.style.use('default')  
        fig, ax = plt.subplots(figsize=(12,7), dpi=600)

        plot_pacf(
            data_diff.iloc[:-(self.test_size + self.validation_size)],
            lags=lags,
            ax=ax
        )

        plt.grid(
            True,
            linestyle='--',
            linewidth=0.7,
            alpha=0.6,
            color='gray'
        )
        plt.title(f"PACF of {self.target_name} with difference order {diff_order}")
        plt.savefig(file_name, bbox_inches='tight')
        plt.show()

    def stationarity_check(self, diff_order, file_name):

        plt.style.use('default')
        plt.figure(figsize=(12,7), dpi=600)
        data_diff = self.dataset
        for iter in range(diff_order):
            data_diff = data_diff.diff().dropna()

            
        print(adfuller(self.dataset))
        print(adfuller(data_diff))

                        
        plt.plot(data_diff, label = f"{self.target_name} with d = {diff_order}", color='red', linewidth=1)
            
        # Grid
        plt.grid(
            True,
            linestyle='--',
            linewidth=0.7,
            alpha=0.6,
            color='gray'
        )

        plt.legend()
        plt.xlabel("Time", fontsize=12)
        plt.ylabel(f"{self.target_name}", fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(fontsize=10)
        plt.title(f"{self.target_name} with differencing order {diff_order} over time", fontsize=14, fontweight='bold')
        plt.savefig(file_name, bbox_inches='tight')
        plt.show()

    def fit_ARIMA(self, order, forecast_steps, testing_size, target_column, is_actuals = False):
        
        model = ARIMA(self.dataset[:-testing_size], order=order, enforce_invertibility=True, enforce_stationarity=True)
        results = model.fit()

        forecast_values = results.forecast(forecast_steps)

        subset_date = self.date.iloc[-testing_size : -testing_size + forecast_steps] if testing_size > forecast_steps else self.date.iloc[-forecast_steps:]

        forecast_series = pd.Series(forecast_values.values, index=subset_date, name=f"{target_column}_forecast")

        subset_data = self.dataset.iloc[-testing_size : -testing_size + forecast_steps] if testing_size > forecast_steps else self.dataset.iloc[-forecast_steps:]

        actuals = subset_data

        mae  = np.mean(np.abs(forecast_series.values - actuals.values))
        rmse = np.sqrt(np.mean((forecast_series.values - actuals.values) ** 2))
        mape = np.mean(np.abs((forecast_series.values - actuals.values) / actuals.values)) * 100
        

        '''print(f"\nForecast Accuracy ({target_column}):")
        #print(f"order = {order}, forecast steps = {forecast_steps}, features = {features}")
        print(f"  MAE  : {mae:.4f}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  MAPE : {mape:.4f}%")'''

        if is_actuals:
            return forecast_series, actuals

        return forecast_series

    def ARIMA_horizons_training(self, order, windows_num):
        
        result = []
        for test_size in range(self.test_size+self.validation_size+windows_num, self.test_size+self.validation_size-1, -1):
            errors = {3:[],6:[],12:[],24:[],36:[],48:[],60:[]}

            forecast_results, actuals = self.fit_ARIMA(order = order,
                                                            forecast_steps=60,
                                                            testing_size= test_size,
                                                            target_column=self.target_name,
                                                            is_actuals=True)
            for h in errors.keys():
                errors[h] = np.sqrt(np.mean((forecast_results.values[:h] - actuals.values[:h]) ** 2))
            result.append(errors)
        return pd.DataFrame(result)

    def plot_ARIMA_results(self, number_of_steps, order, model, file_name = ""):


        plt.style.use('default')  
        plt.figure(figsize=(12,7), dpi=600)
        
        # plot the results
        result = []
        plt.plot(self.date, self.dataset.values, label="Actual Data", color='red', linewidth=1)

        for step in range(1, number_of_steps):

            errors = {3:[],6:[],12:[],24:[],36:[],48:[],60:[]}
            forecast_results, actuals =  self.fit_ARIMA(order = order,
                                                        forecast_steps=60,
                                                        testing_size= 60*step,
                                                        target_column=self.target_name,
                                                        is_actuals=True)
            for h in errors.keys():
                errors[h] = np.sqrt(np.mean((forecast_results.values[h-1] - actuals.values[h-1]) ** 2))
            result.append(errors)

            time_slice = self.date[-60*step : -60*(step-1) if step > 1 else None]
            plt.plot(
                time_slice, 
                forecast_results, 
                label="Forecasted values", 
                color='blue', 
                linestyle='--', 
                linewidth=2
            )
            plt.axvline(x= self.date.values[-60*step], color='black', linestyle='--', alpha=0.7)
                
        # Grid
        plt.grid(
            True,
            linestyle='--',
            linewidth=0.7,
            alpha=0.6,
            color='gray'
        )

        # Add a custom legend
        custom_lines = [
            Line2D([0], [0], color='red', lw=1),
            Line2D([0], [0], color='blue', lw=2, linestyle='--'),
            Line2D([0], [0], color='black', lw=1, linestyle='--', alpha=0.7)
        ]
        plt.legend(custom_lines, ['Actual Data', 'Forecasted values', 'Step boundaries'])
        plt.xlabel("Time", fontsize=12)
        plt.ylabel(self.target_name, fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(fontsize=10)
        plt.title(f"Comparison of actual {self.target_name} data vs the forecast results for {model}", fontsize=14, fontweight='bold')

        plt.savefig(file_name, bbox_inches='tight')
        plt.show()
        print(pd.DataFrame(result))

    def select_order(self, p_values, d_values, q_values):
        

        results_dict = {"p": [], "d": [], "q": [], "AIC": [], "BIC": []}

        for p in p_values:
            for d in d_values:
                for q in q_values:

                    try:
                        model  = ARIMA(self.dataset[:-(self.test_size+self.validation_size)],
                                            order=(p, d, q), 
                                            enforce_invertibility=True, 
                                            enforce_stationarity= True)
                        result = model.fit()
                        results_dict["p"].append(p)
                        results_dict["d"].append(d)
                        results_dict["q"].append(q)
                        results_dict["AIC"].append(result.aic)
                        results_dict["BIC"].append(result.bic)
                    except Exception as e:
                        print(f"Failed for p={p}, d = {d}, q={q}: {e}")
                        continue                    
                                

        results_df = pd.DataFrame(results_dict)

        # Best by AIC and BIC
        best_aic_row = results_df.loc[results_df["AIC"].idxmin()]
        best_bic_row = results_df.loc[results_df["BIC"].idxmin()]

        print("=" * 65)
        print(f"Best by AIC for {self.target_name} -> p: {int(best_aic_row['p'])}, d: {int(best_aic_row['d'])}, q: {int(best_aic_row['q'])}, AIC: {best_aic_row['AIC']:.4f}")
        print(f"Best by BIC for {self.target_name} -> p: {int(best_bic_row['p'])}, d: {int(best_bic_row['d'])}, q: {int(best_bic_row['q'])}, BIC: {best_bic_row['BIC']:.4f}")
        print("=" * 65)

        print("\nTop 10 by AIC:")
        print(results_df.nsmallest(10, "AIC")[["p", "d", "q", "AIC", "BIC"]].to_string(index=False))
        print("\nTop 10 by BIC:")
        print(results_df.nsmallest(10, "BIC")[["p", "d", "q", "AIC", "BIC"]].to_string(index=False))

        return best_aic_row, best_bic_row, results_df

    def select_order_rmse(self,p_values, d_values, q_values, n_windows=7):

        results_dict = {"p": [], "d": [], "q": [], "RMSE_mean": [], "RMSE_std": []}

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    window_rmses = []
                    for w in range(2, n_windows + 1):
                        window_test_size = (self.test_size * w) + self.validation_size
                        try:
                            forecast, actuals = self.fit_ARIMA(order = [p, d, q],
                                                              forecast_steps=60,
                                                            testing_size= window_test_size,
                                                            target_column=self.target_name,
                                                            is_actuals=True)
                            rmse = np.sqrt(np.mean((forecast.values - actuals.values) ** 2))
                            window_rmses.append(rmse)
                        except Exception as e:
                            print(f"Failed for p={p}, d = {d}, q={q}: {e}")
                            continue  

                    results_dict["p"].append(p)
                    results_dict["d"].append(d)
                    results_dict["q"].append(q)
                    results_dict["RMSE_mean"].append(np.mean(window_rmses))
                    results_dict["RMSE_std"].append(np.std(window_rmses))        

        results_df = pd.DataFrame(results_dict)

        if results_df.empty:
            raise RuntimeError("No models were successfully evaluated.")

        best_rmse_row = results_df.loc[results_df["RMSE_mean"].idxmin()]

        print("=" * 65)
        print(f"Best by RMSE for {self.target_name} -> p: {int(best_rmse_row['p'])}, "
            f"d: {int(best_rmse_row['d'])}, "
            f"q: {int(best_rmse_row['q'])}, "
            f"RMSE: {best_rmse_row['RMSE_mean']:.4f} ± {best_rmse_row['RMSE_std']:.4f}")
        print("=" * 65)

        print("\nTop 10 by RMSE:")
        print(results_df.nsmallest(10, "RMSE_mean")[
            ["p", "d", "q", "RMSE_mean", "RMSE_std"]
        ].to_string(index=False))

        return best_rmse_row, results_df


def adf_test():
    # ADF test and plot of differenced data, differencing order and file name where to save plot are the parameters

    #0 path to the file with original dataset
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    # create the instance that will be used to store data and process it 
    Data = Dataset(filepath, sep=';')

    cpifood = ARIMA_Data(Data.dataset['CPI(Food)'], Data.dataset['Date'], 60, 80, "CPI(Food)")
    cpienergy = ARIMA_Data(Data.dataset['CPI(Energy)'], Data.dataset['Date'], 60, 80, "CPI(Energy)")

    cpifood.stationarity_check(1, "/home/fedor/Downloads/ImagesOverleaf/CPIFOODDIFF.png")
    cpienergy.stationarity_check(1, "/home/fedor/Downloads/ImagesOverleaf/CPIENERGYDIFF.png")


def plot_acf_pacf():

    # plot acf and pacf for the differenced data to get possible AR and MA orders 

    #0 path to the file with original dataset
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    # create the instance that will be used to store data and process it 
    Data = Dataset(filepath, sep=';')

    cpifood = ARIMA_Data(Data.dataset['CPI(Food)'], Data.dataset['Date'], 60, 80, "CPI(Food)")
    cpienergy = ARIMA_Data(Data.dataset['CPI(Energy)'], Data.dataset['Date'], 60, 80, "CPI(Energy)")

    cpifood.plot_acf(30, 1, "/home/fedor/Downloads/ImagesOverleaf/CPIFOODACF.png")
    cpifood.plot_pacf(30, 1, "/home/fedor/Downloads/ImagesOverleaf/CPIFOODPACF.png")

    cpienergy.plot_acf(30, 1, "/home/fedor/Downloads/ImagesOverleaf/CPIENERGYACF.png")
    cpienergy.plot_pacf(30, 1, "/home/fedor/Downloads/ImagesOverleaf/CPIENERGYPACF.png")


def ARIMA_energy():
    #0 path to the file with original dataset
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    # create the instance that will be used to store data and process it 
    Data = Dataset(filepath, sep=';')

    cpienergy = ARIMA_Data(Data.dataset['CPI(Energy)'], Data.dataset['Date'], 60, 80, "CPI(Energy)")

    # perform grid search to identify optimal subset of parameters
    cpienergy.select_order([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cpienergy.select_order_rmse([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # check best models to select one of them
    # best mode by AIC BIC
    cpienergy.check_model([0, 1, 3])
    result = cpienergy.ARIMA_horizons_training([0, 1, 3], 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")

    # best mode by ACF PACF
    cpienergy.check_model([0, 1, 1])
    result = cpienergy.ARIMA_horizons_training([0, 1, 1], 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")

    # best model by RMSE
    cpienergy.check_model([2, 2, 5])
    result = cpienergy.ARIMA_horizons_training([2, 2, 5], 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")

    # best model by RMSE
    cpienergy.check_model([1, 1, 3])
    result = cpienergy.ARIMA_horizons_training([1, 1, 3], 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")

    # best model by RMSE
    cpienergy.check_model([7, 2, 9])
    result = cpienergy.ARIMA_horizons_training([7, 2, 9], 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")

    # plot results of a best model
    cpienergy.plot_ARIMA_results(9, [1, 1, 3], "ARIMA(1, 1, 3)", "/home/fedor/Downloads/ImagesOverleaf/CPIENERGYARIMA.png")


def ARIMA_food():

    #0 path to the file with original dataset
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    # create the instance that will be used to store data and process it 
    Data = Dataset(filepath, sep=';')

    cpifood = ARIMA_Data(Data.dataset['CPI(Food)'], Data.dataset['Date'], 60, 80, "CPI(Food)")

    # perform grid search for the best model, based on AIC BIC and mean RMSE
    cpifood.select_order([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cpifood.select_order_rmse([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # check best models for CPI(Food) to select one of them
    # best mode by AIC BIC
    cpifood.check_model([3, 2, 4])
    result = cpifood.ARIMA_horizons_training([3, 2, 4], 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")
    # no correlation, normal
    # AIC = 346, BIC = 378
    # RMSE = 1.45
    # only 1 coefficient is insignificant

    # best mode by ACF PACF
    cpifood.check_model([5, 2, 1])
    result = cpifood.ARIMA_horizons_training([5, 2, 1], 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")
    # no correlation, not normal
    # AIC = 348 BIC = 376
    # RMSE = 1.33
    # 2 coefficients are insignificant

    # best model by RMSE
    cpifood.check_model([1, 1, 0])
    result = cpifood.ARIMA_horizons_training([1, 1, 0], 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")
    # correlation, not normal
    # AIC = 443  BIC = 451
    # RMSE = 5.08
    # significant parameters

    # best model by RMSE
    cpifood.check_model([3, 2, 8])
    result = cpifood.ARIMA_horizons_training([3, 2, 8], 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")
    # no correlation, not normal
    # AIC = 349 BIC = 397
    # RMSE = 1.33
    # 5 insignificant parameters


    # best model by RMSE
    cpifood.check_model([1, 2, 1])
    result = cpifood.ARIMA_horizons_training([1, 2, 1], 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")
    # no correlation, no normality
    # AIC = 359, BIC = 371
    # RMSE = 1.95
    # all parameters ae sigificant

    # plot results of a selected model
    cpifood.plot_ARIMA_results(9, [1, 2, 1], 'ARIMA(1, 2, 1)', "/home/fedor/Downloads/ImagesOverleaf/CPIFOODARIMA.png")





