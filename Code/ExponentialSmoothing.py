from importlib import reload
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from Data import Dataset
reload(Dataset)
from Data.Dataset import Dataset
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings("ignore")
from matplotlib.lines import Line2D
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import scipy.stats as stats


class ExpSmth:

    def __init__(self, dataset, date, test_size, validation_size, target_name) -> None:
        self.dataset = dataset
        self.date = date
        self.test_size = test_size
        self.validation_size = validation_size
        self.target_name = target_name

    def plot_diagnosis(self, residuals, file_name):

        plt.figure(figsize=(12,7), dpi=600)
        
        plt.subplot(2,2,1)
        plt.plot(residuals)
        plt.title('Residuals over time')
        plt.axhline(0, color='red', linestyle='--')
        
        # 2. Histogram of residuals
        plt.subplot(2,2,2)
        sns.histplot(residuals, kde=True)
        plt.title('Residuals Histogram')
        
        # 3. Q-Q plot
        plt.subplot(2,2,3)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q plot')
        
        # 4. Autocorrelation
        plt.subplot(2,2,4)
        plot_acf(residuals, lags=20, ax=plt.gca())
        plt.title('ACF of residuals')
        
        plt.tight_layout()
        plt.savefig(file_name, bbox_inches='tight')
        plt.show()
    
    def check_model(self, smoothing_level, smoothing_trend, trend, file_name = ""):

        model = ExponentialSmoothing(self.dataset[:-(self.test_size+self.validation_size)], trend=trend, seasonal=None)
        results = model.fit(smoothing_level=smoothing_level, smoothing_trend=smoothing_trend, optimized=False)
        residuals = results.resid
        print(results.summary())
        self.plot_diagnosis(residuals, file_name)
        plt.show()

        ljung_box_results = acorr_ljungbox(residuals, lags=[5, 10, 20, 40, 60], return_df=True)

        print("\nLjung-Box Test Results:")
        print(ljung_box_results)

        print(f"AIC: {results.aic}")
        print(f"BIC: {results.bic}")

    def ExpSmth_horizons_training(self, smoothing_level, smoothing_trend, trend,  windows_num):
        
        result = []
        for test_size in range(self.test_size+self.validation_size+windows_num, self.test_size+self.validation_size-1, -1):
            errors = {3:[],6:[],12:[],24:[],36:[],48:[],60:[]}

            forecast_results, actuals = self.fit_ExpSmth(smoothing_level=smoothing_level, 
                                                        smoothing_trend=smoothing_trend, 
                                                        testing_size=test_size, 
                                                        forecast_steps=60, 
                                                        trend=trend, 
                                                        target_column=self.target_name, 
                                                        is_actuals = True)
            for h in errors.keys():
                errors[h] = np.sqrt(np.mean((forecast_results.values[:h] - actuals.values[:h]) ** 2))
            result.append(errors)
        return pd.DataFrame(result)

    def fit_ExpSmth(self, smoothing_level, smoothing_trend, testing_size, forecast_steps, trend, target_column , is_actuals = False):
        
        training_data = self.dataset[:-testing_size]
        model = ExponentialSmoothing(training_data, trend=trend, seasonal=None)

        results = model.fit(smoothing_level=smoothing_level, smoothing_trend=smoothing_trend, optimized=False)

        forecast = results.forecast(forecast_steps)

        subset_date = self.date.iloc[-testing_size : -testing_size + forecast_steps] if testing_size > forecast_steps else self.date.iloc[-forecast_steps:]

        forecast_series = pd.Series(forecast.values, index=subset_date, name=f"{target_column}_forecast")
        subset_data = self.dataset.iloc[-testing_size : -testing_size + forecast_steps] if testing_size > forecast_steps else self.dataset.iloc[-forecast_steps:]

        actuals = subset_data

        mae  = np.mean(np.abs(forecast_series.values - actuals.values))
        rmse = np.sqrt(np.mean((forecast_series.values - actuals.values) ** 2))
        mape = np.mean(np.abs((forecast_series.values - actuals.values) / actuals.values)) * 100
        
        #print(f"\nForecast Accuracy ({target_column}):")
        #print(f"  MAE  : {mae:.4f}")
        #print(f"  RMSE : {rmse:.4f}")
        #print(f"  MAPE : {mape:.4f}%")

        if is_actuals:
            return forecast_series, actuals

        return forecast_series

    def plot_ExpSmth_results(self, smoothing_level, smoothing_trend, trend, number_of_steps, file_name, model):


        plt.style.use('default')  
        plt.figure(figsize=(12,7), dpi=600)

        result = []
        plt.plot(self.date, self.dataset.values, label=f"Actual Data", color='red', linewidth=1)

        for step in range(1, number_of_steps):

            errors = {"Forecast_start_year":[], 3:[],6:[],12:[],24:[],36:[],48:[],60:[]}
            forecast_results, actuals = self.fit_ExpSmth(smoothing_level=smoothing_level, 
                                                        smoothing_trend=smoothing_trend, 
                                                        testing_size=self.test_size*step, 
                                                        forecast_steps=60, 
                                                        trend=trend, 
                                                        target_column=self.target_name, 
                                                        is_actuals = True)
            for h in errors.keys():
                if h != "Forecast_start_year":
                    errors[h] = np.sqrt(np.mean((forecast_results.values[h-1] - actuals.values[h-1]) ** 2))

            forecast_start_date = self.date.iloc[-60*step]
            errors["Forecast_start_year"] = pd.to_datetime(forecast_start_date).year
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

    def select_order(self, max_smoohting_level, max_smoothing_trend, trend):
    

        """
        Search over all subsets of variables, lag orders, and differencing orders
        to find the best VAR specification by out-of-sample RMSE.
        Uses rolling windows of 60-month forecasts to get stable RMSE estimates.

        Parameters:
            target_variable : column that must appear in every subset
            max_order       : maximum lag order to test
            testing_size    : fixed forecast horizon (60 months)
            max_vars        : cap subset size. None = all subsets.
            max_diff        : maximum differencing order to test
            n_windows       : number of rolling windows to average RMSE over
        """

        results_dict = {"Smoothing_level": [], "Smoothing_trend": [], "AIC": [], "BIC": []}
        trend_values = [None] if max_smoothing_trend==None else np.arange(0.01, max_smoothing_trend, 0.01)

        for smoothing_level in np.arange(0.01, max_smoohting_level, 0.01):
            for smoothing_trend in trend_values:

                try:
                    model = ExponentialSmoothing(self.dataset[:-(self.test_size+self.validation_size)], trend=trend, seasonal=None)
                    result = model.fit(smoothing_level=smoothing_level, smoothing_trend=smoothing_trend, optimized=False)

                except Exception as e:
                    print(f"Failed for smoothing_level={smoothing_level}, smoothing_trend={smoothing_trend}: {e}")
                    continue

                results_dict["Smoothing_level"].append(smoothing_level)
                results_dict["Smoothing_trend"].append(smoothing_trend)
                results_dict["AIC"].append(result.aic)
                results_dict["BIC"].append(result.bic)

        results_df = pd.DataFrame(results_dict)


        if results_df.empty:
            raise RuntimeError("No models were successfully evaluated.")

        best_aic_row = results_df.loc[results_df["AIC"].idxmin()]
        best_bic_row = results_df.loc[results_df["BIC"].idxmin()]

        if max_smoothing_trend == None:
            print("=" * 65)
            print(f"Best by AIC -> Smoothing_level: {best_aic_row['Smoothing_level']}, AIC: {best_aic_row['AIC']:.4f}")
            print(f"Best by BIC -> Smoothing_level: {best_bic_row['Smoothing_level']}, BIC: {best_bic_row['BIC']:.4f}")
            print("=" * 65)

            print("\nTop 10 by AIC:")
            print(results_df.nsmallest(10, "AIC")[["Smoothing_level", "AIC", "BIC"]].to_string(index=False))
            print("\nTop 10 by BIC:")
            print(results_df.nsmallest(10, "BIC")[["Smoothing_level", "AIC", "BIC"]].to_string(index=False))

        else:
            print("=" * 65)
            print(f"Best by AIC -> Smoothing_level: {int(best_aic_row['Smoothing_level'])}, Smoothing_trend: {best_aic_row['Smoothing_trend']}, AIC: {best_aic_row['AIC']:.4f}")
            print(f"Best by BIC -> Smoothing_level: {int(best_bic_row['Smoothing_level'])}, Smoothing_trend: {best_bic_row['Smoothing_trend']}, BIC: {best_bic_row['BIC']:.4f}")
            print("=" * 65)

            print("\nTop 10 by AIC:")
            print(results_df.nsmallest(10, "AIC")[["Smoothing_level","Smoothing_trend",  "AIC", "BIC"]].to_string(index=False))
            print("\nTop 10 by BIC:")
            print(results_df.nsmallest(10, "BIC")[["Smoothing_level", "Smoothing_trend", "AIC", "BIC"]].to_string(index=False))
        
        return best_aic_row, best_bic_row, results_df

    def select_order_rmse(self, max_smoohting_level, max_smoothing_trend, trend, n_windows=7, data_leakage=False):

        results_dict = {"Smoothing_level": [], "Smoothing_trend": [], "RMSE_mean": [], "RMSE_std": []}
        trend_values = [None] if max_smoothing_trend==None else np.arange(0.01, max_smoothing_trend, 0.01)

        for smoothing_level in np.arange(0.01, max_smoohting_level, 0.01):
            for smoothing_trend in trend_values:

                window_rmses = []
                w_start = 1 if data_leakage else 2
                for w in range(w_start, n_windows + 1):
                    window_test_size = (self.test_size * w)+self.validation_size

                    try:
                        forecast, actuals = self.fit_ExpSmth(smoothing_level=smoothing_level, 
                                                         smoothing_trend=smoothing_trend, 
                                                         testing_size=window_test_size, 
                                                         forecast_steps=60, 
                                                         trend=trend, 
                                                         target_column=self.target_name, 
                                                         is_actuals = True)

                        rmse = np.sqrt(np.mean((forecast.values - actuals.values) ** 2))
                        window_rmses.append(rmse)

                    except Exception as e:
                        print(f"Failed for smoothing_level={smoothing_level}, smoothing_trend={smoothing_trend}, window={w}: {e} {window_test_size}")
                        continue

                if len(window_rmses) == 0:
                    continue
                
                results_dict["Smoothing_level"].append(smoothing_level)
                results_dict["Smoothing_trend"].append(smoothing_trend)
                results_dict["RMSE_mean"].append(np.mean(window_rmses))
                results_dict["RMSE_std"].append(np.std(window_rmses))

        results_df = pd.DataFrame(results_dict)


        if results_df.empty:
            raise RuntimeError("No models were successfully evaluated.")

        best_rmse_row = results_df.loc[results_df["RMSE_mean"].idxmin()]

        if max_smoothing_trend == None:
            print("=" * 65)
            print(f"Best by RMSE -> Smoothing_level: {best_rmse_row['Smoothing_level']}, "
                f"RMSE: {best_rmse_row['RMSE_mean']:.4f} ± {best_rmse_row['RMSE_std']:.4f}")
            print("=" * 65)

            print("\nTop 10 by RMSE:")
            print(results_df.nsmallest(10, "RMSE_mean")[
                ["Smoothing_level", "RMSE_mean", "RMSE_std"]
            ].to_string(index=False))

        else:
            print("=" * 65)
            print(f"Best by RMSE -> Smoothing_level: {best_rmse_row['Smoothing_level']}, "
                f"Smoothing_trend: {best_rmse_row['Smoothing_trend']}, "
                f"RMSE: {best_rmse_row['RMSE_mean']:.4f} ± {best_rmse_row['RMSE_std']:.4f}")
            print("=" * 65)

            print("\nTop 10 by RMSE:")
            print(results_df.nsmallest(10, "RMSE_mean")[
                ["Smoothing_level", "Smoothing_trend", "RMSE_mean", "RMSE_std"]
            ].to_string(index=False))
        return best_rmse_row, results_df
        
    def select_order_rmse_appropriate(self, max_smoohting_level, max_smoothing_trend, trend, n_windows=7, data_leakage=False, lb_lags=10, lb_significance=0.05):
    

        results_dict = {"Smoothing_level": [], "Smoothing_trend": [], "RMSE_mean": [], "RMSE_std": []}
        trend_values = [None] if max_smoothing_trend==None else np.arange(0.01, max_smoothing_trend, 0.01)

        for smoothing_level in np.arange(0.01, max_smoohting_level, 0.01):
            for smoothing_trend in trend_values:
                
                # ── Single Ljung-Box check on the full training dataset ────────
                try:
                    model = ExponentialSmoothing(self.dataset[:-self.test_size], trend=trend, seasonal=None)
                    fitted = model.fit(
                        smoothing_level=smoothing_level,
                        smoothing_trend=smoothing_trend,
                        optimized=False,
                    )
                    lb_result = acorr_ljungbox(fitted.resid, lags=lb_lags, return_df=True)
                    if (lb_result["lb_pvalue"] < lb_significance).any():
                        continue                      # residuals are autocorrelated — skip
                except Exception as e:
                    print(f"LB check failed for smoothing_level={smoothing_level:.2f}, "
                        f"smoothing_trend={smoothing_trend}: {e}")
                    continue


                window_rmses = []
                w_start = 1 if data_leakage else 2
                for w in range(w_start, n_windows + 1):
                    window_test_size = (self.test_size * w)+self.validation_size

                    try:
                        forecast, actuals = self.fit_ExpSmth(smoothing_level=smoothing_level, 
                                                         smoothing_trend=smoothing_trend, 
                                                         testing_size=window_test_size, 
                                                         forecast_steps=60, 
                                                         trend=trend, 
                                                         target_column=self.target_name, 
                                                         is_actuals = True)

                        rmse = np.sqrt(np.mean((forecast.values - actuals.values) ** 2))
                        window_rmses.append(rmse)

                    except Exception as e:
                        print(f"Failed for smoothing_level={smoothing_level}, smoothing_trend={smoothing_trend}, window={w}: {e} {window_test_size}")
                        continue

                if len(window_rmses) == 0:
                    continue
                
                results_dict["Smoothing_level"].append(smoothing_level)
                results_dict["Smoothing_trend"].append(smoothing_trend)
                results_dict["RMSE_mean"].append(np.mean(window_rmses))
                results_dict["RMSE_std"].append(np.std(window_rmses))

        results_df = pd.DataFrame(results_dict)


        if results_df.empty:
            raise RuntimeError("No models were successfully evaluated.")

        best_rmse_row = results_df.loc[results_df["RMSE_mean"].idxmin()]

        if max_smoothing_trend == None:
            print("=" * 65)
            print(f"Best by RMSE -> Smoothing_level: {best_rmse_row['Smoothing_level']}, "
                f"RMSE: {best_rmse_row['RMSE_mean']:.4f} ± {best_rmse_row['RMSE_std']:.4f}")
            print("=" * 65)

            print("\nTop 10 by RMSE:")
            print(results_df.nsmallest(10, "RMSE_mean")[
                ["Smoothing_level", "RMSE_mean", "RMSE_std"]
            ].to_string(index=False))

        else:
            print("=" * 65)
            print(f"Best by RMSE -> Smoothing_level: {best_rmse_row['Smoothing_level']}, "
                f"Smoothing_trend: {best_rmse_row['Smoothing_trend']}, "
                f"RMSE: {best_rmse_row['RMSE_mean']:.4f} ± {best_rmse_row['RMSE_std']:.4f}")
            print("=" * 65)

            print("\nTop 10 by RMSE:")
            print(results_df.nsmallest(10, "RMSE_mean")[
                ["Smoothing_level", "Smoothing_trend", "RMSE_mean", "RMSE_std"]
            ].to_string(index=False))
        return best_rmse_row, results_df


def SES_energy():

    # path to the file with original dataset
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    # create the instance that will be used to store data and process it 
    Data = Dataset(filepath, sep=';')

    cpienergy = ExpSmth(Data.dataset["CPI(Energy)"],Data.dataset["Date"], 60, 80, "CPI(Energy)")


    cpienergy.select_order(max_smoohting_level = 1.000001, max_smoothing_trend = None, trend = None)
    cpienergy.select_order_rmse(max_smoohting_level = 1.000001, max_smoothing_trend = None, trend = None, n_windows=7)

    # best by AIC BIC
    cpienergy.check_model(1, None, None, "/home/fedor/Downloads/ImagesOverleaf/CPIENERGYSESRESIDUALS.png")
    result = cpienergy.ExpSmth_horizons_training(1, None, None, 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")

    # best by RMSE
    cpienergy.check_model(0.08, None, None)
    result = cpienergy.ExpSmth_horizons_training(0.08, None, None, 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")

    cpienergy.plot_ExpSmth_results(1.0, None, None, 9, "/home/fedor/Downloads/ImagesOverleaf/CPIENERGYSES60.png", "SES(1)")


def SES_food():
    # path to the file with original dataset
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    # create the instance that will be used to store data and process it 
    Data = Dataset(filepath, sep=';')
    # create class instance
    cpifood = ExpSmth(Data.dataset["CPI(Food)"],Data.dataset["Date"], 60, 80, "CPI(Food)")


    cpifood.select_order(max_smoohting_level = 1.000001, max_smoothing_trend = None, trend = None)
    cpifood.select_order_rmse(max_smoohting_level = 1.000001, max_smoothing_trend = None, trend = None, n_windows=7)

    cpifood.check_model(1, None, None)
    result = cpifood.ExpSmth_horizons_training(1, None, None, 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")


    cpifood.plot_ExpSmth_results(1, None, None, 9, "/home/fedor/Downloads/ImagesOverleaf/CPIFOODSES60.png", "SES(1)")


def DES_energy():
    # path to the file with original dataset
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    # create the instance that will be used to store data and process it 
    Data = Dataset(filepath, sep=';')

    cpienergy = ExpSmth(Data.dataset["CPI(Energy)"],Data.dataset["Date"], 60, 80, "CPI(Energy)")

    cpienergy.select_order( max_smoohting_level = 1.000001, max_smoothing_trend = 1.00001, trend = "add")
    cpienergy.select_order(max_smoohting_level = 1.000001, max_smoothing_trend = 1.00001, trend = "mul")
    cpienergy.select_order_rmse(max_smoohting_level = 1.000001, max_smoothing_trend = 1.00001, trend = "add")
    cpienergy.select_order_rmse(max_smoohting_level = 1.000001, max_smoothing_trend = 1.00001, trend = "mul")

    cpienergy.check_model(1.0, 0.01, "mul")
    result = cpienergy.ExpSmth_horizons_training(1.0, 0.01, "mul", 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")

    cpienergy.check_model(1.0, 0.01, "add", "/home/fedor/Downloads/ImagesOverleaf/CPIENERGYDESRESIDUALS.png")
    result = cpienergy.ExpSmth_horizons_training(1.0, 0.01, "add", 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")

    cpienergy.check_model(0.6, 0.23, "add")
    result = cpienergy.ExpSmth_horizons_training(0.6, 0.23, "add", 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")

    cpienergy.check_model(0.62, 0.24, "mul")
    result = cpienergy.ExpSmth_horizons_training(0.62, 0.24, "mul", 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")

    cpienergy.plot_ExpSmth_results(1.0, 0.01, "add", 9, "/home/fedor/Downloads/ImagesOverleaf/CPIENERGYDES60.png", "DES(1.0, 0.01, add)")


def DES_food():
    # path to the file with original dataset
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    # create the instance that will be used to store data and process it 
    Data = Dataset(filepath, sep=';')
    # create class instance
    cpifood = ExpSmth(Data.dataset["CPI(Food)"],Data.dataset["Date"], 60, 80, "CPI(Food)")

    cpicore.select_order( max_smoohting_level = 1.000001, max_smoothing_trend = 1.00001, trend = "add")
    cpicore.select_order_rmse(max_smoohting_level = 1.000001, max_smoothing_trend = 1.00001, trend = "add")
    # create class instance
    cpicore = ExpSmth(Data.dataset["CPI(Food)"],Data.dataset["Date"])


    cpifood.select_order( max_smoohting_level = 1.000001, max_smoothing_trend = 1.00001, trend = "add")
    cpifood.select_order(max_smoohting_level = 1.000001, max_smoothing_trend = 1.00001, trend = "mul")
    cpifood.select_order_rmse(max_smoohting_level = 1.000001, max_smoothing_trend = 1.00001, trend = "add")
    cpifood.select_order_rmse(max_smoohting_level = 1.000001, max_smoothing_trend = 1.00001, trend = "mul")

    cpifood.check_model(1.0, 0.22, "add")
    result = cpifood.ExpSmth_horizons_training(1.0, 0.22, "add", 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")
    # no correlation, normal
    # AIC = -781 BIC = -765
    # RMSE = 1.67 

    cpifood.check_model(1.0, 0.22, "mul")
    result = cpifood.ExpSmth_horizons_training(1.0, 0.22, "add", 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")
    # no correlation, normal
    # AIC = -778 BIC = -762
    # RMSE 1.67

    cpifood.check_model(0.21, 0.02, "add")
    result = cpifood.ExpSmth_horizons_training(0.21, 0.02, "add", 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")
    # correlation, trend
    # AIC = 84 BIC = 100
    # RMSE 2.82

    cpifood.check_model(0.15, 0.04, "mul")
    result = cpifood.ExpSmth_horizons_training(0.15, 0.04, "mul", 20)
    print(result)
    print(f"Mean for each horizon: {result.mean()}")
    print(f"Overall RMSE mean: {result.mean().mean()}")
    # correlation, trend
    # AIC = 295 BIC = 311
    # RMSE 3.82


    cpifood.plot_ExpSmth_results(1.0, 0.22, "add", 9, "/home/fedor/Downloads/ImagesOverleaf/CPIFOODDES60.png", "DES(1, 0.22, add)")

