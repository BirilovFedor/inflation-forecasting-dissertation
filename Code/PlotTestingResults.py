'''from VAR import VAR_Data
from ARIMA import ARIMA_Data
from ExponentialSmoothing import ExpSmth
from LSTM_DIIF import LSTM, WindowData
from KNN import run_model
from kNNSOLO import run_model_solo
from RandomForest import run_model_RF'''
from importlib import reload
import pandas as pd
import matplotlib.pyplot as plt
from Data import Dataset
reload(Dataset)
from Data.Dataset import Dataset
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from matplotlib.lines import Line2D


def plot_stat_models_results_energy(target_variable, dataset, file_name=''):

    plt.style.use('default')  
    #plt.figure(figsize=(12,7), dpi=600)

    result = []
    model_names = []

    time_slice = dataset["Date"][-60:]
    plt.plot(time_slice, dataset[target_variable][-60:].values, label="Actual Data", color='red', linewidth=1)


    # SES
    cpienergy = ExpSmth(Data.dataset["CPI(Energy)"], Data.dataset["Date"], 60, 80, "CPI(Energy)")

    errors = {3:[], 6:[], 12:[], 24:[], 36:[], 48:[], 60:[]}
    forecast_results, actuals = cpienergy.fit_ExpSmth(smoothing_level=1.0, 
                                                    smoothing_trend=None, 
                                                    testing_size=60, 
                                                    forecast_steps=60, 
                                                    trend=None, 
                                                    target_column=target_variable, 
                                                    is_actuals=True)
    for h in errors.keys():
        errors[h] = np.sqrt(np.mean((forecast_results.values[h-1] - actuals.values[h-1]) ** 2))
    result.append(errors)
    model_names.append('SES')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="SES", 
        color='orange', 
        linestyle='--', 
        linewidth=2
    )

    # DES
    cpienergy = ExpSmth(Data.dataset["CPI(Energy)"], Data.dataset["Date"], 60, 80, "CPI(Energy)")

    errors = {3:[], 6:[], 12:[], 24:[], 36:[], 48:[], 60:[]}
    forecast_results, actuals = cpienergy.fit_ExpSmth(smoothing_level=1.0, 
                                                    smoothing_trend=0.01, 
                                                    testing_size=60, 
                                                    forecast_steps=60, 
                                                    trend="add", 
                                                    target_column=target_variable, 
                                                    is_actuals=True)
    for h in errors.keys():
        errors[h] = np.sqrt(np.mean((forecast_results.values[h-1] - actuals.values[h-1]) ** 2))
    result.append(errors)
    model_names.append('DES')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="DES", 
        color='purple', 
        linestyle='--', 
        linewidth=2
    )

        # ARIMA
    cpienergy = ARIMA_Data(Data.dataset['CPI(Energy)'], Data.dataset['Date'], 60, 80, "CPI(Energy)")

    errors = {3:[], 6:[], 12:[], 24:[], 36:[], 48:[], 60:[]}
    forecast_results, actuals = cpienergy.fit_ARIMA(order=[1, 1, 3],
                                                    forecast_steps=60,
                                                    testing_size=60,
                                                    target_column=target_variable,
                                                    is_actuals=True)
    for h in errors.keys():
        errors[h] = np.sqrt(np.mean((forecast_results.values[h-1] - actuals.values[h-1]) ** 2))
    result.append(errors)
    model_names.append('ARIMA')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="ARIMA", 
        color='green', 
        linestyle='--', 
        linewidth=2
    )

    # VAR
    cpienergy = VAR_Data(dataset.drop(["CPI(Core)", "CPI(Food)", "Date"], axis=1), dataset["Date"])
    cpienergy.differentiate(1)

    errors = {3:[], 6:[], 12:[], 24:[], 36:[], 48:[], 60:[]}
    forecast_results, actuals = cpienergy.fit_VAR(9, 60, ['CPI(Energy)', 'UNRATE', 'IP', 'GS5'], 60, target_variable, True)
    for h in errors.keys():
        errors[h] = np.sqrt(np.mean((forecast_results.values[h-1] - actuals.values[h-1]) ** 2))
    result.append(errors)
    model_names.append('VAR')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="VAR", 
        color='blue', 
        linestyle='--', 
        linewidth=2
    )




    # Grid
    plt.grid(
        True,
        linestyle='--',
        linewidth=0.7,
        alpha=0.6,
        color='gray'
    )

    for idx in [2, 5, 11, 23, 35, 47, 59]:
        plt.axvline(x=time_slice.iloc[idx], color='black', linestyle='--', alpha=0.7, linewidth=0.8)

    # Legend
    custom_lines = [
        Line2D([0], [0], color='red',    lw=1),
        Line2D([0], [0], color='blue',   lw=2, linestyle='--'),
        Line2D([0], [0], color='green',  lw=2, linestyle='--'),
        Line2D([0], [0], color='orange', lw=2, linestyle='--'),
        Line2D([0], [0], color='purple', lw=2, linestyle='--'),
    ]
    plt.legend(custom_lines, ['Actual Data', 'VAR', 'ARIMA', 'SES', 'DES'])

    plt.xlabel("Time", fontsize=12)
    plt.ylabel(target_variable, fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=10)
    plt.title(f"Model Comparison — {target_variable}", fontsize=14, fontweight='bold')

    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()

    # Results table
    df_results = pd.DataFrame(result, index=model_names)
    df_results.columns = [f'h={h}' for h in errors.keys()]
    print(df_results)


def plot_stat_models_results_food(target_variable, dataset, file_name=''):


    #plt.figure(figsize=(12,7), dpi=600)
    plt.style.use('default')  
    result = []
    model_names = []

    time_slice = dataset["Date"][-60:]
    plt.plot(time_slice, dataset[target_variable][-60:].values, label="Actual Data", color='red', linewidth=1)

        # SES
    cpienergy = ExpSmth(Data.dataset["CPI(Food)"], Data.dataset["Date"], 60, 80, "CPI(Food)")

    errors = {3:[], 6:[], 12:[], 24:[], 36:[], 48:[], 60:[]}
    forecast_results, actuals = cpienergy.fit_ExpSmth(smoothing_level=1.0, 
                                                    smoothing_trend=None, 
                                                    testing_size=60, 
                                                    forecast_steps=60, 
                                                    trend=None, 
                                                    target_column=target_variable, 
                                                    is_actuals=True)
    for h in errors.keys():
        errors[h] = np.sqrt(np.mean((forecast_results.values[h-1] - actuals.values[h-1]) ** 2))
    result.append(errors)
    model_names.append('SES')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="SES", 
        color='orange', 
        linestyle='--', 
        linewidth=2
    )

    # DES
    cpienergy = ExpSmth(Data.dataset["CPI(Food)"], Data.dataset["Date"], 60, 80, "CPI(Food)")

    errors = {3:[], 6:[], 12:[], 24:[], 36:[], 48:[], 60:[]}
    forecast_results, actuals = cpienergy.fit_ExpSmth(smoothing_level=1.0, 
                                                    smoothing_trend=0.22, 
                                                    testing_size=60, 
                                                    forecast_steps=60, 
                                                    trend="add", 
                                                    target_column=target_variable, 
                                                    is_actuals=True)
    for h in errors.keys():
        errors[h] = np.sqrt(np.mean((forecast_results.values[h-1] - actuals.values[h-1]) ** 2))
    result.append(errors)
    model_names.append('DES')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="DES", 
        color='purple', 
        linestyle='--', 
        linewidth=2
    )

    # ARIMA
    cpifood = ARIMA_Data(Data.dataset['CPI(Food)'], Data.dataset['Date'], 60, 80, "CPI(Food)")

    errors = {3:[], 6:[], 12:[], 24:[], 36:[], 48:[], 60:[]}
    forecast_results, actuals = cpifood.fit_ARIMA(order=[1, 2, 1],
                                                    forecast_steps=60,
                                                    testing_size=60,
                                                    target_column=target_variable,
                                                    is_actuals=True)
    for h in errors.keys():
        errors[h] = np.sqrt(np.mean((forecast_results.values[h-1] - actuals.values[h-1]) ** 2))
    result.append(errors)
    model_names.append('ARIMA')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="ARIMA", 
        color='green', 
        linestyle='--', 
        linewidth=2
    )

    # VAR
    cpifood = VAR_Data(dataset.drop(["CPI(Core)", "CPI(Energy)", "Date"], axis=1), dataset["Date"])
    cpifood.differentiate(1)

    errors = {3:[], 6:[], 12:[], 24:[], 36:[], 48:[], 60:[]}
    forecast_results, actuals = cpifood.fit_VAR(7, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5'], 60, target_variable, True)
    for h in errors.keys():
        errors[h] = np.sqrt(np.mean((forecast_results.values[h-1] - actuals.values[h-1]) ** 2))
    result.append(errors)
    model_names.append('VAR')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="VAR", 
        color='blue', 
        linestyle='--', 
        linewidth=2
    )



    for idx in [2, 5, 11, 23, 35, 47, 59]:
        plt.axvline(x=time_slice.iloc[idx], color='black', linestyle='--', alpha=0.7, linewidth=0.8)

    # Grid
    plt.grid(
        True,
        linestyle='--',
        linewidth=0.7,
        alpha=0.6,
        color='gray'
    )

    # Legend
    custom_lines = [
        Line2D([0], [0], color='red',    lw=1),
        Line2D([0], [0], color='blue',   lw=2, linestyle='--'),
        Line2D([0], [0], color='green',  lw=2, linestyle='--'),
        Line2D([0], [0], color='orange', lw=2, linestyle='--'),
        Line2D([0], [0], color='purple', lw=2, linestyle='--'),
    ]
    plt.legend(custom_lines, ['Actual Data', 'VAR', 'ARIMA', 'SES', 'DES'])

    plt.xlabel("Time", fontsize=12)
    plt.ylabel(target_variable, fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=10)
    plt.title(f"Model Comparison — {target_variable}", fontsize=14, fontweight='bold')

    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()

    # Results table
    df_results = pd.DataFrame(result, index=model_names)
    df_results.columns = [f'h={h}' for h in errors.keys()]
    print(df_results)


def plot_ml_models_results_energy(target_variable, dataset, file_name=''):

    plt.figure(figsize=(12,7), dpi=600)
    plt.style.use('default')  

    result = []
    model_names = []

    time_slice = dataset["Date"][-60:]
    plt.plot(time_slice, dataset[target_variable][-60:].values, label="Actual Data", color='red', linewidth=1)

    model_configs_energy_nodiff = [
            (['CPI(Energy)', 'IP'], 1, 9, 'manhattan', 'uniform', 3,  range(1,  5)),
            (['CPI(Energy)', 'UNRATE', 'IP'], 1, 20, 'manhattan', 'uniform', 6,  range(5,  9)),
            (['CPI(Energy)', 'UNRATE', 'GS5'], 1, 20, 'manhattan', 'distance', 12, range(9,  19)),
            (['CPI(Energy)', 'PPI', 'UNRATE', 'IP', 'GS5'], 3, 2, 'euclidean', 'uniform', 24, range(19, 31)),
            (['CPI(Energy)', 'IP'], 1, 20, 'manhattan', 'uniform', 36, range(31, 43)),
            (['CPI(Energy)', 'UNRATE', 'IP'], 1, 20, 'euclidean', 'uniform', 48, range(43, 55)),
            (['CPI(Energy)', 'M2'], 60, 4, 'manhattan', 'distance', 60, range(55, 61)),
        ]
    
    # kNN Multivariate
    errors = {3:[],6:[],12:[],24:[],36:[],48:[],60:[]}
    previous_prediction = dataset[target_variable].iloc[-61]
    forecast_results=[]
    for features, lag_size, n_neighbors, metric, weight, base_horizon, horizon_range in model_configs_energy_nodiff:
        for h in horizon_range:
            rmse, y_pred = run_model(
                        dataset=dataset,
                        features=features,
                        val_size=0,
                        test_size=60,
                        lag_size=lag_size,
                        forecast_horizon=h,
                        metric=metric,
                        weight=weight,
                        diff_order=0,
                        previous_prediction=previous_prediction,
                        n_neighbors=n_neighbors,
                        target_name=target_variable
            )
            previous_prediction = y_pred
            for ha in errors.keys():
                if ha == h:
                    errors[h] = rmse
            forecast_results.append(y_pred)
    result.append(errors)
    print(result)
    model_names.append('Multivariate kNN')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="Multivariate kNN", 
        color='blue', 
        linestyle='--', 
        linewidth=2
    )

    model_configs_energy_nodiff_solo2 = [
            (['CPI(Energy)'], 1, 9,  3,  range(1,  5)),
            (['CPI(Energy)'], 1, 20, 6,  range(5,  9)),
            (['CPI(Energy)'], 1, 20,  12, range(9,  19)),
            (['CPI(Energy)'], 3, 2,  24, range(19, 31)),
            (['CPI(Energy)'], 1, 20,  36, range(31, 43)),
            (['CPI(Energy)'], 1, 20,  48, range(43, 55)),
            (['CPI(Energy)'], 60, 4, 60, range(55, 61)),
        ]
    
    # kNN Univariate
    errors = {3:[],6:[],12:[],24:[],36:[],48:[],60:[]}
    previous_prediction = dataset[target_variable].iloc[-61]
    forecast_results=[]
    for features, lag_size, n_neighbors, base_horizon, horizon_range in model_configs_energy_nodiff_solo2:
        for h in horizon_range:
            rmse, y_pred = run_model_solo(
                        dataset=dataset,
                        features=features,
                        val_size=0,
                        test_size=60,
                        lag_size=lag_size,
                        forecast_horizon=h,
                        diff_order=0,
                        previous_prediction=previous_prediction,
                        n_neighbors=n_neighbors,
                        target_name=target_variable
            )
            previous_prediction = y_pred
            for ha in errors.keys():
                if ha == h:
                    errors[h] = rmse
            forecast_results.append(y_pred)
    result.append(errors)
    print(result)
    model_names.append('Univariate kNN')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="Univariate kNN", 
        color='green', 
        linestyle='--', 
        linewidth=2
    )

    model_configs_energy_rf_solo = [
            (['CPI(Energy)'], 3,  20, 20,  3,  range(1,  5)),
            (['CPI(Energy)'], 6,  20, 20,  6,  range(5,  9)),
            (['CPI(Energy)'], 60, 20, 20,  12, range(9,  19)),
            (['CPI(Energy)'], 60, 20, 20,  24, range(19, 31)),
            (['CPI(Energy)'], 60, 20, 20,  36, range(31, 43)),
            (['CPI(Energy)'], 48, 20, 20,  48, range(43, 55)),
            (['CPI(Energy)'], 24, 20, 20,  60, range(55, 61)),
        ]

    # RF Univariate
    errors = {3:[],6:[],12:[],24:[],36:[],48:[],60:[]}
    previous_prediction = dataset[target_variable].iloc[-61]
    forecast_results=[]
    for features, lag_size, n_estimators, max_depth, base_horizon, horizon_range in model_configs_energy_rf_solo:
        for h in horizon_range:
            rmse, y_pred = run_model_RF(
                        dataset=dataset,
                        features=features,
                        test_size=60,
                        lag_size=lag_size,
                        forecast_horizon=h,
                        diff_order=0,
                        previous_prediction=previous_prediction,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        target_name=target_variable
            )
            previous_prediction = y_pred
            for ha in errors.keys():
                if ha == h:
                    errors[h] = rmse
            forecast_results.append(y_pred)
    result.append(errors)
    print(result)
    model_names.append('Univariate RF')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="Univariate RF", 
        color='orange', 
        linestyle='--', 
        linewidth=2
    )

    model_configs_energy_rf = [
            (['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'], 120, 20, 10,  3,  range(1,  5)),
            (['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'], 120, 20, 10,  6,  range(5,  9)),
            (['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'], 120, 20, 10,  12, range(9,  19)),
            (['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'], 120, 20, 10,  24, range(19, 31)),
            (['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'], 24,  20, 10,  36, range(31, 43)),
            (['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'], 6,   20, 10,  48, range(43, 55)),
            (['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'], 60,  20, 10,  60, range(55, 61)),
        ]

    # RF Multivariate
    errors = {3:[],6:[],12:[],24:[],36:[],48:[],60:[]}
    previous_prediction = dataset[target_variable].iloc[-61]
    forecast_results=[]
    for features, lag_size, n_estimators, max_depth, base_horizon, horizon_range in model_configs_energy_rf:
        for h in horizon_range:
            rmse, y_pred = run_model_RF(
                        dataset=dataset,
                        features=features,
                        test_size=60,
                        lag_size=lag_size,
                        forecast_horizon=h,
                        diff_order=0,
                        previous_prediction=previous_prediction,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        target_name=target_variable
            )
            previous_prediction = y_pred
            for ha in errors.keys():
                if ha == h:
                    errors[h] = rmse
            forecast_results.append(y_pred)
    result.append(errors)
    print(result)
    model_names.append('Multivariate RF')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="Multivariate RF", 
        color='purple', 
        linestyle='--', 
        linewidth=2
    )

    # LSTM Multivariate
    errors = {3: [], 6: [], 12: [], 24: [], 36: [], 48: [], 60: []}

    window = WindowData(dataset[['CPI(Energy)', 'UNRATE', 'IP', 'GS5']], val_size=0, test_size=60,
                            lag_size=120, window_step=1, forecast_horizon=60,
                            target_name=target_variable)
    window.create_rolling_windows()

    lstm_model = LSTM(10, len(['CPI(Energy)', 'UNRATE', 'IP', 'GS5']), 120, 60)
    lstm_model.training(X=window.window_X_train, y=window.window_y_train,
                             X_val=window.window_X_val, y_val=window.window_y_val,
                             learning_rate=0.1, window_data=window, num_epoches=40)

    y_pred, y_actual = lstm_model.get_testing_results(
            X=window.window_X_test, y=window.window_y_test, window_data=window
    )

    for ha in errors.keys():
        errors[ha] = np.sqrt(np.mean((y_pred[ha - 1] - y_actual[ha - 1]) ** 2))
    result.append(errors)
    print(result)
    model_names.append('Multivariate LSTM')

    plt.plot(
        time_slice,
        [y_pred[i] for i in range(60)],
        label="Multivariate LSTM",
        color='brown',
        linestyle='--',
        linewidth=2
    )

    # LSTM Univariate
    errors = {3: [], 6: [], 12: [], 24: [], 36: [], 48: [], 60: []}

    window = WindowData(dataset[['CPI(Energy)']], val_size=0, test_size=60,
                            lag_size=120, window_step=1, forecast_horizon=60,
                            target_name=target_variable)
    window.create_rolling_windows()

    lstm_model = LSTM(10, len(['CPI(Energy)']), 120, 60)
    lstm_model.training(X=window.window_X_train, y=window.window_y_train,
                             X_val=window.window_X_val, y_val=window.window_y_val,
                             learning_rate=0.1, window_data=window, num_epoches=40)

    y_pred, y_actual = lstm_model.get_testing_results(
            X=window.window_X_test, y=window.window_y_test, window_data=window
    )

    for ha in errors.keys():
        errors[ha] = np.sqrt(np.mean((y_pred[ha - 1] - y_actual[ha - 1]) ** 2))
    result.append(errors)
    print(result)
    model_names.append('Univariate LSTM')

    plt.plot(
        time_slice,
        [y_pred[i] for i in range(60)],
        label="Univariate LSTM",
        color='cyan',
        linestyle='--',
        linewidth=2
    )

    # Grid
    plt.grid(
        True,
        linestyle='--',
        linewidth=0.7,
        alpha=0.6,
        color='gray'
    )

    for idx in [2, 5, 11, 23, 35, 47, 59]:
        plt.axvline(x=time_slice.iloc[idx], color='black', linestyle='--', alpha=0.7, linewidth=0.8)

    # Legend
    custom_lines = [
        Line2D([0], [0], color='red',    lw=1),
        Line2D([0], [0], color='blue',   lw=2, linestyle='--'),
        Line2D([0], [0], color='green',  lw=2, linestyle='--'),
        Line2D([0], [0], color='orange', lw=2, linestyle='--'),
        Line2D([0], [0], color='purple', lw=2, linestyle='--'),
        Line2D([0], [0], color='brown',  lw=2, linestyle='--'),
        Line2D([0], [0], color='cyan',   lw=2, linestyle='--'),
        Line2D([0], [0], color='black',  lw=1, linestyle='--', alpha=0.7),
    ]
    plt.legend(custom_lines, [
        'Actual Data',
        'Multivariate kNN',
        'Univariate kNN',
        'Univariate RF',
        'Multivariate RF',
        'Multivariate LSTM',
        'Univariate LSTM',
        'Forecast horizons'
    ])

    plt.xlabel("Time", fontsize=12)
    plt.ylabel(target_variable, fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=10)
    plt.title(f"ML Model Comparison — {target_variable}", fontsize=14, fontweight='bold')

    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()

    # Results table
    df_results = pd.DataFrame(result, index=model_names)
    df_results.columns = [f'h={h}' for h in errors.keys()]
    print(df_results)


def plot_ml_models_results_food(target_variable, dataset, file_name=''):

    plt.figure(figsize=(12,7), dpi=600)
    plt.style.use('default')  

    result = []
    model_names = []

    time_slice = dataset["Date"][-60:]
    plt.plot(time_slice, dataset[target_variable][-60:].values, label="Actual Data", color='red', linewidth=1)

    model_configs_food_diff = [
            (['CPI(Food)', 'UNRATE', 'IP'], 12, 20, 'euclidean', 'uniform', 3,  range(1,  5)),
            (['CPI(Food)', 'UNRATE', 'IP'], 6, 20, 'manhattan', 'distance', 6,  range(5,  9)),
            (['CPI(Food)', 'UNRATE'], 1, 12, 'euclidean', 'uniform', 12,  range(9,  19)),
            (['CPI(Food)', 'UNRATE', 'M2'], 60, 4, 'chebyshev', 'distance', 24,  range(19, 31)),
            (['CPI(Food)', 'M2', 'GS5'], 48, 4, 'chebyshev', 'distance', 36, range(31, 43)),
            (['CPI(Food)', 'M2', 'GS5'], 36, 4, 'chebyshev', 'distance', 48, range(43, 55)),
            (['CPI(Food)', 'GS5'], 36, 9, 'euclidean', 'distance', 60, range(55, 61)),
    ]   
    
    # kNN Multivariate
    errors = {3:[],6:[],12:[],24:[],36:[],48:[],60:[]}
    previous_prediction = dataset[target_variable].iloc[-61]
    forecast_results=[]
    for features, lag_size, n_neighbors, metric, weight, base_horizon, horizon_range in model_configs_food_diff:
        for h in horizon_range:
            rmse, y_pred = run_model(
                        dataset=dataset,
                        features=features,
                        val_size=0,
                        test_size=60,
                        lag_size=lag_size,
                        forecast_horizon=h,
                        metric=metric,
                        weight=weight,
                        diff_order=1,
                        previous_prediction=previous_prediction,
                        n_neighbors=n_neighbors,
                        target_name=target_variable
            )
            previous_prediction = y_pred
            for ha in errors.keys():
                if ha == h:
                    errors[h] = rmse
            forecast_results.append(y_pred)
    result.append(errors)
    print(result)
    model_names.append('Multivariate kNN')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="Multivariate kNN", 
        color='blue', 
        linestyle='--', 
        linewidth=2
    )

    model_configs_food_diff_solo = [
            (['CPI(Food)'], 12, 20, 3,  range(1,  5)),
            (['CPI(Food)'], 6, 20, 6,  range(5,  9)),
            (['CPI(Food)'], 1, 12, 12,  range(9,  19)),
            (['CPI(Food)'], 60, 4,  24,  range(19, 31)),
            (['CPI(Food)'], 48, 4,  36, range(31, 43)),
            (['CPI(Food)'], 36, 4,  48, range(43, 55)),
            (['CPI(Food)'], 36, 9,  60, range(55, 61)),
    ]   
    
    # kNN Univariate
    errors = {3:[],6:[],12:[],24:[],36:[],48:[],60:[]}
    previous_prediction = dataset[target_variable].iloc[-61]
    forecast_results=[]
    for features, lag_size, n_neighbors, base_horizon, horizon_range in model_configs_food_diff_solo:
        for h in horizon_range:
            rmse, y_pred = run_model_solo(
                        dataset=dataset,
                        features=features,
                        val_size=0,
                        test_size=60,
                        lag_size=lag_size,
                        forecast_horizon=h,
                        diff_order=1,
                        previous_prediction=previous_prediction,
                        n_neighbors=n_neighbors,
                        target_name=target_variable
            )
            previous_prediction = y_pred
            for ha in errors.keys():
                if ha == h:
                    errors[h] = rmse
            forecast_results.append(y_pred)
    result.append(errors)
    print(result)
    model_names.append('Univariate kNN')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="Univariate kNN", 
        color='green', 
        linestyle='--', 
        linewidth=2
    )

    model_configs_food_rf_solo = [
            (['CPI(Food)'],           24, 40, 20 ,  3,  range(1,  5)),
            (['CPI(Food)'],           24, 40 ,20 , 6,  range(5,  9)),
            (['CPI(Food)'],           12, 40, 20,  12, range(9,  19)),
            (['CPI(Food)'],           24,  40, 20 ,  24, range(19, 31)),
            (['CPI(Food)'],           3,  40, 20 ,  36, range(31, 43)),
            (['CPI(Food)'],           3,  40,  20 ,48, range(43, 55)),
            (['CPI(Food)'],           3,  40, 20 ,60, range(55, 61)),
        ]

    # RF Univariate
    errors = {3:[],6:[],12:[],24:[],36:[],48:[],60:[]}
    previous_prediction = dataset[target_variable].iloc[-61]
    forecast_results=[]
    for features, lag_size, n_estimators, max_depth, base_horizon, horizon_range in model_configs_food_rf_solo:
        for h in horizon_range:
            rmse, y_pred = run_model_RF(
                        dataset=dataset,
                        features=features,
                        test_size=60,
                        lag_size=lag_size,
                        forecast_horizon=h,
                        diff_order=1,
                        previous_prediction=previous_prediction,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        target_name=target_variable
            )
            previous_prediction = y_pred
            for ha in errors.keys():
                if ha == h:
                    errors[h] = rmse
            forecast_results.append(y_pred)
    result.append(errors)
    print(result)
    model_names.append('Univariate RF')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="Univariate RF", 
        color='orange', 
        linestyle='--', 
        linewidth=2
    )

    model_configs_food_rf = [
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           6, 40, 20 ,  3,  range(1,  5)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           6, 40 ,20 , 6,  range(5,  9)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           12, 40, 20,  12, range(9,  19)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           6,  40,20 ,  24, range(19, 31)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           3,  40,20 ,  36, range(31, 43)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           3,  40,  20 ,48, range(43, 55)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           3,  40, 20 ,60, range(55, 61)),
        ]

    # RF Multivariate
    errors = {3:[],6:[],12:[],24:[],36:[],48:[],60:[]}
    previous_prediction = dataset[target_variable].iloc[-61]
    forecast_results=[]
    for features, lag_size, n_estimators, max_depth, base_horizon, horizon_range in model_configs_food_rf:
        for h in horizon_range:
            rmse, y_pred = run_model_RF(
                        dataset=dataset,
                        features=features,
                        test_size=60,
                        lag_size=lag_size,
                        forecast_horizon=h,
                        diff_order=1,
                        previous_prediction=previous_prediction,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        target_name=target_variable
            )
            previous_prediction = y_pred
            for ha in errors.keys():
                if ha == h:
                    errors[h] = rmse
            forecast_results.append(y_pred)
    result.append(errors)
    print(result)
    model_names.append('Multivariate RF')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="Multivariate RF", 
        color='purple', 
        linestyle='--', 
        linewidth=2
    )

    # LSTM Multivariate
    errors = {3: [], 6: [], 12: [], 24: [], 36: [], 48: [], 60: []}

    window = WindowData(dataset[['CPI(Food)', 'UNRATE',  'IP', 'GS5']], val_size=0, test_size=60,
                            lag_size=60, window_step=1, forecast_horizon=60,
                            target_name=target_variable,  difference_cols=['CPI(Food)', 'UNRATE',  'IP', 'GS5'])
    window.create_rolling_windows()

    lstm_model = LSTM(10, len(['CPI(Food)', 'UNRATE',  'IP', 'GS5']), 48, 60)
    lstm_model.training(X=window.window_X_train, y=window.window_y_train,
                             X_val=window.window_X_val, y_val=window.window_y_val,
                             learning_rate=0.1, window_data=window, num_epoches=40)

    y_pred, y_actual = lstm_model.get_testing_results(
            X=window.window_X_test, y=window.window_y_test, window_data=window
    )

    for ha in errors.keys():
        errors[ha] = np.sqrt(np.mean((y_pred[ha - 1] - y_actual[ha - 1]) ** 2))
    result.append(errors)
    print(result)
    model_names.append('Multivariate LSTM')

    plt.plot(
        time_slice,
        [y_pred[i] for i in range(60)],
        label="Multivariate LSTM",
        color='brown',
        linestyle='--',
        linewidth=2
    )

    # LSTM Univariate
    errors = {3: [], 6: [], 12: [], 24: [], 36: [], 48: [], 60: []}

    window = WindowData(dataset[['CPI(Food)']], val_size=0, test_size=60,
                            lag_size=24, window_step=1, forecast_horizon=60,
                            target_name=target_variable, difference_cols=['CPI(Food)'])
    window.create_rolling_windows()

    lstm_model = LSTM(10, len(['CPI(Food)']), 12, 60)
    lstm_model.training(X=window.window_X_train, y=window.window_y_train,
                             X_val=window.window_X_val, y_val=window.window_y_val,
                             learning_rate=0.1, window_data=window, num_epoches=40)

    y_pred, y_actual = lstm_model.get_testing_results(
            X=window.window_X_test, y=window.window_y_test, window_data=window
    )

    for ha in errors.keys():
        errors[ha] = np.sqrt(np.mean((y_pred[ha - 1] - y_actual[ha - 1]) ** 2))
    result.append(errors)
    print(result)
    model_names.append('Univariate LSTM')

    plt.plot(
        time_slice,
        [y_pred[i] for i in range(60)],
        label="Univariate LSTM",
        color='cyan',
        linestyle='--',
        linewidth=2
    )

    # Grid
    plt.grid(
        True,
        linestyle='--',
        linewidth=0.7,
        alpha=0.6,
        color='gray'
    )

    for idx in [2, 5, 11, 23, 35, 47, 59]:
        plt.axvline(x=time_slice.iloc[idx], color='black', linestyle='--', alpha=0.7, linewidth=0.8)

    # Legend
    custom_lines = [
        Line2D([0], [0], color='red',    lw=1),
        Line2D([0], [0], color='blue',   lw=2, linestyle='--'),
        Line2D([0], [0], color='green',  lw=2, linestyle='--'),
        Line2D([0], [0], color='orange', lw=2, linestyle='--'),
        Line2D([0], [0], color='purple', lw=2, linestyle='--'),
        Line2D([0], [0], color='brown',  lw=2, linestyle='--'),
        Line2D([0], [0], color='cyan',   lw=2, linestyle='--'),
        Line2D([0], [0], color='black',  lw=1, linestyle='--', alpha=0.7),
    ]
    plt.legend(custom_lines, [
        'Actual Data',
        'Multivariate kNN',
        'Univariate kNN',
        'Univariate RF',
        'Multivariate RF',
        'Multivariate LSTM',
        'Univariate LSTM',
        'Forecast horizons'
    ])

    plt.xlabel("Time", fontsize=12)
    plt.ylabel(target_variable, fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=10)
    plt.title(f"ML Model Comparison — {target_variable}", fontsize=14, fontweight='bold')

    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()

    # Results table
    df_results = pd.DataFrame(result, index=model_names)
    df_results.columns = [f'h={h}' for h in errors.keys()]
    print(df_results)


def plot_statvsml_models_results_energy(target_variable, dataset, file_name=''):

    plt.style.use('default')  
    plt.figure(figsize=(12,7), dpi=600)

    result = []
    model_names = []

    time_slice = dataset["Date"][-60:]
    plt.plot(time_slice, dataset[target_variable][-60:].values, label="Actual Data", color='red', linewidth=1)


    # VAR
    cpienergy = VAR_Data(dataset.drop(["CPI(Core)", "CPI(Food)", "Date"], axis=1), dataset["Date"])
    cpienergy.differentiate(1)

    errors = {3:[], 6:[], 12:[], 24:[], 36:[], 48:[], 60:[]}
    forecast_results, actuals = cpienergy.fit_VAR(9, 60, ['CPI(Energy)', 'UNRATE', 'IP', 'GS5'], 60, target_variable, True)
    for h in errors.keys():
        errors[h] = np.sqrt(np.mean((forecast_results.values[h-1] - actuals.values[h-1]) ** 2))
    result.append(errors)
    model_names.append('VAR')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="VAR", 
        color='blue', 
        linestyle='--', 
        linewidth=2
    )

    # LSTM Multivariate
    errors = {3: [], 6: [], 12: [], 24: [], 36: [], 48: [], 60: []}

    window = WindowData(dataset[['CPI(Energy)', 'UNRATE', 'IP', 'GS5']], val_size=0, test_size=60,
                            lag_size=120, window_step=1, forecast_horizon=60,
                            target_name=target_variable)
    window.create_rolling_windows()

    lstm_model = LSTM(10, len(['CPI(Energy)', 'UNRATE', 'IP', 'GS5']), 120, 60)
    lstm_model.training(X=window.window_X_train, y=window.window_y_train,
                             X_val=window.window_X_val, y_val=window.window_y_val,
                             learning_rate=0.1, window_data=window, num_epoches=40)

    y_pred, y_actual = lstm_model.get_testing_results(
            X=window.window_X_test, y=window.window_y_test, window_data=window
    )

    for ha in errors.keys():
        errors[ha] = np.sqrt(np.mean((y_pred[ha - 1] - y_actual[ha - 1]) ** 2))
    result.append(errors)
    print(result)
    model_names.append('Multivariate LSTM')

    plt.plot(
        time_slice,
        [y_pred[i] for i in range(60)],
        label="Multivariate LSTM",
        color='brown',
        linestyle='--',
        linewidth=2
    )


    # Grid
    plt.grid(
        True,
        linestyle='--',
        linewidth=0.7,
        alpha=0.6,
        color='gray'
    )

    for idx in [2, 5, 11, 23, 35, 47, 59]:
        plt.axvline(x=time_slice.iloc[idx], color='black', linestyle='--', alpha=0.7, linewidth=0.8)

    # Legend
    custom_lines = [
        Line2D([0], [0], color='red',    lw=1),
        Line2D([0], [0], color='blue',   lw=2, linestyle='--'),
        Line2D([0], [0], color='brown',  lw=2, linestyle='--'),
    ]
    plt.legend(custom_lines, ['Actual Data', 'VAR', 'Multivariate LSTM'])

    plt.xlabel("Time", fontsize=12)
    plt.ylabel(target_variable, fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=10)
    plt.title(f"Model Comparison — {target_variable}", fontsize=14, fontweight='bold')

    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()

    # Results table
    df_results = pd.DataFrame(result, index=model_names)
    df_results.columns = [f'h={h}' for h in errors.keys()]
    print(df_results)


def plot_statvsml_models_results_food(target_variable, dataset, file_name=''):


    plt.figure(figsize=(12,7), dpi=600)
    plt.style.use('default')  
    result = []
    model_names = []

    time_slice = dataset["Date"][-60:]
    plt.plot(time_slice, dataset[target_variable][-60:].values, label="Actual Data", color='red', linewidth=1)


    # DES
    cpienergy = ExpSmth(Data.dataset["CPI(Food)"], Data.dataset["Date"], 60, 80, "CPI(Food)")

    errors = {3:[], 6:[], 12:[], 24:[], 36:[], 48:[], 60:[]}
    forecast_results, actuals = cpienergy.fit_ExpSmth(smoothing_level=1.0, 
                                                    smoothing_trend=0.22, 
                                                    testing_size=60, 
                                                    forecast_steps=60, 
                                                    trend="add", 
                                                    target_column=target_variable, 
                                                    is_actuals=True)
    for h in errors.keys():
        errors[h] = np.sqrt(np.mean((forecast_results.values[h-1] - actuals.values[h-1]) ** 2))
    result.append(errors)
    model_names.append('DES')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="DES", 
        color='purple', 
        linestyle='--', 
        linewidth=2
    )

    # ARIMA
    cpifood = ARIMA_Data(Data.dataset['CPI(Food)'], Data.dataset['Date'], 60, 80, "CPI(Food)")

    errors = {3:[], 6:[], 12:[], 24:[], 36:[], 48:[], 60:[]}
    forecast_results, actuals = cpifood.fit_ARIMA(order=[1, 2, 1],
                                                    forecast_steps=60,
                                                    testing_size=60,
                                                    target_column=target_variable,
                                                    is_actuals=True)
    for h in errors.keys():
        errors[h] = np.sqrt(np.mean((forecast_results.values[h-1] - actuals.values[h-1]) ** 2))
    result.append(errors)
    model_names.append('ARIMA')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="ARIMA", 
        color='green', 
        linestyle='--', 
        linewidth=2
    )

    model_configs_food_rf = [
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           6, 40, 20 ,  3,  range(1,  5)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           6, 40 ,20 , 6,  range(5,  9)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           12, 40, 20,  12, range(9,  19)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           6,  40,20 ,  24, range(19, 31)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           3,  40,20 ,  36, range(31, 43)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           3,  40,  20 ,48, range(43, 55)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           3,  40, 20 ,60, range(55, 61)),
        ]

    # RF Multivariate
    errors = {3:[],6:[],12:[],24:[],36:[],48:[],60:[]}
    previous_prediction = dataset[target_variable].iloc[-61]
    forecast_results=[]
    for features, lag_size, n_estimators, max_depth, base_horizon, horizon_range in model_configs_food_rf:
        for h in horizon_range:
            rmse, y_pred = run_model_RF(
                        dataset=dataset,
                        features=features,
                        test_size=60,
                        lag_size=lag_size,
                        forecast_horizon=h,
                        diff_order=1,
                        previous_prediction=previous_prediction,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        target_name=target_variable
            )
            previous_prediction = y_pred
            for ha in errors.keys():
                if ha == h:
                    errors[h] = rmse
            forecast_results.append(y_pred)
    result.append(errors)
    print(result)
    model_names.append('Multivariate RF')

    plt.plot(
        time_slice, 
        forecast_results, 
        label="Multivariate RF", 
        color='blue', 
        linestyle='--', 
        linewidth=2
    )



    for idx in [2, 5, 11, 23, 35, 47, 59]:
        plt.axvline(x=time_slice.iloc[idx], color='black', linestyle='--', alpha=0.7, linewidth=0.8)

    # Grid
    plt.grid(
        True,
        linestyle='--',
        linewidth=0.7,
        alpha=0.6,
        color='gray'
    )

    # Legend
    custom_lines = [
        Line2D([0], [0], color='red',    lw=1),
        Line2D([0], [0], color='purple',   lw=2, linestyle='--'),
        Line2D([0], [0], color='green',  lw=2, linestyle='--'),
        Line2D([0], [0], color='blue', lw=2, linestyle='--'),
    ]
    plt.legend(custom_lines, ['Actual Data', 'DES', 'ARIMA', 'Multivariate RF'])

    plt.xlabel("Time", fontsize=12)
    plt.ylabel(target_variable, fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=10)
    plt.title(f"Model Comparison — {target_variable}", fontsize=14, fontweight='bold')

    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()

    # Results table
    df_results = pd.DataFrame(result, index=model_names)
    df_results.columns = [f'h={h}' for h in errors.keys()]
    print(df_results)


def plot_var(dataset, target_name, target_variable, file_name=''):

    if file_name:
        plt.figure(figsize=(12,7), dpi=600)
    plt.style.use('default')  

    plt.plot(dataset["Date"], dataset[target_variable].values, label="Actual Data", color='red', linewidth=1)

    # Grid
    plt.grid(
        True,
        linestyle='--',
        linewidth=0.7,
        alpha=0.6,
        color='gray'
    )

    plt.legend(target_name)

    plt.xlabel("Time", fontsize=12)
    plt.ylabel(target_variable, fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=10)
    plt.title(f"The {target_variable} over time", fontsize=14, fontweight='bold')

    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
    #plt.show()
    print(f"Done for {target_name}")


# path to the file with original dataset
filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
# create the instance that will be used to store data and process it 
Data = Dataset(filepath, sep=';')


#plot_var(Data.dataset, "CPI Energy", "CPI(Energy)", "/home/fedor/Downloads/ImagesOverleaf/CPIENERGYPLOT.png")
#plot_var(Data.dataset, "CPI Food", "CPI(Food)", "/home/fedor/Downloads/ImagesOverleaf/CPIFOODPLOT.png")

plot_var(Data.dataset, "PPI", "PPI", "/home/fedor/Downloads/ImagesOverleaf/CPIPPIPLOT.png")
plot_var(Data.dataset, "IP", "IP", "/home/fedor/Downloads/ImagesOverleaf/CPIIPPLOT.png")
plot_var(Data.dataset, "M2", "M2", "/home/fedor/Downloads/ImagesOverleaf/CPIM2PLOT.png")
plot_var(Data.dataset, "UNRATE", "UNRATE", "/home/fedor/Downloads/ImagesOverleaf/CPIUNRATEPLOT.png")
plot_var(Data.dataset, "GS5", "GS5", "/home/fedor/Downloads/ImagesOverleaf/CPIGS5PLOT.png")
plot_var(Data.dataset, "GDP", "GDP", "/home/fedor/Downloads/ImagesOverleaf/CPIGDPPLOT.png")
#plot_statvsml_models_results_energy("CPI(Energy)", Data.dataset, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIENERGYTESTINGCOMPARIZONSTATVSMLMODELS.png")
#plot_statvsml_models_results_food("CPI(Food)", Data.dataset, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIFOODTESTINGCOMPARIZONSTATVSMLMODELS.png")

#plot_ml_models_results_energy("CPI(Energy)", Data.dataset, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIENERGYTESTINGCOMPARIZONMLMODELS.png")
#plot_ml_models_results_food("CPI(Food)", Data.dataset, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIFOODTESTINGCOMPARIZONMLMODELS.png")
#plot_models_results_energy('CPI(Energy)', Data.dataset, file_name = "/home/fedor/Downloads/ImagesOverleaf/CPIENERGYTESTINGCOMPARIZONSTATMODELS.png")
#plot_models_results_food('CPI(Food)', Data.dataset, file_name = "/home/fedor/Downloads/ImagesOverleaf/CPIFOODTESTINGCOMPARIZONSTATMODELS.png")