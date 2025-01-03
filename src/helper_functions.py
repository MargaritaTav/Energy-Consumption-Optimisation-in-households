class Helper:

    @staticmethod
    def read_txt(filename):
        fl = open(filename, 'r')
        print(fl.read())
        fl.close


    def get_timespan(self, df, start, timedelta_params):
        import pandas as pd

        start = pd.to_datetime(start) if type(start) != type(pd.to_datetime('1970-01-01')) else start 
        end = start + pd.Timedelta(**timedelta_params)
        return df[start:end]


    @staticmethod
    def load_txt(filename):
        fl = open(filename, 'r')
        output = fl.read()
        fl.close
        return output


    def get_column_labels(self, filename):
        columns = {}
        readme = self.load_txt(filename)
        temp = readme[readme.find('\nHouse'):]

        for house in range(1, 22):
            cols = {}
            temp = readme[readme.find('\nHouse '+str(house)):]
    
            for idx in range(10):
                start = temp.find(str(idx)+'.')+2
                stop = temp.find(',') if temp.find(',') < temp.find('\n\t') else temp.find('\n\t')
                cols.update({'Appliance'+str(idx):temp[start:stop]})
                temp = temp[stop+1:]

            columns.update({house: cols})
        return columns
    def save_weather(self, REFIT_dir='..code/data/', EXPORT_PATH='../export/'):
       
        import pandas as pd
        filename = r"\household.csv"

            # Load the household data
        house = pd.read_csv(filename)
        house.set_index(pd.DatetimeIndex(house['timestamps']), inplace=True)

        # Add Weather
        ################################
        from meteostat import Point, Hourly
        from datetime import datetime

        lough = Point(52.766593, -1.223511)
        time = house.index.to_series(name="time")
        time = time.dt.floor('H').tolist()
        start_time = time[0]
        end_time = time[-1]
        weather = Hourly(lough, start_time, end_time)
        weather = weather.fetch()

        from sklearn.impute import KNNImputer
        import numpy as np

        headers = weather.columns.values

        empty_train_columns = []
        for col in weather.columns.values:
            if sum(weather[col].isnull()) == weather.shape[0]:
                empty_train_columns.append(col)
        headers = np.setdiff1d(headers, empty_train_columns)

        imputer = KNNImputer(missing_values=np.nan, n_neighbors=7, weights="distance")
        weather = imputer.fit_transform(weather)
        weather = pd.DataFrame(weather)
        time_unique = pd.date_range(start_time, end_time, freq='h')
        weather["time"] = time_unique
        house["time"] = time

        weather.columns = np.append(headers, "time")

        house = pd.merge(house, weather, how="left", on="time")
        house.drop("time", axis=1, inplace=True)
        house.set_index(pd.DatetimeIndex(house['timestamps']), inplace=True)
        ################################

        # Filter out non-numeric columns before resampling
        numeric_house = house.select_dtypes(include=[np.number])

        # Daily
        daily_weather = numeric_house.resample("24H").mean().copy()
        daily_weather.to_pickle(EXPORT_PATH + "weather_unscaled_daily.pkl")
        
        # Hourly
        hourly_weather = numeric_house.resample("60T").mean().copy()
        hourly_weather.to_pickle(EXPORT_PATH + "weather_unscaled_hourly.pkl")

    def load_household(self, REFIT_dir='../data/', weather_sel=False):
       
        import pandas as pd
        filename =  r"\household.csv"

        # Load the household data
        house = pd.read_csv(filename)
        house.set_index(pd.DatetimeIndex(house['timestamps']), inplace=True)

        if weather_sel:
            # Add Weather
            ################################
            from meteostat import Point, Hourly
            from datetime import datetime

            lough = Point(52.766593, -1.223511)
            time = house.index.to_series(name="time")
            time = time.dt.floor('H').tolist()
            start_time = time[0]
            end_time = time[-1]
            weather = Hourly(lough, start_time, end_time)
            weather = weather.fetch()

            from sklearn.impute import KNNImputer
            from sklearn.preprocessing import MinMaxScaler
            import numpy as np

            headers = weather.columns.values

            empty_train_columns = []
            for col in weather.columns.values:
                if sum(weather[col].isnull()) == weather.shape[0]:
                    empty_train_columns.append(col)
            headers = np.setdiff1d(headers, empty_train_columns)

            imputer = KNNImputer(missing_values=np.nan, n_neighbors=7, weights="distance")
            weather = imputer.fit_transform(weather)
            scaler = MinMaxScaler()
            weather = scaler.fit_transform(weather)
            weather = pd.DataFrame(weather)
            time_unique = pd.date_range(start_time, end_time, freq='h')
            weather["time"] = time_unique
            house["time"] = time

            weather.columns = np.append(headers, "time")

            house = pd.merge(house, weather, how="left", on="time")
            house.drop("time", axis=1, inplace=True)
            house.set_index(pd.DatetimeIndex(house['timestamps']), inplace=True)
            ################################

        return house

    def aggregate(self, df, resample_param):
        import pandas as pd
        
        # Convert 'timestamps' to datetime and set as index if not already done
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            df.set_index('timestamps', inplace=True)

        # Ensure index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Filter numeric columns
        numeric_df = df.select_dtypes(include=['number'])

        # Log dropped columns for debugging
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns
        if len(non_numeric_columns) > 0:
            print(f"Dropping non-numeric columns: {non_numeric_columns.tolist()}")

        # Resample and calculate the mean
        resampled_df = numeric_df.resample(resample_param).mean()

        return resampled_df.copy()



    def plot_consumption(self, df, features='all', figsize='default', threshold=None, title='Consumption'):
        import matplotlib.pyplot as plt

        df = df.copy()
        features = [column for column in df.columns if column not in ['Unix', 'Issues']] if features == 'all' else features

        fig, ax = plt.subplots(figsize=figsize) if figsize != 'default' else plt.subplots()
        if threshold != None:
            df['threshold'] = [threshold]*df.shape[0]
            ax.plot(df['threshold'], color = 'tab:red')
        for feature in features:
            ax.plot(df[feature])
        ax.legend(['threshold'] + features) if threshold != None else ax.legend(features)
        ax.set_title(title);

    def create_day_ahead_prices_df(self, FILE_PATH, filename):
      import pandas as pd
      electricity_prices1 = pd.read_csv(FILE_PATH + "\\" + filename)
      electricity_prices1["MTU (UTC)"] = electricity_prices1["MTU (UTC)"].str.split(pat = "-", n = 0).str[0]
      electricity_prices1["MTU (UTC)"] = electricity_prices1["MTU (UTC)"].str.replace("2015", "2013")

      electricity_prices2 = pd.read_csv(FILE_PATH + filename)
      electricity_prices2["MTU (UTC)"] = electricity_prices2["MTU (UTC)"].str.split(pat = "-", n = 0).str[0]
      electricity_prices2["MTU (UTC)"] = electricity_prices2["MTU (UTC)"].str.replace("2015", "2014")

      electricity_prices3 = pd.read_csv(FILE_PATH + filename)
      electricity_prices3["MTU (UTC)"] =  electricity_prices3["MTU (UTC)"].str.split(pat = "-", n = 0).str[0]

      electricity_prices = pd.concat([electricity_prices1, electricity_prices2, electricity_prices3])
      electricity_prices.columns = ["Time", "Price"]
      electricity_prices = electricity_prices.set_index(pd.DatetimeIndex(electricity_prices['Time']), drop = True)
      electricity_prices = electricity_prices["Price"]
      return electricity_prices
    
    
    
    def concat_household_scores(self, agent_scores):
        import pandas as pd
        df_names = list(list(agent_scores.values())[0].keys())
        output = {}
        for name in df_names:
            output[name] = pd.concat([scores[name] for household, scores in agent_scores.items()])
        return pd.concat(output, axis=1)
    
    
    def shiftable_device_legend(self, EXPORT_PATH):
        from os import walk
        import json
        import pandas as pd
        # get config files stored at the export path
        _, _, filenames = next(walk(EXPORT_PATH))
        config_files = [file for file in filenames if file.find('config.json') != -1]

        legend_shiftable_devices = pd.DataFrame()
        for config_file in config_files:
            config = json.load(open(EXPORT_PATH+config_file, 'r'))
            household_id = config['data']['household']
            devices = config['user_input']['shiftable_devices']
            i = 0
            for device in devices:
                legend_shiftable_devices.loc[household_id, i] = device
                i += 1

        legend_shiftable_devices.sort_index(inplace=True)        
        legend_shiftable_devices.columns.name = 'device'
        legend_shiftable_devices.index.name = 'household'
        return legend_shiftable_devices
    


