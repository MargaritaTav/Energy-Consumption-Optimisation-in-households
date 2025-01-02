
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from helper_functions import Helper
import matplotlib.pyplot as plt
from datetime import datetime
import shap as shap
import numpy as np
import xgboost as xgb
import xgboost
import multiprocessing

# More ML Models
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

# Glasbox Model 
import interpret
from interpret.glassbox import ExplainableBoostingClassifier

# NILM Agent
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef, mean_absolute_error
import matplotlib.pyplot as plt


class NILMAgent:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    @staticmethod
    def resample_meter(store=None, building=1, meter=1, period='1min', cutoff=1000.):
        key = f'/building{building}/elec/meter{meter}'
        m = store[key]
        v = m.values.flatten()
        t = m.index
        s = pd.Series(v, index=t).clip(0., cutoff)
        s[s < 10.] = 0.
        return s.resample('1s').ffill(limit=300).fillna(0.).resample(period).mean().tz_convert('UTC')

    @staticmethod
    def get_series(datastore, house, label, cutoff):
        filename = f'./house_{house}_labels.dat'
        labels = pd.read_csv(filename, delimiter=' ', header=None, index_col=0).to_dict()[1]

        for i in labels:
            if labels[i] == label:
                s = NILMAgent.resample_meter(datastore, house, i, '1min', cutoff)

        s.index.name = 'datetime'
        return s

    @staticmethod
    def get_status(app, threshold, min_off, min_on):
        condition = app > threshold
        d = np.diff(condition)
        idx, = d.nonzero()
        idx += 1

        if condition[0]:
            idx = np.r_[0, idx]
        if condition[-1]:
            idx = np.r_[idx, condition.size]

        idx.shape = (-1, 2)
        on_events = idx[:, 0].copy()
        off_events = idx[:, 1].copy()

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000.)
            on_events = on_events[off_duration > min_off]
            off_events = off_events[np.roll(off_duration, -1) > min_off]

            on_duration = off_events - on_events
            on_events = on_events[on_duration > min_on]
            off_events = off_events[on_duration > min_on]

        s = app.copy()
        s[:] = 0.
        for on, off in zip(on_events, off_events):
            s[on:off] = 1.

        return s

    class Power(Dataset):
        def __init__(self, meter=None, appliance=None, status=None, length=256, border=680, max_power=1., train=False):
            self.length = length
            self.border = border
            self.max_power = max_power
            self.train = train

            self.meter = meter.copy() / self.max_power
            self.appliance = appliance.copy() / self.max_power
            self.status = status.copy()

            self.epochs = (len(self.meter) - 2 * self.border) // self.length

        def __getitem__(self, index):
            i = index * self.length + self.border
            if self.train:
                i = np.random.randint(self.border, len(self.meter) - self.length - self.border)

            x = self.meter.iloc[i - self.border:i + self.length + self.border].values.astype('float32')
            y = self.appliance.iloc[i:i + self.length].values.astype('float32')
            s = self.status.iloc[i:i + self.length].values.astype('float32')
            x -= x.mean()

            return x, y, s

        def __len__(self):
            return self.epochs

    class Encoder(nn.Module):
        def __init__(self, in_features=3, out_features=1, kernel_size=3, padding=1, stride=1):
            super().__init__()
            self.conv = nn.Conv1d(in_features, out_features, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
            self.bn = nn.BatchNorm1d(out_features)
            self.drop = nn.Dropout(0.1)

        def forward(self, x):
            return self.drop(self.bn(F.relu(self.conv(x))))

    class TemporalPooling(nn.Module):
        def __init__(self, in_features=3, out_features=1, kernel_size=2):
            super().__init__()
            self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size)
            self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, padding=0)
            self.bn = nn.BatchNorm1d(out_features)
            self.drop = nn.Dropout(0.1)
            self.scale_factor = kernel_size

        def forward(self, x):
            x = self.pool(x)
            x = self.conv(x)
            x = self.bn(F.relu(x))
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='linear', align_corners=True)
            return self.drop(x)

    class Decoder(nn.Module):
        def __init__(self, in_features=3, out_features=1, kernel_size=2, stride=2):
            super().__init__()
            self.conv = nn.ConvTranspose1d(in_features, out_features, kernel_size=kernel_size, stride=stride, bias=False)
            self.bn = nn.BatchNorm1d(out_features)

        def forward(self, x):
            return F.relu(self.conv(x))

    class PTPNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=1, init_features=32):
            super().__init__()
            features = init_features
            self.encoder1 = NILMAgent.Encoder(in_channels, features)
            self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
            self.encoder2 = NILMAgent.Encoder(features, features * 2)
            self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
            self.encoder3 = NILMAgent.Encoder(features * 2, features * 4)
            self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
            self.encoder4 = NILMAgent.Encoder(features * 4, features * 8)

            self.tpool1 = NILMAgent.TemporalPooling(features * 8, features * 2, kernel_size=5)
            self.tpool2 = NILMAgent.TemporalPooling(features * 8, features * 2, kernel_size=10)
            self.tpool3 = NILMAgent.TemporalPooling(features * 8, features * 2, kernel_size=20)
            self.tpool4 = NILMAgent.TemporalPooling(features * 8, features * 2, kernel_size=30)

            self.decoder = NILMAgent.Decoder(features * 10, features)
            self.activation = nn.Conv1d(features, out_channels, kernel_size=1)

        def forward(self, x):
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.pool1(enc1))
            enc3 = self.encoder3(self.pool2(enc2))
            enc4 = self.encoder4(self.pool3(enc3))

            tp1 = self.tpool1(enc4)
            tp2 = self.tpool2(enc4)
            tp3 = self.tpool3(enc4)
            tp4 = self.tpool4(enc4)

            dec = self.decoder(torch.cat([enc4, tp1, tp2, tp3, tp4], dim=1))
            act = self.activation(dec)
            return act

    def train_model(self, model, batch_size, n_epochs, filename, train_loader, valid_loader, test_loader):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        min_loss = float('inf')

        train_losses = []
        valid_losses = []
        for epoch in range(1, n_epochs + 1):
            # Training Phase
            model.train()
            epoch_train_loss = 0
            for batch in train_loader:
                data, target_power, target_status = batch
                data = data.unsqueeze(1).to('cpu')
                target_status = target_status.to('cpu')

                optimizer.zero_grad()
                output_status = model(data).permute(0, 2, 1)
                loss = criterion(output_status, target_status)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()

            # Validation Phase
            model.eval()
            epoch_valid_loss = 0
            with torch.no_grad():
                for batch in valid_loader:
                    data, target_power, target_status = batch
                    data = data.unsqueeze(1).to('cpu')
                    target_status = target_status.to('cpu')

                    output_status = model(data).permute(0, 2, 1)
                    loss = criterion(output_status, target_status)
                    epoch_valid_loss += loss.item()

            # Averaging the losses
            train_losses.append(epoch_train_loss / len(train_loader))
            valid_losses.append(epoch_valid_loss / len(valid_loader))

            print(f'Epoch {epoch}/{n_epochs}, Train Loss: {train_losses[-1]:.5f}, Validation Loss: {valid_losses[-1]:.5f}')

            # Save the best model
            if valid_losses[-1] < min_loss:
                print(f'Validation loss improved from {min_loss:.5f} to {valid_losses[-1]:.5f}. Saving model...')
                torch.save(model.state_dict(), filename)
                min_loss = valid_losses[-1]

        return model, train_losses, valid_losses


# Preparation Agent
# ===============================================================================================
class Preparation_Agent:
    def __init__(self, REFIT_df):
        self.input = REFIT_df

    # stardard data preprocessing
    # -------------------------------------------------------------------------------------------
    def outlier_truncation(self, series, factor=1.5, verbose=0):

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        output = []
        counter = 0
        for item in (
            tqdm(series, desc=f"[outlier truncation: {series.name}]")
            if verbose != 0
            else series
        ):
            if item > upper_bound:
                output.append(int(upper_bound))
                counter += 1
            elif item < lower_bound:
                output.append(int(lower_bound))
                counter += 1

            else:
                output.append(item)
        print(
            f"[outlier truncation: {series.name}]: {counter} outliers were truncated."
        ) if verbose != 0 else None
        return output

    def truncate(self, df, features="all", factor=1.5, verbose=0):

        output = df.copy()
        features = (
            df.select_dtypes(include=["number"]).columns
            if features == "all"
            else features
        )

        for feature in features:
            time.sleep(0.2) if verbose != 0 else None
            row_nn = (
                df[feature] != 0
            )  # truncate only the values for which the device uses energy
            output.loc[row_nn, feature] = self.outlier_truncation(
                df.loc[row_nn, feature], factor=factor, verbose=verbose
            )  # Truncatation factor = 1.5 * IQR
            print("\n") if verbose != 0 else None
        return output

    def scale(self, df, features="all", kind="MinMax", verbose=0):
        output = df.copy()
        features = (
            df.select_dtypes(include=["number"]).columns
            if features == "all"
            else features
        )

        if kind == "MinMax":
            scaler = MinMaxScaler()
            output[features] = scaler.fit_transform(df[features])
            print("[MinMaxScaler] Finished scaling the data.") if verbose != 0 else None
        else:
            print("Chosen scaling method is not available.")
        return output

    # feature creation
    # -------------------------------------------------------------------------------------------
    def get_device_usage(self, df, device, threshold):
        return (df.loc[:, device] > threshold).astype("int")

    def get_last_usage(self, series):

        last_usage = []
        for idx in range(len(series)):
            shift = 1
            if pd.isna(series.shift(periods=1)[idx]):
                shift = None
            else:
                while series.shift(periods=shift)[idx] == 0:
                    shift += 1
            last_usage.append(shift)
        return last_usage

    def get_last_usages(self, df, features):

        output = pd.DataFrame()
        for feature in features:
            output["periods_since_last_" + feature] = self.get_last_usage(df[feature])
        output.set_index(df.index, inplace=True)
        return output

    def get_activity(self, df, active_appliances, threshold):

        active = pd.DataFrame(
            {appliance: df[appliance] > threshold for appliance in active_appliances}
        )
        return active.apply(any, axis=1).astype("int")

    def get_time_feature(self, df, features="all"):

        functions = {
            "hour": lambda df: df.index.hour,
            "day_of_week": lambda df: df.index.dayofweek,
            "day_name": lambda df: df.index.day_name().astype("category"),
            "month": lambda df: df.index.month,
            "month_name": lambda df: df.index.month_name().astype("category"),
        }
        if features == "all":
            output = pd.DataFrame(
                {function[0]: function[1](df) for function in functions.items()}
            )
        else:
            output = pd.DataFrame(
                {
                    function[0]: function[1](df)
                    for function in functions.items()
                    if function[0] in features
                }
            )
        output.set_index(df.index, inplace=True)
        return output

    def get_time_lags(self, df, features, lags):

        output = pd.DataFrame()
        for feature in features:
            for lag in lags:
                output[f"{feature}_lag_{lag}"] = df[feature].shift(periods=lag)
        return output

    # determining the optimal energy consumption threshold for target creation (usage, activity)
    # -------------------------------------------------------------------------------------------
    def visualize_threshold(self, df, threshold, appliances, figsize=(18, 5)):
        # data prep
        for appliance in appliances:
            df[appliance + "_usage"] = self.get_device_usage(df, appliance, threshold)
        df = df.join(self.get_time_feature(df))
        df["activity"] = self.get_activity(df, appliances, threshold)

        usage_cols = [column for column in df.columns if column.endswith("_usage")]
        columns = ["activity"] + usage_cols

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # hour
        hour = df.groupby("hour").mean()[columns]
        hour.plot(ax=axes[0])
        axes[0].set_ylim(-0.1, 1.1)
        axes[0].set_title(f"[threshold: {round(threshold, 4)}] Activity ratio per hour")

        # week
        usage_cols = [column for column in df.columns if column.endswith("_usage")]
        week = df.groupby("day_name").mean()[columns]
        week = week.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        week = week.rename(index=(lambda day: day[:3]))
        week.plot(ax=axes[1])
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].set_title(
            f"[threshold: {round(threshold, 4)}] Activity ratio per day of the week"
        )

        # month
        usage_cols = [column for column in df.columns if column.endswith("_usage")]
        month = df.groupby("month").mean()[columns]
        month.plot(ax=axes[2])
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].set_title(
            f"[threshold: {round(threshold, 4)}] Activity ratio per month"
        )

    def validate_thresholds(self, df, thresholds, appliances, figsize=(18, 5)):

        for threshold in tqdm(thresholds):
            self.visualize_threshold(df, threshold, appliances, figsize)
        time.sleep(0.2)
        print("\n")

    # pipeline functions: preparing the input for the following agents
    # -------------------------------------------------------------------------------------------
    def pipeline_activity(self, df, params):

        helper = Helper()
        df = df.copy()
        output = pd.DataFrame()

        # Data cleaning
        df = self.truncate(
            df,
            **params["truncate"],
        )
        df = self.scale(df, **params["scale"])

        # Aggregate to hour level
        df = helper.aggregate(df, **params["aggregate"])

        # Activity feature
        output["activity"] = self.get_activity(df, **params["activity"])

        ## Time feature
        output = output.join(self.get_time_feature(df, **params["time"]))

        # Activity lags
        output = output.join(self.get_time_lags(output, **params["activity_lag"]))

        # Dummy coding
        output = pd.get_dummies(output, drop_first=True)

        # Add weather possibly
        if "temp" in df.columns:
            output["temp"] = df["temp"].fillna(method="backfill")
        if "dwpt" in df.columns:
            output["dwpt"] = df["dwpt"].fillna(method="backfill")
        if "rhum" in df.columns:
            output["rhum"] = df["rhum"].fillna(method="backfill")
        if "wdir" in df.columns:
            output["wdir"] = df["wdir"].fillna(method="backfill")
        if "wspd" in df.columns:
            output["wspd"] = df["wspd"].fillna(method="backfill")
        return output

    def pipeline_load(self, df, params):

        import pandas as pd

        helper = Helper()
        df = df.copy()
        output = pd.DataFrame()

        # Data cleaning
        df = self.truncate(
            df,
            **params["truncate"],
        )
        scaled = self.scale(df, **params["scale"])

        # aggregate
        df = helper.aggregate(df, **params["aggregate"])
        scaled = helper.aggregate(scaled, **params["aggregate"])

        # Get device usage and transform to energy consumption
        for device in params["shiftable_devices"]:
            df[device + "_usage"] = self.get_device_usage(
                scaled, device, **params["device"]
            )
            output[device] = df.apply(
                lambda timestamp: timestamp[device] * timestamp[device + "_usage"],
                axis=1,
            )

        return output, scaled, df

    def pipeline_usage(self, df, params):
        helper = Helper()
        df = df.copy()
        output = pd.DataFrame()

        # Data cleaning
        df = self.truncate(
            df,
            **params["truncate"],
        )
        scaled = self.scale(df, **params["scale"])

        # Aggregate to hour level
        scaled = helper.aggregate(scaled, **params["aggregate_hour"])

        # Activity feature
        output["activity"] = self.get_activity(scaled, **params["activity"])

        # Get device usage and transform to energy consumption
        for device in params["shiftable_devices"]:
            output[device + "_usage"] = self.get_device_usage(
                scaled, device, **params["device"]
            )

        # aggregate and convert from mean to binary
        output = helper.aggregate(output, **params["aggregate_day"])
        output = output.apply(lambda x: (x > 0).astype("int"))

        # Last usage
        output = output.join(self.get_last_usages(output, output.columns))

        # Time features
        output = output.join(self.get_time_feature(output, **params["time"]))

        # lags
        output = output.join(
            self.get_time_lags(
                output,
                ["activity"]
                + [device + "_usage" for device in params["shiftable_devices"]],
                [1, 2, 3],
            )
        )
        output["active_last_2_days"] = (
            (output.activity_lag_1 == 1) | (output.activity_lag_2 == 1)
        ).astype("int")

        # dummy coding
        output = pd.get_dummies(output, drop_first=True)

        # Add weather possibly
        if "temp" in df.columns:
            output["temp"] = df["temp"]
            output["temp"].fillna(method="backfill", inplace=True)
        if "dwpt" in df.columns:
            output["dwpt"] = df["dwpt"]
            output["dwpt"].fillna(method="backfill", inplace=True)
        if "rhum" in df.columns:
            output["rhum"] = df["rhum"]
            output["rhum"].fillna(method="backfill", inplace=True)
        if "wdir" in df.columns:
            output["wdir"] = df["wdir"]
            output["wdir"].fillna(method="backfill", inplace=True)
        if "wspd" in df.columns:
            output["wspd"] = df["wspd"]
            output["wspd"].fillna(method="backfill", inplace=True)

        return output


# Activity Agent
# ===============================================================================================
class Activity_Agent:
    def __init__(self, activity_input_df):
        self.input = activity_input_df

    # train test split
    # -------------------------------------------------------------------------------------------
    def get_Xtest(self, df, date, time_delta="all", target="activity"):
        import pandas as pd
        from helper_functions import Helper

        helper = Helper()

        if time_delta == "all":
            output = df.loc[pd.to_datetime(date):, df.columns != target]
        else:
            df = helper.get_timespan(df, date, time_delta)
            output = df.loc[:, df.columns != target]
        return output

    def get_ytest(self, df, date, time_delta="all", target="activity"):
        import pandas as pd
        from helper_functions import Helper

        helper = Helper()

        if time_delta == "all":
            output = df.loc[pd.to_datetime(date):, target]
        else:
            output = helper.get_timespan(df, date, time_delta)[target]
        return output

    def get_Xtrain(self, df, date, start="2013-05-16", target="activity"):
        import pandas as pd

        if type(start) == int:
            start = pd.to_datetime(date) + pd.Timedelta(days=start)
            start = (
                pd.to_datetime("2013-05-16")
                if start < pd.to_datetime("2013-05-16")
                else start
            )
        else:
            start = pd.to_datetime(start)
        end = pd.to_datetime(date) + pd.Timedelta(seconds=-1)
        return df.loc[start:end, df.columns != target]

    def get_ytrain(self, df, date, start="2013-05-16", target="activity"):
        import pandas as pd

        if type(start) == int:
            start = pd.to_datetime(date) + pd.Timedelta(days=start)
            start = (
                pd.to_datetime("2013-05-16")
                if start < pd.to_datetime("2013-05-16")
                else start
            )
        else:
            start = pd.to_datetime(start)
        end = pd.to_datetime(date) + pd.Timedelta(seconds=-1)
        return df.loc[start:end, target]

    def train_test_split(
        self, df, date, train_start="2013-05-16", test_delta="all", target="activity"
    ):
        X_train = self.get_Xtrain(df, date, start=train_start, target=target)
        y_train = self.get_ytrain(df, date, start=train_start, target=target)
        X_test = self.get_Xtest(df, date, time_delta=test_delta, target=target)
        y_test = self.get_ytest(df, date, time_delta=test_delta, target=target)
        return X_train, y_train, X_test, y_test

    # model training and evaluation
    # -------------------------------------------------------------------------------------------
    def fit_Logit(self, X, y, max_iter=100):
        return LogisticRegression(random_state=0, max_iter=max_iter).fit(X, y)

    # Other ML Models
     # ---------------------------------------------------------------------------------------------

    def fit_knn(self, X, y, n_neighbors=15, leaf_size=30, metric='manhattan'):
        return KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, metric=metric, algorithm="auto", n_jobs=-1).fit(X, y)


    def fit_random_forest(self, X, y, max_depth=10, n_estimators=1000, max_features="log2"):
        return RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features=max_features, n_jobs=-1).fit(X, y)


    def fit_ADA(self, X, y, learning_rate=0.1, n_estimators=50):
        return AdaBoostClassifier(learning_rate=learning_rate, n_estimators=n_estimators).fit(X, y)


    def fit_XGB(self, X, y, learning_rate=0.01, max_depth=6, subsample=0.8, gamma=0.5, reg_lambda=1, reg_alpha=0):
        return xgb.XGBClassifier(verbosity=0, use_label_encoder=False, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, gamma=gamma, reg_lambda=reg_lambda, reg_alpha=reg_alpha).fit(X, y)

    def fit_EBM(self, X, y):
        return ExplainableBoostingClassifier().fit(X,y)

    def fit(self, X, y, model_type, **args):
        model = None
        if model_type == "logit":
            model = self.fit_Logit(X, y, **args)
        elif model_type == "ada":
            model = self.fit_ADA(X, y, **args)
        elif model_type == "knn":
            model = self.fit_knn(X, y, **args)
        elif model_type == "random forest":
            model = self.fit_random_forest(X, y, **args)
        elif model_type == "xgboost":
            model = self.fit_XGB(X, y, **args)
        elif model_type == "ebm":
            model = self.fit_EBM(X,y, **args)
        else:
            raise InputError("Unknown model type.")
        return model


    def predict(self, model, X):
        import numpy as np
        import pandas

        if type(model) == sklearn.linear_model.LogisticRegression:
            y_hat = model.predict_proba(X)[:,1]

        elif type(model) == sklearn.neighbors._classification.KNeighborsClassifier:
            y_hat = model.predict_proba(X)[:,1]

        elif type(model) == sklearn.ensemble._forest.RandomForestClassifier:
            y_hat = model.predict_proba(X)[:,1]

        elif type(model) ==  sklearn.ensemble._weight_boosting.AdaBoostClassifier:
            y_hat = model.predict_proba(X)[:,1]

        elif type(model) == xgboost.sklearn.XGBClassifier:
            y_hat = model.predict_proba(X)[:,1]

        elif type(model) == ExplainableBoostingClassifier:
            y_hat = model.predict_proba(X)[:,1]

        else:
            raise InputError("Unknown model type.")

        y_hat = pd.Series(y_hat, index=X.index)

        return y_hat


    def auc(self, y_true, y_hat):
        import sklearn.metrics
        return sklearn.metrics.roc_auc_score(y_true, y_hat)

    def plot_model_performance(self, auc_train, auc_test, ylim="default"):
        import matplotlib.pyplot as plt

        plt.plot(list(auc_train.keys()), list(auc_train.values()))
        plt.plot(list(auc_train.keys()), list(auc_test.values()))
        plt.xticks(list(auc_train.keys()), " ")
        plt.ylim(ylim) if ylim != "default" else None

    def evaluate(
        self, df, model_type, split_params, predict_start="2013-05-16", predict_end=-1, return_errors=False,
        weather_sel=False, xai=False, **args
):
        import pandas as pd
        import numpy as np
        from tqdm import tqdm
        import shap as shap
        import time

        dates = (
            pd.DataFrame(df.index)
            .set_index(df.index)["timestamps"]
            .apply(lambda date: str(date)[:10])
            .drop_duplicates()
        )
        predict_start = pd.to_datetime(predict_start)
        predict_end = (
            pd.to_datetime(dates[predict_end] if isinstance(dates, np.ndarray) else dates.iloc[predict_end])
            if type(predict_end) == int
            else pd.to_datetime(predict_end)
        )

        dates = dates.loc[predict_start:predict_end]
        y_true = []
        y_hat_train = {}
        y_hat_test = []
        y_hat_shap = []
        auc_train_dict = {}
        auc_test = []
        xai_time_shap = []

        predictions_list = []

        if weather_sel:
            print("Crawl weather data....")
            # Add Weather Logic Here (if applicable)

        if not xai:
            for date in tqdm(dates):
                errors = {}
                try:
                    X_train, y_train, X_test, y_test = self.train_test_split(
                        df, date, **split_params
                    )

                    # Fit model
                    model = self.fit(X_train, y_train, model_type, **args)

                    y_hat_train.update({date: self.predict(model, X_train)})
                    y_hat_test += list(self.predict(model, X_test))

                    # Evaluate train data
                    auc_train_dict.update(
                        {date: self.auc(y_train, list(y_hat_train.values())[-1])}
                    )
                    y_true += list(y_test)

                except Exception as e:
                    errors[date] = e
        else:
            print('The explainability approaches in the Activity Agent are being evaluated for model: ' + str(model_type))
            print('Start evaluation with SHAP')
            
            for date in tqdm(dates):
                errors = {}
                try:
                    X_train, y_train, X_test, y_test = self.train_test_split(
                        df, date, **split_params
                    )

                    # Fit model
                    model = self.fit(X_train, y_train, model_type)

                    # Predict
                    y_hat_train.update({date: self.predict(model, X_train)})
                    y_hat_test += list(self.predict(model, X_test))

                    # Evaluate train data
                    auc_train_dict.update(
                        {date: self.auc(y_train, list(y_hat_train.values())[-1])}
                    )
                    y_true += list(y_test)

                    # SHAP
                    start_time = time.time()

                    if model_type in ["logit", "ada", "knn", "random forest"]:
                        X_train_summary = shap.sample(X_train, 100)
                        explainer = shap.KernelExplainer(model.predict_proba, X_train_summary)
                    elif model_type == "xgboost":
                        explainer = shap.TreeExplainer(model, X_train, model_output='predict_proba')
                    else:
                        raise InputError("Unknown model type.")

                    base_value = explainer.expected_value[1]  # the mean prediction

                    for local in range(len(X_test)):
                        shap_values = explainer.shap_values(X_test.iloc[local, :])
                        contribution_to_class_1 = np.array(shap_values).sum(axis=1)[1]
                        shap_prediction = base_value + contribution_to_class_1

                        # Prediction from SHAP:
                        y_hat_shap.append(shap_prediction)

                    # Track time
                    end_time = time.time()
                    xai_time_shap.append(end_time - start_time)

                except Exception as e:
                    errors[date] = e

        auc_test = self.auc(y_true, y_hat_test)
        auc_train = np.mean(list(auc_train_dict.values()))
        predictions_list.append(y_true)
        predictions_list.append(y_hat_test)
        predictions_list.append(y_hat_shap)

        # Efficiency
        time_mean_shap = np.mean(xai_time_shap)
        print('Mean time needed for SHAP: ' + str(time_mean_shap))

        if return_errors:
            return auc_train, auc_test, auc_train_dict, time_mean_shap, predictions_list, errors
        else:
            return auc_train, auc_test, auc_train_dict, time_mean_shap, predictions_list

    # pipeline function: predicting user activity
    # -------------------------------------------------------------------------------------------
    def pipeline(self, df, date, model_type, split_params, weather_sel=False):

        if weather_sel:

            # Add Weather
            ################################
            from meteostat import Point, Hourly
            from datetime import datetime

            lough = Point(52.766593, -1.223511)
            time = df.index.to_series(name="time").tolist()
            weather = Hourly(lough, time[0], time[len(df) - 1])
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
            scaler = MinMaxScaler()
            weather = scaler.fit_transform(weather)
            weather = pd.DataFrame(weather)
            weather["time"] = time[0:len(weather)]
            df["time"] = time

            weather.columns = np.append(headers, "time")

            df = pd.merge(df, weather, how="right", on="time")
            df = df.set_index("time")
            ################################

        # train test split
        X_train, y_train, X_test, y_test = self.train_test_split(
            df, date, **split_params
        )

        # fit model
        model = self.fit(X_train, y_train, model_type)

        # predict
        return self.predict(model, X_test)


# pipeline function: predicting user activity with xai
    # -------------------------------------------------------------------------------------------
    def pipeline_xai(self, df, date, model_type, split_params, weather_sel=False):

        if weather_sel:

            # Add Weather
            ################################
            from meteostat import Point, Hourly
            from datetime import datetime

            lough = Point(52.766593, -1.223511)
            time = df.index.to_series(name="time").tolist()
            weather = Hourly(lough, time[0], time[len(df) - 1])
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
            scaler = MinMaxScaler()
            weather = scaler.fit_transform(weather)
            weather = pd.DataFrame(weather)
            weather["time"] = time[0:len(weather)]
            df["time"] = time

            weather.columns = np.append(headers, "time")

            df = pd.merge(df, weather, how="right", on="time")
            df = df.set_index("time")

            ################################

        # train test split
        X_train, y_train, X_test, y_test = self.train_test_split(
            df, date, **split_params
        )

        # fit model
        model = self.fit(X_train, y_train, model_type)

        # predict
        return self.predict(model, X_test), X_train, X_test, model


class Load_Agent:
    def __init__(self, load_input_df):
        self.input = load_input_df

    # selecting the correct data, identifying device runs, creating load profiles
    # -------------------------------------------------------------------------------------------
    def prove_start_end_date(self, df, date):
        # Convert index to datetime if not already
        df.index = pd.to_datetime(df.index)
        date = pd.to_datetime(date)

        start_date = df.index.min().strftime("%Y-%m-%d")  # Earliest date in the DataFrame
        end_date = date.strftime("%Y-%m-%d")             # Provided date

        # Ensure the start_date exists and covers at least 24 rows (hours)
        if start_date not in df.index.strftime("%Y-%m-%d").tolist() or len(df[df.index.strftime("%Y-%m-%d") == start_date]) < 24:
            start_date = (pd.to_datetime(start_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # Filter DataFrame for the date range
        df = df[start_date:end_date]  # This slices DataFrame by datetime
        return df


    def df_yesterday_date(self, df, date):
        import pandas as pd

        yesterday = (pd.to_datetime(date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return df[:yesterday]

    def load_profile_raw(self, df, shiftable_devices):
        import pandas as pd

        hours = []
        for hour in range(1, 25):
            hours.append("h" + str(hour))
        df_hours = {}

        for idx, appliance in enumerate(
            shiftable_devices
        ):
            df_hours[appliance] = pd.DataFrame(index=None, columns=hours)
            column = df[appliance]

            for i in range(len(column)):

                if (i == 0) and (column[0] > 0):
                    df_hours[appliance].loc[0, "h" + str(1)] = column[0]

                elif (column[i - 1] == 0) and (column[i] > 0):
                    for j in range(0, 24):
                        if (i + j) < len(column):
                            if column[i + j] > 0:
                                df_hours[appliance].loc[i, "h" + str(j + 1)] = column[
                                    i + j
                                ]
        return df_hours

    def load_profile_cleaned(self, df_hours):
        import numpy as np

        for app in df_hours.keys():
            for i in df_hours[app].index:
                for j in df_hours[app].columns:
                    if np.isnan(df_hours[app].loc[i, j]):
                        df_hours[app].loc[i, j:] = 0
        return df_hours

    def load_profile(self, df_hours, shiftable_devices):
        import pandas as pd

        hours = df_hours[shiftable_devices[0]].columns
        loads = pd.DataFrame(columns=hours)

        for app in df_hours.keys():
            app_mean = df_hours[app].apply(lambda x: x.mean(), axis=0)
            for hour in app_mean.index:
                loads.loc[app, hour] = app_mean[hour]

        loads = loads.fillna(0)
        return loads

    # evaluating the performance of the load agent
    # -------------------------------------------------------------------------------------------
    def get_true_loads(self, shiftable_devices):
        true_loads = self.load_profile_raw(self.input, shiftable_devices)
        true_loads = self.load_profile_cleaned(true_loads)
        for device, loads in true_loads.items():
            true_loads[device].rename(
                index=dict(enumerate(self.input.index)), inplace=True
            )
        return true_loads

    def evaluate(self, shiftable_devices, metric="mse", aggregate=True, evaluation=False):
        from tqdm import tqdm
        import pandas as pd
        import numpy as np

        tqdm.pandas()

        if metric == "mse":
            import sklearn.metrics

            metric = sklearn.metrics.mean_squared_error

        true_loads = self.get_true_loads(shiftable_devices)

        scores = {}
        if not evaluation:
            for device in shiftable_devices:
                scores[device] = true_loads[device].progress_apply(
                    lambda row: metric(
                        row.values,
                        self.pipeline(
                            self.input, str(row.name)[:10], [device]
                        ).values.reshape(
                            -1,
                        ),
                    ),
                    axis=1,
                )
        else:
            for device in shiftable_devices:
                scores[device] = {}
                for idx in tqdm(true_loads[device].index):
                    date = str(idx)[:10]
                    y_true = true_loads[device].loc[idx, :].values
                    try:
                        y_hat = (
                            evaluation[date]
                            .loc[device]
                            .values.reshape(
                                -1,
                            )
                        )
                    except KeyError:
                        try:
                            y_hat = self.pipeline(
                                self.input, date, [device]
                            ).values.reshape(
                                -1,
                            )
                        except:
                            y_hat = np.full(24, 0)
                    scores[device][idx] = metric(y_true, y_hat)
                scores[device] = pd.Series(scores[device])

        if aggregate:
            scores = {device: scores_df.mean() for device, scores_df in scores.items()}
        return scores

    # pipeline function: creating typical load profiles
    # -------------------------------------------------------------------------------------------
    def pipeline(self, df, date, shiftable_devices):
        df = self.prove_start_end_date(df, date)
        df = self.df_yesterday_date(df, date)
        df_hours = self.load_profile_raw(df, shiftable_devices)
        df_hours = self.load_profile_cleaned(df_hours)
        loads = self.load_profile(df_hours, shiftable_devices)
        return loads

# Price Agent
# ===============================================================================================================
class Price_Agent:
    def __init__(self, Prices_df):
        self.input = Prices_df

    # pipeline function: return day ahead prices
    # -------------------------------------------------------------------------------------------
    def return_day_ahead_prices(self, Date):
        import pandas as pd

        range = pd.date_range(start=Date, freq="H", periods=48)
        prices = self.input.loc[range]
        return prices


# Usage Agent
# ===============================================================================================
class Usage_Agent:
    import pandas as pd

    def __init__(self, input_df, device):
        self.input = input_df
        self.device = device

    # train test split
    # -------------------------------------------------------------------------------------------
    def train_test_split(self, df, date, train_start="2013-05-16"):
        df.columns = df.columns.map(str)
        select_vars = [
            self.device + "_usage",
            self.device + "_usage_lag_1",
            self.device + "_usage_lag_2",
            "active_last_2_days",
        ]
        # Add weather possibly
        if "temp" in df.columns:
            select_vars.append("temp")
            df["temp"].fillna(method="backfill", inplace=True)
        if "dwpt" in df.columns:
            select_vars.append("dwpt")
            df["dwpt"].fillna(method="backfill", inplace=True)
        if "rhum" in df.columns:
            select_vars.append("rhum")
            df["rhum"].fillna(method="backfill", inplace=True)
        if "wdir" in df.columns:
            select_vars.append("wdir")
            df["wdir"].fillna(method="backfill", inplace=True)
        if "wspd" in df.columns:
            select_vars.append("wspd")
            df["wspd"].fillna(method="backfill", inplace=True)

        df = df[select_vars]
        X_train = df.loc[train_start:date, df.columns != self.device + "_usage"]
        y_train = df.loc[train_start:date, df.columns == self.device + "_usage"]
        X_test = df.loc[pd.to_datetime(date), df.columns != self.device + "_usage"]
        y_test = df.loc[pd.to_datetime(date), df.columns == self.device + "_usage"]
        return X_train, y_train, X_test, y_test

    
    # model training and evaluation
    # -------------------------------------------------------------------------------------------
    def fit_Logit(self, X, y, max_iter=100):
        return LogisticRegression(random_state=0, max_iter=max_iter).fit(X, y)

    # Other ML Models
    # ---------------------------------------------------------------------------------------------

    def fit_knn(self, X, y, n_neighbors=15, leaf_size=20):
        return KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, algorithm="auto", n_jobs=-1).fit(X, y)

    def fit_random_forest(self, X, y, max_depth=10, n_estimators=1000, max_features="log2"):
        return RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features=max_features, n_jobs=-1).fit(X, y)

    def fit_ADA(self, X, y, learning_rate=0.1, n_estimators=100):
        return AdaBoostClassifier(learning_rate=learning_rate, n_estimators=n_estimators).fit(X, y)
    
    def fit_XGB(self, X, y, learning_rate=0.05, max_depth=8, reg_lambda=0.5, reg_alpha=0.5):
        return xgb.XGBClassifier(verbosity=0, use_label_encoder=False, learning_rate=learning_rate, max_depth=max_depth, reg_lambda=reg_lambda, reg_alpha=reg_alpha).fit(X, y)
    # Add Glasbox model that includes global and local explanability without Lime and SHAP
    def fit_EBM(self, X, y):
        return ExplainableBoostingClassifier().fit(X,y)

    def fit(self, X, y, model_type, **args):
        model = None
        if model_type == "logit":
            model = self.fit_Logit(X, y, **args)
        elif model_type == "ada":
            model = self.fit_ADA(X, y, **args)
        elif model_type == "knn":
            model = self.fit_knn(X, y, **args)
        elif model_type == "random forest":
            model = self.fit_random_forest(X,y, **args)
        elif model_type == "xgboost":
           model = self.fit_XGB(X,y, **args)
        elif model_type == "ebm":
            model = self.fit_EBM(X,y, **args)
        else:
            raise InputError("Unknown model type.")
        return model

    def predict(self, model, X):
        import numpy as np
        res = 3  # Base number of features
        cols = ["temp", "dwpt", "rhum", "wdir", "wspd"]  # Additional features
        
        # Count additional features
        for e in cols:
            if isinstance(X, pd.DataFrame) and e in X.columns:
                res += 1
            elif isinstance(X, pd.Series) and e in X.index:
                res += 1

        # Ensure X is 2D
        if isinstance(X, pd.Series) or len(X.shape) == 1:
            X = np.array(X).reshape(1, -1)  # Reshape to 2D (1, n_features)

        # Validate the shape before reshaping
        if X.shape[1] != res:
            raise ValueError(f"Expected {res} features, but got {X.shape[1]}")

        # Predict probabilities
        if isinstance(model, sklearn.linear_model.LogisticRegression):
            y_hat = model.predict_proba(X)[:, 1]
        elif isinstance(model, sklearn.neighbors._classification.KNeighborsClassifier):
            y_hat = model.predict_proba(X)[:, 1]
        elif isinstance(model, sklearn.ensemble._forest.RandomForestClassifier):
            y_hat = model.predict_proba(X)[:, 1]
        elif isinstance(model, sklearn.ensemble._weight_boosting.AdaBoostClassifier):
            y_hat = model.predict_proba(X)[:, 1]
        elif isinstance(model, xgboost.sklearn.XGBClassifier):
            y_hat = model.predict_proba(X)[:, 1]
        elif isinstance(model, ExplainableBoostingClassifier):
            y_hat = model.predict_proba(X)[:, 1]
        else:
            raise ValueError("Unknown model type.")

        return y_hat

    def auc(self, y_true, y_hat):
        import sklearn.metrics
        return sklearn.metrics.roc_auc_score(y_true, y_hat)

    def evaluate(
            self, df, model_type, train_start, predict_start="2013-05-16", predict_end=-1, return_errors=False,
            weather_sel=False, xai=False, **args
    ):
        import pandas as pd
        import numpy as np
        from tqdm import tqdm

        dates = pd.DataFrame(df.index)
        dates = dates.set_index(df.index)["timestamp"]
        predict_start = pd.to_datetime(predict_start)
        predict_end = (
    pd.to_datetime(dates[predict_end] if isinstance(dates, np.ndarray) else dates.iloc[predict_end])
    if type(predict_end) == int
    else pd.to_datetime(predict_end)
)


        
        dates = dates.loc[predict_start:predict_end]
        y_true = []
        y_hat_train = {}
        y_hat_test = []
        y_hat_lime = []
        y_hat_shap = []
        auc_train_dict = {}
        auc_test = []
        xai_time_lime = []
        xai_time_shap = []

        predictions_list = []

        if weather_sel:
            print('Crawl weather data....')
            # Add Weather
            ################################
            from meteostat import Point, Daily
            from datetime import datetime, timedelta

            lough = Point(52.766593, -1.223511)
            time = df.index.to_series(name="time").tolist()
            start = time[0]
            end = time[len(time) - 1]
            weather = Daily(lough, start, end)
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
            scaler = MinMaxScaler()
            weather = scaler.fit_transform(weather)
            weather = pd.DataFrame(weather)
            weather["time"] = time[0:len(weather)]
            df["time"] = time

            weather.columns = np.append(headers, "time")

            df = pd.merge(df, weather, how="right", on="time")
            df = df.set_index("time")

            ################################

        if not xai:
            for date in tqdm(dates.index):
                errors = {}
                try:
                    X_train, y_train, X_test, y_test = self.train_test_split(
                        df, date, train_start
                    )
                    # fit model
                    model = self.fit(X_train, y_train, model_type, **args)
                    # predict
                    y_hat_train.update({date: self.predict(model, X_train)})
                    y_hat_test += list(self.predict(model, X_test))
                    # evaluate train data
                    auc_train_dict.update(
                        {date: self.auc(y_train, list(y_hat_train.values())[-1])}
                    )
                    y_true += list(y_test)

                except Exception as e:
                    errors[date] = e
        else:
            print('The explainability approaches in the Usage Agent are being evaluated for model: ' + str(model_type))
            print('Start evaluation with LIME and SHAP')
            import time
            import lime
            import shap as shap
            from lime import lime_tabular

            for date in tqdm(dates):
                errors = {}
                try:
                    # Slice the dataset to reduce the size
                    X_train, y_train, X_test, y_test = self.train_test_split(df, date, **split_params)

                    # Use only a smaller subset of X_test (e.g., first 100 rows)
                    X_test_subset = X_test[:100]  # Only take the first 100 rows of X_test

                    # fit model
                    model = self.fit(X_train, y_train, model_type)

                    # Predict only on the smaller subset
                    y_hat_train.update({date: self.predict(model, X_train)})
                    y_hat_test += list(self.predict(model, X_test_subset))  # Use the subset

                    # Evaluate train data
                    auc_train_dict.update(
                        {date: self.auc(y_train, list(y_hat_train.values())[-1])}
                    )
                    y_true += list(y_test[:100])  # Use the corresponding subset of y_test

                except Exception as e:
                    errors[date] = e

                    start_time = time.time()

                    if model_type == "xgboost":
                        booster = model.get_booster()

                        explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                                           feature_names=X_train.columns,
                                                                           kernel_width=3, verbose=False)

                    else:
                        explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(X_train),
                                                                      mode="classification",
                                                                      feature_names=X_train.columns,
                                                                      categorical_features=[0])

                    if model_type == "xgboost":
                        exp = explainer.explain_instance(X_test, model.predict_proba)
                    else:
                        exp = explainer.explain_instance(data_row=X_test, predict_fn=model.predict_proba)

                    y_hat_lime += list(exp.local_pred)

                    # take time for each day:
                    end_time = time.time()
                    difference_time = end_time - start_time

                    xai_time_lime.append(difference_time)
                    # SHAP
                    # =========================================================================
                    start_time = time.time()

                    if model_type == "logit":
                        X_train_summary = shap.sample(X_train, 100)
                        explainer = shap.KernelExplainer(model.predict_proba, X_train_summary)


                    elif model_type == "ada":
                        X_train_summary = shap.sample(X_train, 100)
                        explainer = shap.KernelExplainer(model.predict_proba, X_train_summary)

                    elif model_type == "knn":
                        X_train_summary = shap.sample(X_train, 100)
                        explainer = shap.KernelExplainer(model.predict_proba, X_train_summary)


                    elif model_type == "random forest":
                        X_train_summary = shap.sample(X_train, 100)
                        explainer = shap.KernelExplainer(model.predict_proba, X_train_summary)

                    elif model_type == "xgboost":
                        explainer = shap.TreeExplainer(model, X_train, model_output='predict_proba')

                    else:
                        raise InputError("Unknown model type.")

                    base_value = explainer.expected_value[1]  # the mean prediction


                    shap_values = explainer.shap_values(
                        X_test)
                    contribution_to_class_1 = np.array(shap_values).sum(axis=1)[1]  # the red part of the diagram
                    shap_prediction = base_value + contribution_to_class_1
                    # Prediction from XAI:
                    y_hat_shap += list([shap_prediction])


                    # take time for each day:
                    end_time = time.time()
                    difference_time = end_time - start_time
                    xai_time_shap.append(difference_time)

                except Exception as e:
                    errors[date] = e

        auc_test = self.auc(y_true, y_hat_test)
        auc_train = np.mean(list(auc_train_dict.values()))
        predictions_list.append(y_true)
        predictions_list.append(y_hat_test)
        predictions_list.append(y_hat_lime)
        predictions_list.append(y_hat_shap)

        # Efficiency
        time_mean_lime = np.mean(xai_time_lime)
        time_mean_shap = np.mean(xai_time_shap)
        print('Mean time nedded by appraoches: ' + str(time_mean_lime) + ' ' + str(time_mean_shap))

        if return_errors:
            return auc_train, auc_test, auc_train_dict, time_mean_lime, time_mean_shap, predictions_list, errors
        else:
            return auc_train, auc_test, auc_train_dict, time_mean_lime, time_mean_shap, predictions_list

    # pipeline function: predicting device usage
    # -------------------------------------------------------------------------------------------
    def pipeline(self, df, date, model_type, train_start, weather_sel=False):

        if weather_sel:
            # Add Weather
            ################################
            from meteostat import Point, Daily
            from datetime import datetime, timedelta

            lough = Point(52.766593, -1.223511)
            time = df.index.to_series(name="time").tolist()
            start = time[0]
            end = time[len(time) - 1]
            weather = Daily(lough, start, end)
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
            scaler = MinMaxScaler()
            weather = scaler.fit_transform(weather)
            weather = pd.DataFrame(weather)
            weather["time"] = time[0:len(weather)]
            df["time"] = time

            weather.columns = np.append(headers, "time")

            df = pd.merge(df, weather, how="right", on="time")
            df = df.set_index("time")
            ################################

        X_train, y_train, X_test, y_test = self.train_test_split(df, date, train_start)
        model = self.fit(X_train, y_train, model_type)
        return self.predict(model, X_test)


    # pipeline function: predicting device usage
    # -------------------------------------------------------------------------------------------
    def pipeline_xai(self, df, date, model_type, train_start, weather_sel=False):

        if weather_sel:
            # Add Weather
            ################################
            from meteostat import Point, Daily
            from datetime import datetime, timedelta

            lough = Point(52.766593, -1.223511)
            time = df.index.to_series(name="time").tolist()
            start = time[0]
            end = time[len(time) - 1]
            weather = Daily(lough, start, end)
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
            scaler = MinMaxScaler()
            weather = scaler.fit_transform(weather)
            weather = pd.DataFrame(weather)
            weather["time"] = time[0:len(weather)]
            df["time"] = time

            weather.columns = np.append(headers, "time")

            df = pd.merge(df, weather, how="right", on="time")
            df = df.set_index("time")
            ################################

        X_train, y_train, X_test, y_test = self.train_test_split(df, date, train_start)
        model = self.fit(X_train, y_train, model_type)
        return self.predict(model, X_test), X_train, X_test, model


# The Original Recommendation Agent
# ===============================================================================================
class Recommendation_Agent:
    def __init__(
        self, activity_input, usage_input, load_input, price_input, shiftable_devices):
        self.activity_input = activity_input
        self.usage_input = usage_input
        self.load_input = load_input
        self.price_input = price_input
        self.shiftable_devices = shiftable_devices
        self.Activity_Agent = Activity_Agent(activity_input)
        # create dicionnary with Usage_Agent for each device
        self.Usage_Agent = {
            name: Usage_Agent(usage_input, name) for name in shiftable_devices
        }
        self.Load_Agent = Load_Agent(load_input)
        self.Price_Agent = Price_Agent(price_input)

    # calculating costs
    # -------------------------------------------------------------------------------------------
    def electricity_prices_from_start_time(self, date):
        import pandas as pd

        prices_48 = self.Price_Agent.return_day_ahead_prices(date)
        prices_from_start_time = pd.DataFrame()
        for i in range(24):
            prices_from_start_time["Price_at_H+" + str(i)] = prices_48.shift(-i)
        # delete last 24 hours
        prices_from_start_time = prices_from_start_time[:-24]
        return prices_from_start_time

    def cost_by_starting_time(self, date, device, evaluation=False):
        import numpy as np
        import pandas as pd

        # get electriciy prices following every device starting hour with previously defined function
        prices = self.electricity_prices_from_start_time(date)
        # build up table with typical load profile repeated for every hour (see Load_Agent)
        if not evaluation:
            device_load = self.Load_Agent.pipeline(
                self.load_input, date, self.shiftable_devices
            ).loc[device]
        else:
            # get device load for one date
            device_load = evaluation["load"][date].loc[device]
        device_load = pd.concat([device_load] * 24, axis=1)
        # multiply both tables and aggregate costs for each starting hour
        costs = np.array(prices) * np.array(device_load)
        costs = np.sum(costs, axis=0)
        # return an array of size 24 containing the total cost at each staring hour.
        return costs

    # creating recommendations
    # -------------------------------------------------------------------------------------------
    def recommend_by_device(
        self,
        date,
        device,
        activity_prob_threshold,
        usage_prob_threshold,
        evaluation=False,
        weather_sel=False
    ):
        import numpy as np

        # add split params as input
        # IN PARTICULAR --> Specify date to start training
        split_params = {
            "train_start": "2013-05-16",
            "test_delta": {"days": 1, "seconds": -1},
            "target": "activity",
        }
        # compute costs by launching time:
        costs = self.cost_by_starting_time(date, device, evaluation=evaluation)
        # compute activity probabilities
        if not evaluation:
            if weather_sel:
                activity_probs = self.Activity_Agent.pipeline(self.activity_input, date, self.model_type, split_params, weather_sel=True)
            else:
                activity_probs = self.Activity_Agent.pipeline(self.activity_input, date, self.model_type, split_params)
        else:
            # get activity probs for date
            activity_probs = evaluation["activity"][date]

        # set values above threshold to 1. Values below to Inf
        # (vector will be multiplied by costs, so that hours of little activity likelihood get cost = Inf)
        activity_probs = np.where(activity_probs >= activity_prob_threshold, 1, float("Inf"))

        # add a flag in case all hours have likelihood smaller than threshold
        no_recommend_flag_activity = 0
        if np.min(activity_probs) == float("Inf"):
            no_recommend_flag_activity = 1

        # compute cheapest hour from likely ones
        best_hour = np.argmin(np.array(costs) * np.array(activity_probs))

        # compute likelihood of usage:
        if not evaluation:
            usage_prob = self.Usage_Agent[device].pipeline(self.usage_input, date, self.model_type, split_params["train_start"])
        else:
            # get usage probs
            name = ("usage_" + device.replace(" ", "_").replace("(", "").replace(")", "").lower())
            usage_prob = evaluation[name][date]


        no_recommend_flag_usage = 0
        if usage_prob < usage_prob_threshold:
            no_recommend_flag_usage = 1

        return {
            "recommendation_date": [date],
            "device": [device],
            "best_launch_hour": [best_hour],
            "no_recommend_flag_activity": [no_recommend_flag_activity],
            "no_recommend_flag_usage": [no_recommend_flag_usage],
            "recommendation": [
                best_hour
                if (no_recommend_flag_activity == 0 and no_recommend_flag_usage == 0)
                else np.nan
            ],
        }

    # visualize recommendation_by device
    def visualize_recommendation_by_device(self, dict):
        recommendation_date = str(dict['recommendation_date'][0])
        recommendation_date = datetime.strptime(recommendation_date, '%Y-%m-%d')
        recommendation_date = recommendation_date.strftime(format = "%d.%m.%Y %H:%M")
        device = dict['device'][0]
        best_launch_hour = dict['best_launch_hour'][0]
        if (dict['no_recommend_flag_activity'][0]== 0 and dict['no_recommend_flag_usage'][0]==0) == True:
            return print('You have one recommendation for the following device: ' + device + '\nPlease use it on ' + recommendation_date[0:10] + ' at '+ recommendation_date[11:]+'.')


    # vizualizing the recommendations
    # -------------------------------------------------------------------------------------------
    def recommendations_on_date_range(
    self, date_range, activity_prob_threshold=0.6, usage_prob_threshold=0.5
):
        import pandas as pd
    
        output = pd.DataFrame()  # Initialize an empty DataFrame
        
        for date in date_range:
            # Get the DataFrame from pipeline
            recommendation = self.pipeline(date, activity_prob_threshold, usage_prob_threshold)
            
            # Concatenate the current recommendation DataFrame to the output DataFrame
            output = pd.concat([output, recommendation], ignore_index=True)
        
        return output


    def visualize_recommendations_on_date_range(self, recs):
        import plotly.express as px
        import plotly.graph_objects as go

        fig = go.Figure()

        for device in recs["device"].unique():
            plot_device = recs[recs["device"] == device]
            fig.add_trace(
                go.Scatter(
                    x=plot_device["recommendation_date"],
                    y=plot_device["recommendation"],
                    mode="lines",
                    name=device,
                )
            )
        fig.show()

    def histogram_recommendation_hour(self, recs):
        import seaborn as sns

        ax = sns.displot(recs, x="recommendation", binwidth=1)
        ax.set(xlabel="Hour of Recommendation", ylabel="counts")

    # pipeline function: create recommendations
    # -------------------------------------------------------------------------------------------
    def pipeline(self, date, activity_prob_threshold, usage_prob_threshold, evaluation=False, weather_sel=False):
        # Initialize an empty dictionary to collect recommendations
        recommendations_dict = {}
    
        # Iterate over all devices and collect their recommendations
        for device in self.shiftable_devices:
            if weather_sel:
                recommendations_by_device = self.recommend_by_device(
                    date,
                    device,
                    activity_prob_threshold,
                    usage_prob_threshold,
                    evaluation=evaluation,
                    weather_sel=True
                )
            else:
                recommendations_by_device = self.recommend_by_device(
                    date,
                    device,
                    activity_prob_threshold,
                    usage_prob_threshold,
                    evaluation=evaluation,
                )
            
            # Store the recommendations for each device in the dictionary
            recommendations_dict[device] = recommendations_by_device
    
        return recommendations_dict


    def visualize_recommendation(self, recommendations_table):

        for i in range(len(recommendations_table)):
            date_and_time = recommendations_table.recommendation_date.iloc[i] + ':' + str(recommendations_table.best_launch_hour.iloc[i])

            date_and_time = datetime.strptime(date_and_time, '%Y-%m-%d:%H')

            date_and_time_show = date_and_time.strftime(format = "%d.%m.%Y %H:%M")
            date_and_time_price = date_and_time.strftime(format = "%Y-%m-%d %H:%M:%S")
            price = price.filter(like=date_and_time_price, axis=0)['Price_at_H+0'].iloc[0]
            output = print('You have a recommendation for the following device: ' + recommendations_table.device.iloc[i]+ '\n\nPlease use the device on the ' + date_and_time_show[0:10] + ' at ' + date_and_time_show[11:] + ' oclock because it costs you only ' + str(price) + ' €.\n')
            if (recommendations_table.no_recommend_flag_activity.iloc[i]==0 and recommendations_table.no_recommend_flag_usage.iloc[i]==0) == True:
                return output
            else:
                return


# Cost-Optimizing Recommendation Agent
# ===============================================================================================
class Cost_Optimizing_Recommendation_Agent:
    def __init__(
        self, activity_input, usage_input, load_input, price_input, shiftable_devices, best_hour = None, model_type = 'random forest'):
        self.activity_input = activity_input
        self.usage_input = usage_input
        self.load_input = load_input
        self.price_input = price_input
        self.shiftable_devices = shiftable_devices
        self.model_type = model_type
        self.Activity_Agent = Activity_Agent(activity_input)
        # create dicionnary with Usage_Agent for each device
        self.Usage_Agent = {
            name: Usage_Agent(usage_input, name) for name in shiftable_devices
        }
        self.Load_Agent = Load_Agent(load_input)
        self.Price_Agent = Price_Agent(price_input)
        self.best_hour = best_hour

    # calculating costs
    # -------------------------------------------------------------------------------------------
    def electricity_prices_from_start_time(self, date):
        import pandas as pd

        prices_48 = self.Price_Agent.return_day_ahead_prices(date)
        prices_from_start_time = pd.DataFrame()
        for i in range(24):
            prices_from_start_time["Price_at_H+" + str(i)] = prices_48.shift(-i)
        # delete last 24 hours
        prices_from_start_time = prices_from_start_time[:-24]
        return prices_from_start_time

    def cost_by_starting_time(self, date, device, evaluation=False):
        import numpy as np
        import pandas as pd

        # get electriciy prices following every device starting hour with previously defined function
        prices = self.electricity_prices_from_start_time(date)
        # build up table with typical load profile repeated for every hour (see Load_Agent)
        if not evaluation:
            device_load = self.Load_Agent.pipeline(
                self.load_input, date, self.shiftable_devices
            ).loc[device]
        else:
            # get device load for one date
            device_load = evaluation["load"][date].loc[device]
        device_load = pd.concat([device_load] * 24, axis=1)
        # multiply both tables and aggregate costs for each starting hour
        costs = np.array(prices) * np.array(device_load)
        costs = np.sum(costs, axis=0)
        # return an array of size 24 containing the total cost at each staring hour.
        return costs

    # creating recommendations
    # -------------------------------------------------------------------------------------------
    def recommend_by_device(
        self,
        date,
        device,
        activity_prob_threshold,
        usage_prob_threshold,
        evaluation=False,
        weather_sel=False
    ):
        import numpy as np

        # add split params as input
        # IN PARTICULAR --> Specify date to start training
        split_params = {
            "train_start": "2013-05-16",
            "test_delta": {"days": 1, "seconds": -1},
            "target": "activity",
        }
        # compute costs by launching time:
        costs = self.cost_by_starting_time(date, device, evaluation=evaluation)

        X_train_activity = None
        X_test_activity = None
        model_activity = None
        model_usage = None

        # compute activity probabilities
        if not evaluation:
            if weather_sel:
                activity_probs, X_train_activity, X_test_activity, model_activity = self.Activity_Agent.pipeline_xai(
                    self.activity_input, date, self.model_type, split_params, weather_sel=True)
            else:
                activity_probs, X_train_activity, X_test_activity, model_activity = self.Activity_Agent.pipeline_xai(
                    self.activity_input, date, self.model_type, split_params, weather_sel=False)
        else:
            # get activity probs for date
            activity_probs = evaluation["activity"][date]

        # set values above threshold to 1. Values below to Inf
        # (vector will be multiplied by costs, so that hours of little activity likelihood get cost = Inf)
        activity_probs = np.where(activity_probs >= activity_prob_threshold, 1, float("Inf"))

        # add a flag in case all hours have likelihood smaller than threshold
        no_recommend_flag_activity = 0
        if np.min(activity_probs) == float("Inf"):
            no_recommend_flag_activity = 1
            print(f"All activity probabilities are below threshold for {device} on {date}")

        # compute cheapest hour from likely ones
        self.best_hour = np.argmin(np.array(costs) * np.array(activity_probs))

        # compute likelihood of usage:
        if not evaluation:
            if weather_sel:
                usage_prob, X_train_usage, X_test_usage, model_usage = self.Usage_Agent[device].pipeline_xai(
                    self.usage_input, date,self.model_type, split_params["train_start"], weather_sel=True)
            else:
                usage_prob, X_train_usage, X_test_usage, model_usage = self.Usage_Agent[device].pipeline_xai(
                self.usage_input, date,self.model_type, split_params["train_start"], weather_sel=False)
        else:
            # get usage probs
            name = ("usage_" + device.replace(" ", "_").replace("(", "").replace(")", "").lower())
            usage_prob = evaluation[name][date]


        no_recommend_flag_usage = 0
        if usage_prob < usage_prob_threshold:
            no_recommend_flag_usage = 1

        self.Explainability_Agent = Explainability_Agent(model_activity, X_train_activity, X_test_activity, self.best_hour, model_usage,
        X_train_usage, X_test_usage, model_type=self.model_type)

        explain = Explainability_Agent(model_activity, X_train_activity, X_test_activity,
                                       self.best_hour,model_usage,X_train_usage, X_test_usage,
                                       model_type= self.model_type)
        feature_importance_activity, feature_importance_usage, explainer_activity, explainer_usage, shap_values, shap_values_usage, X_test_activity, X_test_usage = explain.feature_importance()


        return {
            "recommendation_date": [date],
            "device": [device],
            "best_launch_hour": [self.best_hour],
            "no_recommend_flag_activity": [no_recommend_flag_activity],
            "no_recommend_flag_usage": [no_recommend_flag_usage],
            "recommendation": [
                self.best_hour
                if (no_recommend_flag_activity == 0 and no_recommend_flag_usage == 0)
                else np.nan
            ],
            "feature_importance_activity": [feature_importance_activity],
            "feature_importance_usage": [feature_importance_usage],
            "explainer_activity": [explainer_activity],
            "explainer_usage": [explainer_usage],
            "shap_values": [shap_values],
            "shap_values_usage": [shap_values_usage],
            "X_test_activity": [X_test_activity],
            "X_test_usage": [X_test_usage],
        }

    # visualize recommendation_by device
    def visualize_recommendation_by_device(self, dict):
        recommendation_date = str(dict['recommendation_date'][0])
        recommendation_date = datetime.strptime(recommendation_date, '%Y-%m-%d')
        recommendation_date = recommendation_date.strftime(format = "%d.%m.%Y %H:%M")
        device = dict['device'][0]
        best_launch_hour = dict['best_launch_hour'][0]
        if (dict['no_recommend_flag_activity'][0]== 0 and dict['no_recommend_flag_usage'][0]==0) == True:
            return print('You have one recommendation for the following device: ' + device + '\nPlease use it on ' + recommendation_date[0:10] + ' at '+ recommendation_date[11:]+'.')



    # vizualizing the recommendations
    # -------------------------------------------------------------------------------------------
    def recommendations_on_date_range(
        self, date_range, activity_prob_threshold=0.6, usage_prob_threshold=0.5
    ):
        import pandas as pd

        recommendations = []
        for date in date_range:
            recommendations.append(self.pipeline(date, activity_prob_threshold, usage_prob_threshold))
            output = pd.concat(recommendations)
        return output

    def visualize_recommendations_on_date_range(self, recs):
        import plotly.express as px
        import plotly.graph_objects as go

        fig = go.Figure()

        for device in recs["device"].unique():
            plot_device = recs[recs["device"] == device]
            fig.add_trace(
                go.Scatter(
                    x=plot_device["recommendation_date"],
                    y=plot_device["recommendation"],
                    mode="lines",
                    name=device,
                )
            )
        fig.show()

    def histogram_recommendation_hour(self, recs):
        import seaborn as sns

        ax = sns.displot(recs, x="recommendation", binwidth=1)
        ax.set(xlabel="Hour of Recommendation", ylabel="counts")

    # pipeline function: create recommendations
    # -------------------------------------------------------------------------------------------
    def pipeline(self, date, activity_prob_threshold, usage_prob_threshold, evaluation=False, weather_sel=False):
        import pandas as pd

        recommendations_by_device = self.recommend_by_device(
            date,
            self.shiftable_devices[0],
            activity_prob_threshold,
            usage_prob_threshold,
            evaluation=evaluation,
        )
        recommendations_table = pd.DataFrame.from_dict(recommendations_by_device)

        for device in self.shiftable_devices[1:]:
            if weather_sel:
                recommendations_by_device = self.recommend_by_device(
                    date,
                    device,
                    activity_prob_threshold,
                    usage_prob_threshold,
                    evaluation=evaluation,
                    weather_sel=True
                )
            else:
                recommendations_by_device = self.recommend_by_device(
                    date,
                    device,
                    activity_prob_threshold,
                    usage_prob_threshold,
                    evaluation=evaluation,
                )
        recommendations_table = pd.concat([recommendations_table, pd.DataFrame.from_dict(recommendations_by_device)], ignore_index=True)
            
        return recommendations_table

    def visualize_recommendation(self, recommendations_table, price, diagnostics=False):
        self.diagnostics = diagnostics
        recommendations = False  # Initialize as False before the loop
        
        for r in range(len(recommendations_table)):
            if (recommendations_table.no_recommend_flag_activity.iloc[r] == 0 and
                recommendations_table.no_recommend_flag_usage.iloc[r] == 0):
                
                recommendations = True  # Set to True if any valid recommendation exists
        
                feature_importance_activity = recommendations_table['feature_importance_activity'].iloc[r]
                date = recommendations_table.recommendation_date.iloc[r]
                best_hour = recommendations_table.best_launch_hour.iloc[r]
                explaination_activity = self.Explainability_Agent.explanation_from_feature_importance_activity(
                    feature_importance_activity, date=date, best_hour=best_hour, diagnostics=self.diagnostics
                )
        
                # Combine date and hour into a single string
                date_str = recommendations_table.recommendation_date.iloc[r].strftime('%Y-%m-%d')
                hour_str = str(recommendations_table.best_launch_hour.iloc[r])
                date_and_time_str = f"{date_str}:{hour_str}"
                
                # Convert the combined string into a datetime object
                date_and_time = datetime.strptime(date_and_time_str, '%Y-%m-%d:%H')
                
                # Format the datetime object for display and filtering
                date_and_time_show = date_and_time.strftime("%d.%m.%Y %H:%M")
                date_and_time_price = date_and_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Perform the filtering and calculations
                price_rec = price.filter(like=date_and_time_price, axis=0)['Price_at_H+0'].iloc[0]
                price_mean = price['Price_at_H+0'].sum() / 24
                price_dif = price_rec / price_mean
                price_savings_percentage = round((1 - price_dif) * 100, 2)
            
                print(f"On {date_and_time_show}, the price savings percentage is {price_savings_percentage}%")
        
                output = print(
                    f'You have a recommendation for the following device: {recommendations_table.device.iloc[r]}\n\n'
                    f'Please use the device on {date_and_time_show[0:10]} at {date_and_time_show[11:]} o\'clock because it saves you {price_savings_percentage}% of costs compared to the mean of the day.\n'
                )
        
                feature_importance_usage_device = recommendations_table['feature_importance_usage'].iloc[r]
                explaination_usage = self.Explainability_Agent.explanation_from_feature_importance_usage(
                    feature_importance_usage_device, date=date, diagnostics=self.diagnostics
                )
                print(explaination_usage)
        
                if self.diagnostics:
                    print('Visualizations for further insights into our predictions: ')
                    explainer_usage = recommendations_table['explainer_usage'].iloc[r]
                    shap_values_usage = recommendations_table['shap_values_usage'].iloc[r]
                    X_test_usage = recommendations_table['X_test_usage'].iloc[r]
                    shap_plot_usage = shap.force_plot(explainer_usage.expected_value[1], shap_values_usage[1], X_test_usage)
                    display(shap_plot_usage)
        
        if not recommendations:
            print('There are no recommendations for today.')


class GreenEnergy_Agent:
    def __init__(self, energy_data):
        self.energy_data = energy_data

    def return_day_ahead_green_energy(self, date):
        import pandas as pd

        # Convert string date to datetime for filtering
        date = pd.to_datetime(date)

        # Filter the energy data for the specific date
        daily_data = self.energy_data[self.energy_data['DATETIME'].dt.date == date.date()]

        if daily_data.empty:
            print(f"No energy data available for {date}. Skipping.")
            return None

        # Calculate total green energy production per half-hour or hourly interval
        green_energy_sources = ['WIND', 'SOLAR', 'HYDRO', 'BIOMASS']
        daily_data['total_green_energy'] = daily_data[green_energy_sources].sum(axis=1)

        # Resample to hourly intervals by summing the half-hour intervals
        hourly_data = daily_data.resample('H', on='DATETIME').sum()
        return hourly_data[['total_green_energy']]

    def get_top_green_hours(self, date, top_n=3):
        green_energy_data = self.return_day_ahead_green_energy(date)

        if green_energy_data is None:
            return None

        # Find the hours with the highest green energy production
        top_hours = green_energy_data.nlargest(top_n, 'total_green_energy').index.tolist()
        return top_hours


class GreenEnergy_Recommendation_Agent(Cost_Optimizing_Recommendation_Agent):
    def __init__(
        self, activity_input, usage_input, load_input, energy_data, shiftable_devices, best_hour=None, model_type='random forest'
    ):
        super().__init__(activity_input, usage_input, load_input, None, shiftable_devices, best_hour, model_type)
        self.energy_data = energy_data  # Use energy data instead of price data
        self.GreenEnergy_Agent = GreenEnergy_Agent(energy_data)

    def green_energy_by_starting_time(self, date):
        import pandas as pd

        green_energy_24 = self.GreenEnergy_Agent.return_day_ahead_green_energy(date)
        if green_energy_24 is None:
            return None

        green_energy_from_start_time = pd.DataFrame()
        for i in range(24):
            green_energy_from_start_time[f"GreenEnergy_at_H+{i}"] = green_energy_24['total_green_energy'].shift(-i).iloc[:24].reset_index(drop=True)

        return green_energy_from_start_time

    def cost_by_starting_time(self, date, device, evaluation=False):
        import numpy as np

        # Step 1: Get green energy data for the day
        green_energy = self.green_energy_by_starting_time(date)
        if green_energy is None:
            print(f"No energy data available for {date}. Cannot calculate costs.")
            return np.array([float('inf')] * 24)

        # Step 2: Get device load
        if not evaluation:
            device_load = self.Load_Agent.pipeline(self.load_input, date, self.shiftable_devices).loc[device]
        else:
            device_load = evaluation["load"][date].loc[device]

        # Adjust shapes for matrix multiplication
        green_energy = green_energy.iloc[:, 0].values.reshape(24, 1)  # Reshape to 24x1
        device_load = device_load.values.reshape(24, 1)  # Reshape to 24x1

        # Step 3: Perform element-wise multiplication
        green_energy_usage = np.array(green_energy) * np.array(device_load)

        # Step 4: Aggregate green energy usage to calculate costs
        costs = -np.sum(green_energy_usage, axis=1)

        return costs

    def recommend_by_device(
        self,
        date,
        device,
        activity_prob_threshold,
        usage_prob_threshold,
        evaluation=False,
        weather_sel=False
    ):
        import numpy as np

        # Define parameters for training and testing splits
        split_params = {
            "train_start": "2013-05-16",
            "test_delta": {"days": 1, "seconds": -1},
            "target": "activity",
        }

        # Step 1: Get the cost estimates for the device at different times
        costs = self.cost_by_starting_time(date, device, evaluation=evaluation)
        if np.all(np.isinf(costs)):
            print(f"No valid cost data for {device} on {date}.")
            return None

        # Step 2: Get the top three hours with the highest green energy production
        top_green_hours = self.GreenEnergy_Agent.get_top_green_hours(date, top_n=3)
        if not top_green_hours:
            print(f"No valid green energy data for {date}.")
            return None

        # Ensure `top_green_hours` contains only valid hour indices
        top_green_hours = [hour.hour for hour in top_green_hours if isinstance(hour, pd.Timestamp)]

        # Create a boolean mask for the valid hours
        valid_hours_mask = np.zeros(24, dtype=bool)
        valid_hours_mask[top_green_hours] = True

        # Apply the mask to costs
        costs = np.where(valid_hours_mask, costs, float("inf"))

        # Step 3: Calculate activity probabilities
        if not evaluation:
            if weather_sel:
                activity_probs, X_train_activity, X_test_activity, model_activity = self.Activity_Agent.pipeline_xai(
                    self.activity_input, date, self.model_type, split_params, weather_sel=True
                )
            else:
                activity_probs, X_train_activity, X_test_activity, model_activity = self.Activity_Agent.pipeline_xai(
                    self.activity_input, date, self.model_type, split_params, weather_sel=False
                )
        else:
            activity_probs = evaluation["activity"][date]

        if activity_probs is None or len(activity_probs) == 0:
            print(f"Missing activity probabilities for {device} on {date}.")
            return None

        activity_probs = np.where(activity_probs >= activity_prob_threshold, 1, float("inf"))
        no_recommend_flag_activity = np.min(activity_probs) == float("inf")

        # Combine all filters for valid hours
        final_valid_hours = valid_hours_mask & (activity_probs < float("inf"))
        filtered_costs = np.where(final_valid_hours, costs, float("inf"))

        if np.all(filtered_costs == float("inf")):
            print(f"No valid hours for recommendations on {date}.")
            return None

        # Step 4: Find the best hour
        self.best_hour = np.argmin(filtered_costs)

        # Step 5: Compute usage probabilities
        if not evaluation:
            if weather_sel:
                usage_prob, X_train_usage, X_test_usage, model_usage = self.Usage_Agent[device].pipeline_xai(
                    self.usage_input, date, self.model_type, split_params["train_start"], weather_sel=True
                )
            else:
                usage_prob, X_train_usage, X_test_usage, model_usage = self.Usage_Agent[device].pipeline_xai(
                    self.usage_input, date, self.model_type, split_params["train_start"], weather_sel=False
                )
        else:
            usage_prob = evaluation["usage"][date]

        no_recommend_flag_usage = usage_prob < usage_prob_threshold

        # Step 6: Generate explainability data
        explain = Explainability_Agent(
            model_activity, X_train_activity, X_test_activity, self.best_hour,
            model_usage, X_train_usage, X_test_usage, model_type=self.model_type
        )
        feature_importance_activity, feature_importance_usage, explainer_activity, explainer_usage, shap_values, shap_values_usage, X_test_activity, X_test_usage = explain.feature_importance()

        # Return results
        return {
            "recommendation_date": [date],
            "device": [device],
            "best_launch_hour": [self.best_hour if not (no_recommend_flag_activity or no_recommend_flag_usage) else np.nan],
            "no_recommend_flag_activity": [no_recommend_flag_activity],
            "no_recommend_flag_usage": [no_recommend_flag_usage],
            "recommendation": [
                self.best_hour if not (no_recommend_flag_activity or no_recommend_flag_usage) else np.nan
            ],
            "feature_importance_activity": [feature_importance_activity],
            "feature_importance_usage": [feature_importance_usage],
            "explainer_activity": [explainer_activity],
            "explainer_usage": [explainer_usage],
            "shap_values": [shap_values],
            "shap_values_usage": [shap_values_usage],
            "X_test_activity": [X_test_activity],
            "X_test_usage": [X_test_usage],
        }


    def visualize_recommendation(self, recommendations_table, diagnostics=False):
        self.diagnostics = diagnostics
        recommendations = False  # Initialize as False before the loop
        
        for r in range(len(recommendations_table)):
            if (recommendations_table.no_recommend_flag_activity.iloc[r] == 0 and
                recommendations_table.no_recommend_flag_usage.iloc[r] == 0):
                
                recommendations = True  # Set to True if any valid recommendation exists
            
                date = recommendations_table.recommendation_date.iloc[r]
                best_hour = recommendations_table.best_launch_hour.iloc[r]
            
                # Combine date and hour into a single string
                date_str = pd.to_datetime(recommendations_table.recommendation_date.iloc[r]).strftime('%Y-%m-%d')
                hour_str = str(recommendations_table.best_launch_hour.iloc[r])
                date_and_time_str = f"{date_str}:{hour_str}"
                
                # Convert the combined string into a datetime object
                date_and_time = pd.to_datetime(date_and_time_str, format='%Y-%m-%d:%H')
                
                # Format the datetime object for display
                date_and_time_show = date_and_time.strftime("%d.%m.%Y %H:%M")
            
                output = print(
                    f'You have a recommendation for the following device: {recommendations_table.device.iloc[r]}\n\n'
                    f'Please use the device on {date_and_time_show[0:10]} at {date_and_time_show[11:]} o\'clock to maximize the use of green energy.\n'
                )
        
        if not recommendations:
            print('There are no recommendations for today.')


    def visualize_recommendations_on_date_range(self, recs):
        import plotly.express as px
        import plotly.graph_objects as go

        fig = go.Figure()

        for device in recs["device"].unique():
            plot_device = recs[recs["device"] == device]
            fig.add_trace(
                go.Scatter(
                    x=plot_device["recommendation_date"],
                    y=plot_device["recommendation"],
                    mode="lines",
                    name=device,
                )
            )
        fig.show()

    def histogram_recommendation_hour(self, recs):
        import seaborn as sns

        ax = sns.displot(recs, x="recommendation", binwidth=1)
        ax.set(xlabel="Hour of Recommendation", ylabel="counts")


# Original Evaluation Agent
# ===============================================================================================
class Evaluation_Agent:
    def __init__(self, DATA_PATH, model_type, config, load_data=True, load_files=None, weather_sel=False, xai = False):
        import src.agents as agents
        from helper_functions import Helper
        import pandas as pd

        helper = Helper()

        self.model_type = model_type
        self.config = config
        self.weather_sel = weather_sel
        self.xai = xai
        house_df = helper.load_household(REFIT_dir=DATA_PATH, weather_sel=weather_sel)
        if load_data:
            self.preparation = agents.Preparation_Agent(house_df)
        else:
            self.preparation = agents.Preparation_Agent(None)

        self.price = (
            agents.Price_Agent(helper.create_day_ahead_prices_df(DATA_PATH, "\Day-ahead Prices_201501010000-201601010000.csv"))
            if load_data
            else None
        )
        self.activity = None
        self.load = None
        for device in self.config["user_input"]["shiftable_devices"]:
            name = ("usage_" + device.replace(" ", "_").replace("(", "").replace(")", "").lower())
            exec(f"self.{name} = None")
        self.recommendation = None
        self.df = {}
        self.output = {}
        self.errors = {}
        self.agent_scores = {}
        self.agent_predictions_list_activity = {}
        self.agent_predictions_list_usage = {}
        self.cold_start_scores = {}
        self.results = {}
        self.cold_start_days = pd.DataFrame()
        if load_files != None:
            self.load_from_drive(load_files)

    # helper: loading and storing intermediary results and further helper
    # -------------------------------------------------------------------------------------------
    def _load_object(self, filename):
        import pickle
        import json
        import yaml

        # using a command dict as a if-list
        commands = {
            "pkl": f"pickle.load(open('{filename}', 'rb'))",
            "json": f"json.load(open('{filename}', 'r'))",
            "yaml": f"yaml.load(open('{filename}', 'r'), Loader = yaml.Loader)",
        }

        *_, name, ftype = filename.split(".")
        name = name[name.rfind("_") + 1:]
        obj = eval(commands[ftype])
        self[name] = obj

    def load_from_drive(self, files):
        files = [files] if type(files) != list else files
        for filename in files:
            self._load_object(filename)

    def dump(self, EXPORT_PATH):
        import json
        import yaml
        import pickle

        # storing the current configuration
        json.dump(self.config, open(EXPORT_PATH + str(self.config["data"]["household"]) + '_' + str(self.config["activity"]["model_type"]) +'_'
                                          + str(self.config["usage"]["model_type"]) + '_' + str(self.weather_sel) + "_config.json", "w"), indent=4)

        # storing the prepared data
        if self.df != {}:
            pickle.dump(self.df, open(EXPORT_PATH + str(self.config["data"]["household"]) + '_' + str(self.config["activity"]["model_type"]) +'_'
                                         + str(self.config["usage"]["model_type"]) + '_' + str(self.weather_sel) + "_df.pkl", "wb"))

        # storing the agents' output
        if self.output != {}:
            pickle.dump(self.output, open(EXPORT_PATH + str(self.config["data"]["household"]) + '_' + str(self.config["activity"]["model_type"]) +'_'
                                         + str(self.config["usage"]["model_type"]) + '_' + str(self.weather_sel) + "_output.pkl", "wb"))

        # storing the results
        if self.results != {}:
            pickle.dump(self.results, open(EXPORT_PATH + str(self.config["data"]["household"]) + '_' + str(self.config["activity"]["model_type"]) +'_'
                                        + str(self.config["usage"]["model_type"]) + '_' + str(self.weather_sel) + "_results.pkl", "wb"))

        # storing the results
        if self.agent_scores != {}:
            pickle.dump(self.agent_scores, open(EXPORT_PATH + str(self.config["data"]["household"]) + '_' + str(self.config["activity"]["model_type"]) +'_'
                                        + str(self.config["usage"]["model_type"]) + '_' + str(self.weather_sel) + "_scores.pkl", "wb"))

        if self.agent_predictions_list_activity != {}:
            pickle.dump(self.agent_predictions_list_activity, open(EXPORT_PATH + str(self.config["data"]["household"]) + '_' + str(self.config["activity"]["model_type"]) +'_'
                                         + str(self.config["usage"]["model_type"]) + '_' + str(self.weather_sel) + "_predictions.pkl", "wb"))

        if self.agent_predictions_list_usage != {}:
            pickle.dump(self.agent_predictions_list_usage, open(EXPORT_PATH + str(self.config["data"]["household"]) + '_' + str(self.config["activity"]["model_type"]) +'_'
                                        + str(self.config["usage"]["model_type"]) + '_' + str(self.weather_sel) + "_predictions_usage.pkl", "wb"))

    def __getitem__(self, item):
        return eval(f"self.{item}")

    def __setitem__(self, key, value):
        exec(f"self.{key} = value")

    def _format_time(self, seconds):
        return "{:02.0f}".format(seconds // 60) + ":" + "{:02.0f}".format(seconds % 60)

    def _get_agent_names(self):
        devices = self.config["user_input"]["shiftable_devices"]
        names = ["activity", "load"] + ["usage_"+ str(device).replace(" ", "_").replace("(", "").replace(")", "").lower() for device in devices]
        return names

    # creating the default configuration
    # -------------------------------------------------------------------------------------------
    def get_default_config(self, agents):
        if type(agents) != list:
            agents = [agents]

        agents = [agent.lower() for agent in agents]
        for agent in agents:
            exec(f"self._get_default_{agent}_config()")

    def _get_default_preparation_config(self):
        from copy import deepcopy

        # preparation
        self.config["preparation"] = {}
        ## preparation: activity agent
        self.config["preparation"]["activity"] = {
            "truncate": {"features": "all", "factor": 1.5, "verbose": 0},
            "scale": {"features": "all", "kind": "MinMax", "verbose": 0},
            "aggregate": {"resample_param": "60T"},
            "activity": {
                "active_appliances": deepcopy(self.config["user_input"]["active_appliances"]),
                "threshold": deepcopy(self.config["user_input"]["threshold"]),
            },
            "time": {"features": ["hour", "day_name"]},
            "activity_lag": {"features": ["activity"], "lags": [24, 48, 72]},
        }
        # preparation: usage agent
        self.config["preparation"]["usage"] = {
            "truncate": {"features": "all", "factor": 1.5, "verbose": 0},
            "scale": {"features": "all", "kind": "MinMax", "verbose": 0},
            "activity": {
                "active_appliances": deepcopy(self.config["user_input"]["active_appliances"]),
                "threshold": deepcopy(self.config["user_input"]["threshold"]),
            },
            "aggregate_hour": {"resample_param": "60T"},
            "aggregate_day": {"resample_param": "24H"},
            "time": {"features": ["hour", "day_name"]},
            "shiftable_devices": deepcopy(self.config["user_input"]["shiftable_devices"]),
            "device": {"threshold": deepcopy(self.config["user_input"]["threshold"])},
        }
        # preparation: load agent
        self.config["preparation"]["load"] = {
            "truncate": {"features": "all", "factor": 1.5, "verbose": 0},
            "scale": {"features": "all", "kind": "MinMax", "verbose": 0},
            "aggregate": {"resample_param": "60T"},
            "shiftable_devices": deepcopy(self.config["user_input"]["shiftable_devices"]),
            "device": {"threshold": deepcopy(self.config["user_input"]["threshold"])},
        }

    def _get_default_activity_config(self):
        from copy import deepcopy

        if (self.activity == None):
            self.init_agents()
        self._get_dates()
        self.config["activity"] = {
            "model_type": self.model_type,
            "split_params": {
                "train_start": deepcopy(self.config["data"]["start_dates"]["activity"]),
                "test_delta": {"days": 1, "seconds": -1},
                "target": "activity",
            },
        }

    def _get_default_load_config(self):
        from copy import deepcopy

        if (self.load == None):
            self.init_agents()
        self._get_dates()
        self.config["load"] = {
            "shiftable_devices": deepcopy(self.config["user_input"]["shiftable_devices"])
        }

    def _get_default_usage_config(self):
        from copy import deepcopy

        if (self.activity == None) | (self.load == None):
            self.init_agents()
        self._get_dates()
        self.config["usage"] = {
            "model_type":  self.model_type,
            "train_start": deepcopy(self.config["data"]["start_dates"]["usage"]),
        }
        for device in self.config["user_input"]["shiftable_devices"]:
            name = ("usage_" + device.replace(" ", "_").replace("(", "").replace(")", "").lower())
            self.config[name] = self.config["usage"]
            self.config["data"]["start_dates"][name] = self.config["data"]["start_dates"]["usage"]

    # extracting the available dates in the data
    def get_first_date(self, df):
        import pandas as pd

        first_data = df.index.to_series()[0]
        return (first_data + pd.Timedelta("1D")).replace(hour=0, minute=0, second=0)

    def get_last_date(self, df):
        import pandas as pd

        last_data = df.index.to_series()[-1]
        return (last_data - pd.Timedelta("1D")).replace(hour=23, minute=59, second=59)

    def get_min_start_date(self, df):
        df = df.dropna()
        return df.loc[df.index.hour == 0, :].index[0]

    def _get_dates(self):
        import numpy as np

        # first and last date in the data
        self.config["data"]["first_date"] = str(self.get_first_date(self.preparation.input))[:10]
        self.config["data"]["last_date"] = str(self.get_last_date(self.preparation.input))[:10]
        # start dates
        start_dates = {}
        for agent, data in self.df.items():
            start_dates[agent] = self.get_min_start_date(data)
        start_dates["combined"] = np.max(list(start_dates.values()))
        self.config["data"]["start_dates"] = {
            key: str(value)[:10] for key, value in start_dates.items()
        }


    # running the pipeline
    # -------------------------------------------------------------------------------------------
    def pipeline(self, agents, **kwargs):
        # converting single agent to list
        if type(agents) != list:
            agents = [agents]

        agents = [agent.lower() for agent in agents]

        if 'preparation' in agents:
            self._prepare(**kwargs)
        if 'activity' in agents:
            self._pipeline_activity_usage_load('activity', **kwargs)
        if 'usage' in agents:
            usage_agents = ["usage_"+ device.replace(" ", "_").replace("(", "").replace(")", "").lower() for device in self.config["user_input"]["shiftable_devices"]]
            for agent in usage_agents:
                self._pipeline_activity_usage_load(agent, **kwargs)
        if 'load' in agents:
            self._pipeline_activity_usage_load('load', **kwargs)
        if 'recommendation' in agents:
            self._get_recommendations(**kwargs)

    def init_agents(self):
        import src.agents as agents

        # initialize the agents
        self.activity = agents.Activity_Agent(self.df["activity"])
        self.load = agents.Load_Agent(self.df["load"])

        # initialize usage agents for the shiftable devices: agent = usage_name
        for device in self.config["user_input"]["shiftable_devices"]:
            name = ("usage_" + device.replace(" ", "_").replace("(", "").replace(")", "").lower())
            exec(f'self.{name} = Usage_Agent(self.df["usage"], "{device}")')
            self.df[name] = self.df["usage"]

        self.recommendation = agents.Recommendation_Agent(
            self.df["activity"],
            self.df["usage"],
            self.df["load"],
            self.price.input,
            self.config["user_input"]["shiftable_devices"]
        )

    def _prepare(self, agent="all"):
        lines = {
            "activity": 'self.df["activity"] = self.preparation.pipeline_activity(self.preparation.input, self.config["preparation"]["activity"])',
            "usage": 'self.df["usage"] = self.preparation.pipeline_usage(self.preparation.input, self.config["preparation"]["usage"])',
            "load": 'self.df["load"] ,_,_ = self.preparation.pipeline_load(self.preparation.input, self.config["preparation"]["load"])',
        }
        if agent == "all":
            for agent in ["activity", "usage", "load"]:
                exec(lines[agent])
                print(f"[evaluation agent] Finished preparing the data for the {agent} agent.")
        else:
            exec(lines[agent])
            print(f"[evaluation agent] Finished preparing the data for the {agent} agent.")

    def _pipeline_activity_usage_load(self, agent, verbose=1):
        import pandas as pd
        from IPython.display import clear_output
        import time

        self.output[agent] = {}
        self.errors[agent] = {}

        # init agents
        if (self.activity == None) | (self.load == None):
            self.init_agents()

        # determining the dates
        dates = self.df[agent].index.to_series()
        start = pd.to_datetime(self.config["data"]["start_dates"][agent])
        end = pd.to_datetime(self.config["data"]["last_date"]).replace(
            hour=23, minute=59, second=59
        )
        dates = dates[(dates >= start) & (dates <= end)].resample("1D").count()
        dates = [str(date)[:10] for date in list(dates.index)]

        # pipeline funtion
        start = time.time() if verbose >= 1 else None
        for date in dates:
            try:
                self.output[agent][date] = eval(f'self.{agent}.pipeline(self.{agent}.input, "{date}", **self.config["{agent}"])')
                # verbose
                if verbose >= 1:
                    clear_output(wait=True)
                    elapsed = time.time() - start
                    remaining = (elapsed / (len(dates)) * (len(dates) - (dates.index(date) + 1)))
                    print(f"agent:\t\t{agent}")
                    print(f"progress: \t{dates.index(date)+1}/{len(dates)}")
                    print(f"time:\t\t[{self._format_time(elapsed)}<{self._format_time(remaining)}]\n")
                    print(self.output[agent][date])
            except Exception as e:
                self.errors[agent][date] = type(e).__name__

    def _get_recommendations(
        self, activity_threshold, usage_threshold, dates: tuple = "all"
    ):
        import numpy as np
        from IPython.display import clear_output

        # determining dates
        start = (
            self.config["data"]["start_dates"]["combined"]
            if dates == "all"
            else dates[0]
        )
        end = self.config["data"]["last_date"] if dates == "all" else dates[1]
        dates = np.arange(
            np.datetime64(start),
            np.datetime64(end) + np.timedelta64(1, "D"),
            np.timedelta64(1, "D"),
        )
        dates = [str(date) for date in dates]

        # creating recommendations
        self.errors["recommendation"] = {}
        self.output["recommendation"] = {}
        for date in dates:
            try:
                self.output["recommendation"][date] = self.recommendation.pipeline(
                    date, activity_threshold, usage_threshold, evaluation=self.output
                )
            except Exception as e:
                self.errors["recommendation"][date] = e

        # merging the recommendations into one dataframe
        df = list(self.output["recommendation"].values())[0]

        for idx in range(1, len(self.output["recommendation"].values())):
            df = pd.concat([df, pd.DataFrame([self.output["recommendation"].values()[idx]])], ignore_index=True)

        df.set_index("recommendation_date", inplace=True)
        self.output["recommendation"] = df
        clear_output()

    # individual agent scores
    # -------------------------------------------------------------------------------------------
    def get_agent_scores(self, xai=False, **args):
        self.xai =xai
        scores = {}
        scores['activity_auc'] = None
        scores['time_mean_lime_activity'] = {}
        scores['time_mean_shap_activity'] = {}
        scores['usage_auc'] = {}
        scores['time_mean_lime_usage'] = {}
        scores['time_mean_shap_usage'] = {}
        scores['load_mse'] = {}

        agents = self._get_agent_names()
        for agent in agents:
            agent_type = agent.split('_')[0]

            if agent_type == 'activity':
                _, auc_test_activity, _, time_mean_lime_activity, time_mean_shap_activity, predictions_list_activity = self[agent].evaluate(self[agent].input, **self.config[agent], xai=self.xai, **args)
                scores['activity_auc'] = auc_test_activity
                scores['time_mean_lime_activity'] = time_mean_lime_activity
                scores['time_mean_shap_activity'] = time_mean_shap_activity

            if agent_type == 'usage':
                _, auc_test_usage, _, time_mean_lime_usage, time_mean_shap_usage, predictions_list_usage = self[agent].evaluate(self[agent].input, **self.config[agent], xai=self.xai, **args)
                scores['usage_auc'][self[agent].device] = auc_test_usage
                scores['time_mean_lime_usage'] = time_mean_lime_usage
                scores['time_mean_shap_usage'] = time_mean_shap_usage

            if agent_type == 'load':
                try:
                    scores['load_mse'] = self.load.evaluate(**self.config['load'], evaluation=self.output['load'])
                except KeyError:
                    scores['load_mse'] = self.load.evaluate(**self.config['load'])
        self.agent_scores = scores
        self.agent_predictions_list_activity = predictions_list_activity
        self.agent_predictions_list_usage = predictions_list_usage
        return scores, predictions_list_activity, predictions_list_usage

    def agent_scores_to_summary(self, scores='default'):
        import pandas as pd

        if scores == 'default':
            scores = self.agent_scores

        summary = {}
        summary['activity_auc'] = pd.DataFrame()
        summary['usage_auc'] = pd.DataFrame()
        summary['load_mse'] = pd.DataFrame()

        household_id = self.config['data']['household']
        devices = self.config['user_input']['shiftable_devices']

        # activity
        summary['activity_auc'].loc[household_id, '-'] = scores['activity_auc']
        # usage
        i = 0
        for device in devices:
            summary['usage_auc'].loc[household_id, i] = scores['usage_auc'][device]
            i += 1
        # load
        i = 0
        for device in devices:
            summary['load_mse'].loc[household_id, i] = scores['load_mse'][device]
            i += 1

        summary['activity_auc'].index.name = 'household'
        summary['usage_auc'].index.name = 'household'
        summary['load_mse'].index.name = 'household'
        summary['usage_auc'].columns.name = 'device'
        summary['load_mse'].columns.name = 'device'
        return summary

    def predictions_to_xai_metrics(self, predictions, activity_threshold= 0.5, usage_threshold= 0.5):
        import numpy as np
        import sklearn.metrics

        y_true = np.array(predictions[0])
        y_hat_test = np.array(predictions[1])
        y_hat_lime = np.array(predictions[2])
        y_hat_shap = np.array(predictions[3])

        self.activity_threshold = activity_threshold
        self.usage_threshold = usage_threshold

        self.y_true = y_true
        self.y_hat_test = y_hat_test
        self.y_hat_lime = y_hat_lime
        self.y_hat_shap = y_hat_shap

        #turn y_hat test into binary
        self.y_hat_test_bin = np.where(y_hat_test > activity_threshold, 1, 0)
        self.y_hat_lime_bin = np.where(y_hat_lime > activity_threshold, 1, 0)
        self.y_hat_shap_bin = np.where(y_hat_shap > activity_threshold, 1, 0)


        xai_scores = {}
        xai_scores['activity_lime_auc_true'] = None
        xai_scores['activity_shap_auc_true'] = {}
        xai_scores['activity_lime_auc_pred'] = {}
        xai_scores['activity_shap_auc_pred'] = {}
        xai_scores['activity_lime_MAE'] = {}
        xai_scores['activity_shap_MAE'] = {}

        # AUC of true - xai prediction
        xai_scores['activity_lime_auc_true'] = sklearn.metrics.roc_auc_score(y_true[:len(y_hat_lime)], y_hat_lime)
        xai_scores['activity_shap_auc_true'] = sklearn.metrics.roc_auc_score(y_true[:len(y_hat_shap)], y_hat_shap)

        # AUC of predicted probabilities - xai prediction
        xai_scores['activity_lime_auc_pred'] = sklearn.metrics.roc_auc_score(self.y_hat_test_bin[:len(y_hat_lime)], self.y_hat_lime_bin)
        xai_scores['activity_shap_auc_pred'] = sklearn.metrics.roc_auc_score(self.y_hat_test_bin[:len(y_hat_shap)], self.y_hat_shap_bin)

        # MAE
        MAE_SHAP = []
        zip_object = zip(self.y_hat_test[:len(y_hat_shap)],self.y_hat_shap)
        for list1_i, list2_i in zip_object:
            MAE_SHAP.append(abs(list1_i - list2_i))

        MAE_LIME = []
        zip_object = zip(self.y_hat_test[:len(y_hat_lime)], self.y_hat_lime)
        for list1_i, list2_i in zip_object:
            MAE_LIME.append(abs(list1_i - list2_i))

        xai_scores['activity_lime_MAE'] = np.mean(MAE_LIME)
        xai_scores['activity_shap_MAE'] = np.mean(MAE_SHAP)

        self.xai_scores = xai_scores
        return xai_scores

    def predictions_to_xai_metrics_usage(self, predictions, activity_threshold= 0.5, usage_threshold= 0.5):
        import numpy as np
        import sklearn.metrics

        y_true = np.array(predictions[0])
        y_hat_test = np.array(predictions[1])
        y_hat_lime = np.array(predictions[2])
        y_hat_shap = np.array(predictions[3])

        self.activity_threshold = activity_threshold
        self.usage_threshold = usage_threshold

        self.y_true = y_true
        self.y_hat_test = y_hat_test
        self.y_hat_lime = y_hat_lime
        self.y_hat_shap = y_hat_shap

        #turn y_hat test into binary
        self.y_hat_test_bin = np.where(y_hat_test > usage_threshold, 1, 0)
        self.y_hat_lime_bin = np.where(y_hat_lime > usage_threshold, 1, 0)
        self.y_hat_shap_bin = np.where(y_hat_shap > usage_threshold, 1, 0)


        xai_scores = {}
        xai_scores['usage_lime_auc_true'] = None
        xai_scores['usage_shap_auc_true'] = {}
        xai_scores['usage_lime_auc_pred'] = {}
        xai_scores['usage_shap_auc_pred'] = {}
        xai_scores['usage_lime_MAE'] = {}
        xai_scores['usage_shap_MAE'] = {}

        # AUC of true - xai prediction
        xai_scores['usage_lime_auc_true'] = sklearn.metrics.roc_auc_score(y_true[:len(y_hat_lime)], y_hat_lime)
        xai_scores['usage_shap_auc_true'] = sklearn.metrics.roc_auc_score(y_true[:len(y_hat_shap)], y_hat_shap)

        # AUC of predicted probabilities - xai prediction
        xai_scores['usage_lime_auc_pred'] = sklearn.metrics.roc_auc_score(self.y_hat_test_bin[:len(y_hat_lime)], self.y_hat_lime_bin)
        xai_scores['usage_shap_auc_pred'] = sklearn.metrics.roc_auc_score(self.y_hat_test_bin[:len(y_hat_shap)], self.y_hat_shap_bin)

        # MAE

        MAE_SHAP = []
        zip_object = zip(self.y_hat_test[:len(y_hat_shap)], self.y_hat_shap)
        for list1_i, list2_i in zip_object:
            MAE_SHAP.append(abs(list1_i - list2_i))

        MAE_LIME = []
        zip_object = zip(self.y_hat_test[:len(y_hat_lime)], self.y_hat_lime)
        for list1_i, list2_i in zip_object:
            MAE_LIME.append(abs(list1_i - list2_i))

        xai_scores['usage_lime_MAE'] = np.mean(MAE_LIME)
        xai_scores['usage_shap_MAE'] = np.mean(MAE_SHAP)


        self.xai_scores = xai_scores
        return xai_scores

    # cold start: predict on all data
    # -------------------------------------------------------------------------------------------
    def predict_all(self, agent, **kwargs):
        agent_type = agent.split("_")[0]
        return eval(f"self._predict_all_{agent_type}(agent, **kwargs)")

    def _predict_all_load(self, agent, device):
        y_hat = {
            date: profiles.loc[device, :]
            for date, profiles in self.output[agent].items()
        }
        return y_hat

    def _predict_all_activity(self, agent):
        return self._predict_all_activity_usage(agent)

    def _predict_all_usage(self, agent):
        return self._predict_all_activity_usage(agent)

    def _predict_all_activity_usage(self, agent):
        import pandas as pd
        import numpy as np

        y_hat = {}
        # intitializing the error dict
        try:
            self.errors["evaluation"]
        except KeyError:
            self.errors["evaluation"] = {}

        try:
            self.errors["evaluation"][agent]
        except KeyError:
            self.errors["evaluation"][agent] = {}

        # determining the dates
        dates = np.arange(
            np.datetime64(self.config["data"]["start_dates"][agent]),
            np.datetime64(self.config["data"]["last_date"]) + np.timedelta64(1, "D"),
            np.timedelta64(1, "D"),
        )
        start = dates[0]
        end = dates[-1] + pd.Timedelta(days=1, seconds=-1)

        # creating X_test
        X_test, _, _, _ = self[agent].train_test_split(
            self[agent].input,
            dates[-1] + np.timedelta64(1, "D"),
            train_start=self.config["data"]["start_dates"][agent],
        )

        # creating predictions
        for date in dates:
            X_train, y_train, _, _ = self[agent].train_test_split(
                self[agent].input,
                date,
                train_start=self.config["data"]["start_dates"][agent],
            )
            try:
                model = self[agent].fit(X_train, y_train, self.model_type)
                y_hat[date] = self[agent].predict(model, X_test)
            except Exception as e:
                self.errors["evaluation"][agent][date] = type(e).__name__
        return y_hat

    # cold start: calculate cold start scores
    # -------------------------------------------------------------------------------------------
    def get_cold_start_scores(self, fn: dict = "default"):
        from IPython.display import clear_output

        scores = {}
        fn = {} if fn == "default" else fn

        # activity-agent
        scores["activity"] = self._get_cold_start_score("activity", fn=fn.get("activity", "default"))
        clear_output()

        for device in self.config["user_input"]["shiftable_devices"]:
            name = device.replace(" ", "_").replace("(", "").replace(")", "").lower()
            # usage agent
            scores["usage_" + name] = self._get_cold_start_score("usage_" + name, fn=fn.get("usage", "default"))
            # load agent
            scores["load_" + name] = self._get_cold_start_score("load", fn=fn.get("load", "default"), device=device)
            clear_output()
        self.cold_start_scores = scores

    def _get_cold_start_score(self, agent, fn="default", **kwargs):
        import sklearn.metrics
        import numpy as np

        agent_type = agent.split("_")[0]
        # specifying the correct score function
        fn_dict = {
            "activity": f"self.{agent}.auc",
            "usage": f"self.{agent}.auc",
            "load": "sklearn.metrics.mean_squared_error",
        }
        fn = eval(fn_dict[agent_type]) if fn == "default" else fn

        # specifying the correct y_true, y_hat
        y_dict = {
            "activity": "self[agent].train_test_split(self[agent].input, date=np.datetime64(self.config['data']['last_date'])+np.timedelta64(1, 'D'), train_start=self.config['data']['start_dates'][agent])",
            "usage": "self[agent].train_test_split(self[agent].input, date=np.datetime64(self.config['data']['last_date'])+np.timedelta64(1, 'D'), train_start=self.config['data']['start_dates'][agent])",
            "load": "list(self.output['load'].values())[-1].loc[kwargs['device'], :]",
        }
        y_true = eval(y_dict[agent_type])
        y_true = y_true if agent_type == "load" else y_true[1]
        y_hat = self.predict_all(agent, **kwargs)

        # calculating the scores
        scores = {}
        for date, pred in y_hat.items():
            scores[date] = fn(y_true, pred)
        return scores

    def cold_start_scores_to_df(self):
        import pandas as pd
        import numpy as np

        scores_df = pd.DataFrame()
        # convert dicts into dataframe
        for key in self.cold_start_scores.keys():
            for date, score in self.cold_start_scores[key].items():
                scores_df.loc[str(date), key] = score

        # sort the dataframe
        cols = (
            ["activity"]
            + [col for col in scores_df if col.startswith("usage")]
            + [col for col in scores_df if col.startswith("load")]
        )
        scores_df.index = scores_df.index.map(np.datetime64)
        scores_df = scores_df[cols].sort_index()
        return scores_df

    def get_cold_start_days(self, tolerance_values):
        import pandas as pd

        self.cold_start_days = pd.DataFrame({"tolerance": []}).set_index("tolerance")
        scores_df = self.cold_start_scores_to_df()
        tolerance_fn = {
            "activity": "scores_df[agent].max() * (1 - tolerance[agent_type])",
            "usage": "scores_df[agent].max() * (1 - tolerance[agent_type])",
            "load": "tolerance['load']",
        }

        # agent coldstart days
        for tolerance in tolerance_values:
            tolerance = {"activity": tolerance, "usage": tolerance, "load": tolerance}

            for agent in scores_df.columns:
                agent_type = agent.split("_")[0]

                done = False
                day = 0
                while not done:
                    day += 1
                    tolerance_value = eval(tolerance_fn[agent_type])
                    if agent_type == "load":
                        done = all(scores_df[agent].values[day - 1:] < tolerance_value)
                    else:
                        done = all(scores_df[agent].values[day - 1:] > tolerance_value)
                self.cold_start_days.loc[tolerance[agent_type], agent] = day
        # framework cold start days
        self.cold_start_days['framework'] = self.cold_start_days.max(axis=1)

    def cold_start_to_summary(self, tolerance_values='all'):
        import pandas as pd

        if tolerance_values == 'all':
            tolerance_values = list(self.cold_start_days.index)

        household_id = self.config['data']['household']
        devices = self.config['user_input']['shiftable_devices']

        summary = {}
        summary['activity'] = {}
        summary['usage'] = {}
        summary['load'] = {}
        summary['framework'] = {}

        # activity agent
        summary['activity']['-'] = {}  # '-': placeholder for device
        summary['activity']['-'][household_id] = self.cold_start_days['activity'][tolerance_values].astype(int).to_list()
        # usage agent
        i = 0
        for device in devices:
            name = 'usage_' + device.replace(" ", "_").replace("(", "").replace(")", "").lower()
            summary['usage'][i] = {}
            summary['usage'][i][household_id] = self.cold_start_days[name][tolerance_values].astype(int).to_list()
            i += 1

        # load agent
        i = 0
        for device in devices:
            name = 'load_' + device.replace(" ", "_").replace("(", "").replace(")", "").lower()
            summary['load'][i] = {}
            summary['load'][i][household_id] = self.cold_start_days[name][tolerance_values].astype(int).to_list()
            i += 1

        # framework
        summary['framework']['-'] = {}  # '-': placeholder for device
        summary['framework']['-'][household_id] = self.cold_start_days['framework'][tolerance_values].astype(int).to_list()

        # converting the format
        for key, value in summary.items():
            summary[key] = pd.DataFrame(value)
            summary[key].columns.name = 'device'
            summary[key].index.name = 'household'
        return summary

    # cold start: visualizations
    # -------------------------------------------------------------------------------------------
    def _plot_axs(self, axs, y, x=None, legend=None, **kwargs):
        axs.plot(x, y) if x != None else axs.plot(y)
        axs.set(**kwargs)
        axs.legend(legend) if legend != None else None

    def visualize_cold_start(self, metrics_name: dict, tolerance: dict = None, figsize=(18, 5)):
        import matplotlib.pyplot as plt

        scores_df = self.cold_start_scores_to_df()
        fig, axs = plt.subplots(1, 3, figsize=figsize)

        # activity
        self._plot_axs(
            axs[0],
            x=range(1, scores_df.shape[0] + 1),
            y=scores_df["activity"],
            title=f"[activity] {metrics_name['activity']}",
        )
        legend = ['activity']
        if tolerance != None:
            tolerance_value = scores_df["activity"].max() * (1 - tolerance["activity"])
            color = axs[0].lines[-1].get_color()
            axs[0].plot([tolerance_value] * scores_df.shape[0], "--", c=color)
            legend.append([f"tolerance@{tolerance['activity']}"])
        axs[0].legend(legend)
        axs[0].set_xlabel("days")

        # usage
        usage_agents = [agent for agent in scores_df.columns if agent.find("usage") != -1]
        legend = []
        for agent in usage_agents:
            self._plot_axs(axs[1],
                x=range(1, scores_df.shape[0] + 1),
                y=scores_df[agent],
                title=f"[usage] {metrics_name['usage']}",
            )
            legend += [agent]
            if tolerance != None:
                tolerance_value = scores_df[agent].max() * (1 - tolerance["usage"])
                color = axs[1].lines[-1].get_color()
                axs[1].plot([tolerance_value] * scores_df.shape[0], "--", c=color)
                legend += [f"tolerance_{agent.replace('usage_', '')}@{tolerance['usage']}"]
        axs[1].legend(legend)
        axs[1].set_xlabel("days")

        # load
        load_agents = [agent for agent in scores_df.columns if agent.find("load") != -1]
        legend = []
        for agent in load_agents:
            self._plot_axs(
                axs[2],
                x=range(1, scores_df.shape[0] + 1),
                y=scores_df[agent],
                title=f"[load] {metrics_name['load']}",
            )
            legend += [agent]
        if tolerance != None:
            axs[2].plot([tolerance["load"]] * scores_df.shape[0], "--", c="black")
            legend += [f"tolerance@{tolerance['load']}"]
        axs[2].legend(legend)
        axs[2].set_xlabel("days")

    # evaluation: calculate costs per device run
    # -------------------------------------------------------------------------------------------
    def calculate_cost(self, date, hour, load):
        import numpy as np

        if np.isnan(hour):
            return np.nan
        else:
            price_idx = self.price.input.index.values
            prices = self.price.input.values

            dt = np.datetime64(date) + np.timedelta64(int(hour), "h")
            # getting the correct position for the load in the load array
            i = np.where(price_idx == dt)[0][0]
            i = np.where(price_idx == dt)[0][0]

            # reshaping the load array and calculating the costs
            before = np.zeros(i)
            after = np.zeros(prices.shape[0] - load.shape[0] - before.shape[0])
            load = np.hstack([before, load, after])
            return np.dot(load, prices)

    def _get_usage(self, device, date):
        return self.df["usage"].loc[date, device + "_usage"]

    def _get_activity(self, date, hour):
        import numpy as np

        if np.isnan(hour):
            return np.nan
        else:
            dt = np.datetime64(date) + np.timedelta64(int(hour), "h")
            return self.activity.input.loc[dt, "activity"]

    def _get_starting_times(self, device):
        import numpy as np

        # extracts hours in which the device is turned on,
        # conditional on that the device was turned off the hour before
        times = self.df["load"][device].index.to_numpy()
        hour = self.df["load"][device].values
        before = np.insert(hour, 0, 0)[:-1]
        return times[(before == 0) & (hour != 0)]

    def _get_starting_hours(self, device, date):
        import numpy as np
        import pandas as pd

        times = self._get_starting_times(device)
        date = np.datetime64(date) if type(date) != np.datetime64 else date
        times = times[(times >= date) & (times < date + np.timedelta64(1, "D"))]
        hours = (
            pd.Series(times).apply(lambda x: x.hour).to_numpy()
            if times.shape[0] != 0
            else np.nan
        )
        return hours

    def _get_load(self, true_loads, device, date, hour):
        import numpy as np

        try:
            dt = np.datetime64(date) + np.timedelta64(hour, "h")
        # if hour == NaN, return zero load profile
        except ValueError:
            return np.zeros(24)
        try:
            return true_loads[device].loc[dt].values
        except KeyError as ke:
            # return a zero load profile if the datetime index was not found
            if str(ke).split("(")[0] == "numpy.datetime64":
                return np.zeros(24)
            # in any other case raise the key error
            else:
                raise ke

    # evaluation: performance metrics
    # -------------------------------------------------------------------------------------------
    def evaluate(self, activity_threshold, usage_threshold):
        name = f"activity: {activity_threshold}; usage: {usage_threshold}"
        self.pipeline('recommendation', activity_threshold=activity_threshold, usage_threshold=usage_threshold, dates='all')
        self.results[name] = self._evaluate()

    def _evaluate(self):
        import numpy as np

        df = self.output["recommendation"].copy()

        # usage and activity target
        df["usage_true"] = df.apply(lambda row: self._get_usage(row["device"], row.name), axis=1)
        df["activity_true"] = df.apply(lambda row: self._get_activity(row.name, row["recommendation"]), axis=1)
        df["acceptable"] = df["usage_true"] * df["activity_true"]

        # starting times
        df["starting_times"] = df.apply(
            lambda row: self._get_starting_hours(row["device"], row.name), axis=1
        )
        df["relevant_start"] = abs(df["starting_times"] - df["recommendation"])
        df.loc[df["starting_times"].notna(), "relevant_start"] = df[
            df["starting_times"].notna()
        ].apply(
            lambda row: row["starting_times"][np.argmin(row["relevant_start"])], axis=1
        )

        # actual loads
        true_loads = self.load.get_true_loads(self.config["user_input"]["shiftable_devices"])
        df["load"] = df.apply(lambda row: self._get_load(true_loads, row["device"], row.name, row["relevant_start"]), axis=1)

        # calculating costs
        df["cost_no_recommendation"] = df.apply(lambda row: self.calculate_cost(row.name, row["relevant_start"], row["load"]), axis=1)
        df["cost_recommendation"] = df.apply(lambda row: self.calculate_cost(row.name, row["recommendation"], row["load"]),axis=1)
        df["savings"] = df["cost_no_recommendation"] - df["cost_recommendation"]
        df["relative_savings"] = df["savings"] / df["cost_no_recommendation"]

        return df[
            [
                "device",
                "recommendation",
                "acceptable",
                "relevant_start",
                "cost_no_recommendation",
                "cost_recommendation",
                "savings",
                "relative_savings",
            ]
        ]

    def _result_to_summary(self, result):
        return {
            "n_recommendations": result["recommendation"].count(),
            "acceptable": result["acceptable"].mean(),
            "total_savings": (result["acceptable"] * result["savings"]).sum(),
            "relative_savings_mean": result["relative_savings"].mean(),
            "relative_savings_median": result["relative_savings"].median(),
        }

    def results_to_summary(self):
        import pandas as pd

        summary = {
            name: self._result_to_summary(result)
            for name, result in self.results.items()
        }
        return pd.DataFrame.from_dict(summary, orient="index")

    # evaluation: grid search and sensitivity
    # -------------------------------------------------------------------------------------------
    def grid_search(self, activity_thresholds, usage_thresholds):
        import itertools
        from tqdm import tqdm

        # updating the config
        try:
            self.config['evaluation']
        except:
            self.config['evaluation'] = {}

        self.config['evaluation']['grid_search'] = {}
        self.config['evaluation']['grid_search']['activity_thresholds'] = list(activity_thresholds)
        self.config['evaluation']['grid_search']['usage_thresholds'] = list(usage_thresholds)

        # testing candidate thresholds
        iterator = itertools.product(activity_thresholds, usage_thresholds)
        for thresholds in tqdm(list(iterator)):
            self.evaluate(thresholds[0], thresholds[1])

    def get_sensitivity(self, target):
        import pandas as pd

        df = self.results_to_summary()
        sensitivity = pd.DataFrame()
        for threshold_name in df.index:
            thresholds = threshold_name.split("; ")
            activity_threshold, usage_threshold = [th.split(": ")[1] for th in thresholds]
            sensitivity.loc[activity_threshold, usage_threshold] = df.loc[threshold_name, target]
        # sort and name rows and columns
        sensitivity = sensitivity.loc[sorted(sensitivity.index), :]
        sensitivity = sensitivity.loc[:, sorted(sensitivity.columns)]
        sensitivity.index.name = "activity_threshold"
        sensitivity.columns.name = "usage_threshold"
        return sensitivity

    def get_optimal_thresholds(self):
        df = self.results_to_summary()
        result = df.sort_values(by='total_savings').iloc[-1, :]
        thresholds = result.name.split('; ')
        thresholds = [threshold.split(': ') for threshold in thresholds]
        thresholds = {f"{threshold}_threshold": value for threshold, value in thresholds}
        self.config['evaluation']['grid_search']['optimal_thresholds'] = thresholds
        return thresholds

    def thresholds_to_index(self, activity_threshold='optimal', usage_threshold='optimal'):
        if activity_threshold == 'optimal':
            activity_threshold = self.config['evaluation']['grid_search']['optimal_thresholds']['activity_threshold']
        if usage_threshold == 'optimal':
            usage_threshold = self.config['evaluation']['grid_search']['optimal_thresholds']['usage_threshold']
        return f"activity: {activity_threshold}; usage: {usage_threshold}"

    def optimal_result_to_summary(self):
        import pandas as pd
        optimal_thresholds = self.get_optimal_thresholds()
        optimal_thresholds_index = self.thresholds_to_index()
        result = self.results_to_summary().loc[optimal_thresholds_index, :]
        result = pd.concat([result, pd.Series(optimal_thresholds)], axis=1).T

        result.name = self.config['data']['household']
        return result

# Explainability Agent
# ===============================================================================================
class Explainability_Agent:
    def __init__(self, model_activity, X_train_activity, X_test_activity, best_hour, model_usage,
               X_train_usage, X_test_usage, model_type):
        self.model_activity = model_activity
        self.model_type = model_type
        self.X_train_activity = X_train_activity
        self.X_test_activity = X_test_activity
        self.best_hour = best_hour
        self.model_usage = model_usage
        self.X_train_usage = X_train_usage
        self.X_test_usage = X_test_usage

    def feature_importance(self):
        if self.model_type == "logit":
            X_train_summary = shap.sample(self.X_train_activity, 100)
            self.explainer_activity = shap.KernelExplainer(self.model_activity.predict_proba, X_train_summary)

        elif self.model_type == "ada":
            X_train_summary = shap.sample(self.X_train_activity, 100)
            self.explainer_activity = shap.KernelExplainer(self.model_activity.predict_proba, X_train_summary)

        elif self.model_type == "knn":
            X_train_summary = shap.sample(self.X_train_activity, 100)
            self.explainer_activity = shap.KernelExplainer(self.model_activity.predict_proba, X_train_summary)

        elif self.model_type == "random forest":

            self.explainer_activity = shap.TreeExplainer(self.model_activity, self.X_train_activity)

        elif self.model_type == "xgboost":
            self.explainer_activity = shap.TreeExplainer(self.model_activity, self.X_train_activity, model_output='predict_proba')
        else:
            raise InputError("Unknown model type.")


        self.shap_values = self.explainer_activity.shap_values(
            self.X_test_activity.iloc[self.best_hour, :])

        feature_names_activity = list(self.X_train_activity.columns.values)

        vals_activity = self.shap_values[1]

        feature_importance_activity = pd.DataFrame(list(zip(feature_names_activity, vals_activity)),
                                                   columns=['col_name', 'feature_importance_vals'])
        feature_importance_activity.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)

        # usage
        if self.model_type == "logit":
            X_train_summary = shap.sample(self.X_train_usage, 100)
            self.explainer_usage = shap.KernelExplainer(self.model_usage.predict_proba, X_train_summary)


        elif self.model_type == "ada":
            X_train_summary = shap.sample(self.X_train_usage, 100)
            self.explainer_usage = shap.KernelExplainer(self.model_usage.predict_proba, X_train_summary)

        elif self.model_type == "knn":
            X_train_summary = shap.sample(self.X_train_usage, 100)
            self.explainer_usage = shap.KernelExplainer(self.model_usage.predict_proba, X_train_summary)

        elif self.model_type == "random forest":

            self.explainer_usage = shap.TreeExplainer(self.model_usage, self.X_train_usage)

        elif self.model_type == "xgboost":
            self.explainer_usage = shap.TreeExplainer(self.model_usage, self.X_train_usage, model_output='predict_proba')
        else:
            raise InputError("Unknown model type.")


        self.shap_values_usage = self.explainer_usage.shap_values(
            self.X_test_usage,  check_additivity=False)

        feature_names_usage = list(self.X_train_usage.columns.values)

        vals = self.shap_values_usage[1]

        feature_importance_usage = pd.DataFrame(list(zip(feature_names_usage, vals)),
                                                columns=['col_name', 'feature_importance_vals'])
        feature_importance_usage.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)

        return feature_importance_activity, feature_importance_usage, self.explainer_activity, self.explainer_usage, self.shap_values, self.shap_values_usage, self.X_test_activity, self.X_test_usage
    
    def explanation_from_feature_importance_activity(self, feature_importance_activity, date, best_hour, diagnostics=False):
        self.feature_importance_activity = feature_importance_activity
        self.diagnostics = diagnostics

        sentence = 'We based the recommendation on your past activity and usage of the device. '

        # Check if required columns exist in feature_importance_activity
        required_cols = ['activity_lag_24', 'activity_lag_48', 'activity_lag_72']
        for col in required_cols:
            if not any(feature_importance_activity['col_name'] == col):
                print(f"Warning: Missing expected column '{col}' in feature_importance_activity.")
                return f"Missing data prevents generating a detailed explanation for activity on {date} at hour {best_hour}."

        # Determine past activity status
        try:
            if (
                self.X_test_activity['activity_lag_24'].iloc[self.best_hour] == 0
                and self.X_test_activity['activity_lag_48'].iloc[self.best_hour] == 0
                and self.X_test_activity['activity_lag_72'].iloc[self.best_hour] == 0
            ):
                active_past = 'not '
            else:
                active_past = ''
        except KeyError as e:
            print(f"KeyError in accessing activity lags: {e}")
            return "Activity lag data is missing or corrupted."

        # Input the activity lag with the strongest feature importance
        try:
            importance_values = [
                feature_importance_activity.loc[
                    feature_importance_activity['col_name'] == f'activity_lag_{lag}', 'feature_importance_vals'
                ].to_numpy()[0] if not feature_importance_activity.loc[
                    feature_importance_activity['col_name'] == f'activity_lag_{lag}'
                ].empty else None
                for lag in [24, 48, 72]
            ]

            if any(val is not None and val >= 0 for val in importance_values):
                FI_lag = np.argmax([val if val is not None else float('-inf') for val in importance_values])
                activity_lag = ['day', 'two days', 'three days'][FI_lag]
                part1 = f"We believe you are active today since you were {active_past}active during the last {activity_lag}."
            else:
                part1 = "We could not identify strong activity trends."
        except Exception as e:
            print(f"Error in calculating activity lag importance: {e}")
            part1 = "Activity data is incomplete or invalid."

        # Weather explanation
        try:
            weather_hourly = pd.read_pickle('../export/weather_unscaled_hourly.pkl')
            weather_features = ['dwpt', 'rhum', 'temp', 'wdir', 'wspd']
            d = {
                'features': weather_features,
                'labels': ['dewpoint', 'relative humidity', 'temperature', 'wind direction', 'windspeed'],
                'feature_importances': [
                    feature_importance_activity.loc[
                        feature_importance_activity['col_name'] == feature, 'feature_importance_vals'
                    ].to_numpy()[0] if not feature_importance_activity.loc[
                        feature_importance_activity['col_name'] == feature
                    ].empty else None
                    for feature in weather_features
                ],
                'feature_values': [
                    weather_hourly[date].iloc[best_hour, -5:].loc[feature] if feature in weather_hourly[date].iloc[best_hour, -5:] else None
                    for feature in weather_features
                ],
            }
            df = pd.DataFrame(data=d)
            sorted_df = df.sort_values(by='feature_importances', ascending=False, na_position='last')
            if sorted_df.iloc[0]['feature_importances'] is not None and sorted_df.iloc[0]['feature_importances'] >= 0:
                weather1 = sorted_df.iloc[0]['labels']
                value1 = round(sorted_df.iloc[0]['feature_values'], 2)
                part2 = f"The weather condition ({weather1}: {value1}) supports the recommendation."
            else:
                part2 = "Weather conditions do not strongly influence the recommendation."
        except Exception as e:
            print(f"Error in weather data processing: {e}")
            part2 = "Weather data is unavailable or invalid."

        # Time feature explanation
        try:
            day_names = [
                'day_name_Monday', 'day_name_Tuesday', 'day_name_Wednesday',
                'day_name_Thursday', 'day_name_Saturday', 'day_name_Sunday'
            ]
            strongest_day = next(
    (
        day
        for day in day_names
        if not feature_importance_activity.loc[feature_importance_activity['col_name'] == day].empty
        and feature_importance_activity.loc[feature_importance_activity['col_name'] == day, 'feature_importance_vals'].to_numpy()[0] >= 0
    ),
    None
)

            hour_importance = feature_importance_activity.loc[
                feature_importance_activity['col_name'] == 'hour', 'feature_importance_vals'
            ].to_numpy()[0] if not feature_importance_activity.loc[
                feature_importance_activity['col_name'] == 'hour'
            ].empty else None

            if strongest_day and hour_importance is not None and hour_importance >= 0:
                part3 = "The weekday and hour strengthen this prediction."
            elif strongest_day:
                part3 = "The weekday strengthens this prediction."
            elif hour_importance is not None and hour_importance >= 0:
                part3 = "The hour strengthens this prediction."
            else:
                part3 = "Time features do not strongly influence the prediction."
        except Exception as e:
            print(f"Error in processing time features: {e}")
            part3 = "Time feature data is incomplete or invalid."

        # Combine explanations
        sentence_activity = f"{part1} {part2} {part3}".strip()
        explanation_sentence = f"{sentence} {sentence_activity}"

        return explanation_sentence

    def explanation_from_feature_importance_usage(self, feature_importance_usage, date, diagnostics=False):

        self.feature_importance_usage= feature_importance_usage
        self.diagnostics = diagnostics

        if self.X_test_usage['active_last_2_days'] == 0:
            active_past = 'not'
        else:
            active_past = ''

        if feature_importance_usage.loc[0, 'feature_importance_vals'] >= 0 or feature_importance_usage.loc[1, 'feature_importance_vals'] >= 0:

            FI_lag = np.argmax([feature_importance_usage.loc[0, 'feature_importance_vals'],
                                feature_importance_usage.loc[1, 'feature_importance_vals']])

            if FI_lag == 0:
                device_usage = ""
                number_days = 'day'
            elif FI_lag == 1:
                device_usage = ""
                number_days = 'two days'
            else:
                device_usage = " not"
                number_days = 'two days'

            part1 = f" and have{device_usage} used the device in the last {number_days}"

        else:
            part1= ""

            # Weather data processing
        weather_daily = pd.read_pickle('../export/weather_unscaled_daily.pkl')
        try:
            d = {
                'features': ['dwpt', 'rhum', 'temp', 'wdir', 'wspd'],
                'labels': ['dewpoint', 'relative humidity', 'temperature', 'wind direction', 'windspeed'],
                'feature_importances': [
                    feature_importance_usage.loc[feature_importance_usage['col_name'] == feature, 'feature_importance_vals']
                    .to_numpy()[0] if not feature_importance_usage.loc[feature_importance_usage['col_name'] == feature].empty else None
                    for feature in ['dwpt', 'rhum', 'temp', 'wdir', 'wspd']
                ],
                'feature_values': [
                    weather_daily.loc[date, feature] if feature in weather_daily.columns else None
                    for feature in ['dwpt', 'rhum', 'temp', 'wdir', 'wspd']
                ]
            }
            df = pd.DataFrame(data=d)
            sorted_df = df.sort_values(by='feature_importances', ascending=False, na_position='last')

            # Generate explanation based on the highest feature importance
            if not sorted_df.empty and sorted_df.iloc[0]['feature_importances'] is not None:
                weather1 = sorted_df.iloc[0]['labels']
                value1 = round(sorted_df.iloc[0]['feature_values'], 2)
                part2 = f"The weather condition ({weather1}: {value1}) supports the recommendation."

                # Check if a second feature is significant
                if len(sorted_df) > 1 and sorted_df.iloc[1]['feature_importances'] is not None:
                    weather2 = sorted_df.iloc[1]['labels']
                    value2 = round(sorted_df.iloc[1]['feature_values'], 2)
                    part2 = f"The weather conditions ({weather1}: {value1}, {weather2}: {value2}) support the recommendation."
            else:
                part2 = "Weather conditions do not strongly influence the recommendation."

        except Exception as e:
            print(f"Error processing weather data: {e}")
            part2 = "Weather data is unavailable or invalid."

        # Construct the explanation sentence
        sentence_usage = (
            f"We believe you are likely to use the device in the near future since you "
            f"were {active_past}active during the last 2 days. {part2}"
        )
        return sentence_usage
