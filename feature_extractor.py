import numpy as np
     
class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self
    def transform(self, X_df):
        X_df_new = X_df.copy()

        X_df_new = compute_rolling_min(X_df_new, 'Beta', '1d')
        X_df_new = compute_rolling_max(X_df_new, 'Beta', '2h')
        X_df_new = compute_rolling_median(X_df_new, 'Beta', '2h')
            
        # B
        X_df_new = compute_rolling_min(X_df_new, 'B', '1d')
        X_df_new = compute_rolling_max(X_df_new, 'B', '1d')
            
        # By_rms
        X_df_new = compute_rolling_median(X_df_new, 'By_rms', '2h')
            
        # Bx_rms
        X_df_new = compute_rolling_min(X_df_new, 'Bx_rms', '1d')
        X_df_new = compute_rolling_max(X_df_new, 'Bx_rms', '1d')
        X_df_new = compute_rolling_min(X_df_new, 'Bz_rms', '1d')
        X_df_new = compute_rolling_median(X_df_new, 'Bz_rms', '2h')
            
        # V
        # X_df_new = compute_rolling_std(X_df_new, 'V', '2h')
        X_df_new = compute_rolling_min(X_df_new, 'V', '1d')
            
        # Vth
        X_df_new = compute_rolling_max(X_df_new, 'Vth', '2h')

        X_df_new = compute_rolling_std(X_df_new, 'Beta', '6h')
        X_df_new = compute_rolling_std(X_df_new, 'B', '6h')
        X_df_new = compute_rolling_std(X_df_new, 'Vth', '6h')
        X_df_new = compute_rolling_std(X_df_new, 'V', '6h')       
        X_df_new = compute_rolling_std(X_df_new, 'RmsBob', '6h')
        X_df_new = compute_rolling_std(X_df_new, 'Beta', '18h')
        X_df_new = compute_rolling_std(X_df_new, 'B', '18h')
        X_df_new = compute_rolling_std(X_df_new, 'Vth', '18h')
        X_df_new = compute_rolling_std(X_df_new, 'V', '18h')       
        X_df_new = compute_rolling_std(X_df_new, 'RmsBob', '18h')
            
        X_df_new = cart_to_sph(X_df_new, 'B', 'Bx', 'Bz')
        X_df_new = cart_to_sph(X_df_new, 'V', 'Vx', 'Vz')
            
        X_df_new = compute_rolling_std(X_df_new, 'B_phi', '6h')
        X_df_new = compute_rolling_std(X_df_new, 'B_theta', '6h')
        X_df_new = compute_rolling_std(X_df_new, 'V_phi', '6h')
        X_df_new = compute_rolling_std(X_df_new, 'V_theta', '6h')
            
        X_df_new = compute_rolling_quantile(X_df_new, 'Beta', 9, center=True)
        X_df_new = compute_rolling_quantile(X_df_new, 'B', 9, center=True)
        X_df_new = compute_rolling_quantile(X_df_new, 'Vth', 9, center=True)
        X_df_new = compute_rolling_quantile(X_df_new, 'V', 9, center=True)
        X_df_new = compute_rolling_quantile(X_df_new, 'RmsBob', 9, center=True)
        X_df_new = compute_rolling_mean(X_df_new, 'Beta', '6h')
        X_df_new = compute_rolling_mean(X_df_new, 'B', '6h')
        X_df_new = compute_rolling_mean(X_df_new, 'Vth', '6h')
        X_df_new = compute_rolling_mean(X_df_new, 'V', '6h')       
        X_df_new = compute_rolling_mean(X_df_new, 'RmsBob', '6h')
        X_df_new = compute_rolling_mean(X_df_new, 'Beta', '18h')
        X_df_new = compute_rolling_mean(X_df_new, 'B', '18h')
        X_df_new = compute_rolling_mean(X_df_new, 'Vth', '18h')
        X_df_new = compute_rolling_mean(X_df_new, 'V', '18h')       
        X_df_new = compute_rolling_mean(X_df_new, 'RmsBob', '18h')
            
        X_df_new = compute_rolling_skew(X_df_new, 'Beta', 60, center=True)
        X_df_new = compute_rolling_skew(X_df_new, 'B', 60, center=True)
        X_df_new = compute_rolling_skew(X_df_new, 'Vth', 60, center=True)
        X_df_new = compute_rolling_skew(X_df_new, 'V', 60, center=True)       
        X_df_new = compute_rolling_skew(X_df_new, 'RmsBob', 60, center=True)
            
        X_df_new = compute_rolling_kurt(X_df_new, 'Beta', '60h', center=True)
        X_df_new = compute_rolling_kurt(X_df_new, 'B', '60h', center=True)
        X_df_new = compute_rolling_kurt(X_df_new, 'Vth', '60h', center=True)
        X_df_new = compute_rolling_kurt(X_df_new, 'V', '60h', center=True)       
        X_df_new = compute_rolling_kurt(X_df_new, 'RmsBob', '60h', center=True)
        columns_ext = ['V_phi', 'V_theta', 'B_phi', 'B_theta',
                       'Beta', 'Vth', 'B', 'V', 'RmsBob']
        X_df_new = col_ext(X_df_new, columns_ext)
        return X_df_new   
def col_ext(data, columns_ext):
    return data.loc[:, [xx for xx in data.columns if xx not in columns_ext]]
     
def cart_to_sph(data, feature, featureX, featureZ):
    namePhi = '_'.join([feature, 'phi'])
    nameTheta = '_'.join([feature, 'theta']) 
    data[nameTheta] = np.arccos(data[featureZ]/data[feature])
    data[namePhi] = np.arccos(data[featureX]/(data[feature]*np.sin(data[nameTheta])))
    data[nameTheta] = data[nameTheta].ffill().bfill()
    data[namePhi] = data[namePhi].ffill().bfill()
    return data
     
def compute_rolling_skew(data, feature, time_window, center=False):
    name = '_'.join([feature, str(time_window), 'skew'])
    data[name] = data[feature].rolling(time_window, center=center).skew()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data
     
def compute_rolling_kurt(data, feature, time_window, center=False):
    name = '_'.join([feature, time_window, 'kurt'])
    data[name] = data[feature].rolling(time_window).kurt()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data
     
def compute_rolling_mean(data, feature, time_window, center=False):
    name = '_'.join([feature, time_window, 'mean'])
    data[name] = data[feature].rolling(time_window, center=center).mean()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data
     
def compute_rolling_quantile(data, feature, time_window, center=False):
    name = '_'.join([feature, str(time_window), 'quantile'])
    data[name] = data[feature].rolling(time_window, center=center).quantile(0.75)
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data
     
def compute_rolling_std(data, feature, time_window, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature
     
    Parameters
    ----------
    data : dataframe
    feature : str
       feature in the dataframe we wish to compute the rolling mean from
    time_indow : str
       string that defines the length of the time window passed to `rolling`
    center : bool
       boolean to indicate if the point of the dataframe considered is
       center or end of the window
    """
    name = '_'.join([feature, time_window, 'std'])
    data[name] = data[feature].rolling(time_window, center=center).std()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data
     
def compute_rolling_min(data, feature, time_window, center=False):
    name = '_'.join([feature, time_window, 'min'])
    data[name] = data[feature].rolling(time_window, center=center).min()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data
     
def compute_rolling_max(data, feature, time_window, center=False):
    name = '_'.join([feature, time_window, 'max'])
    data[name] = data[feature].rolling(time_window, center=center).max()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data
     
def compute_rolling_median(data, feature, time_window, center=False):
    name = '_'.join([feature, time_window, 'median'])
    data[name] = data[feature].rolling(time_window, center=center).median()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data