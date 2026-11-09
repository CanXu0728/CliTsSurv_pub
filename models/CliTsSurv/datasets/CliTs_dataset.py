import numpy as np
from torch.utils.data import Dataset

from .utils import positional_encoding


class CliTsSurvDataset(Dataset):
    """
    DeepHit version for survival analysis
    
    Data format: (time_to_event, event_indicator) instead of binary y
    """
    def __init__(self, df, feature_cols, time_to_event_col, event_indicator_col, 
                 time_col, static_feature_cols=[], infer=False, balance=False, incremental=False):
        super(CliTsSurvDataset, self).__init__()
        
        self.infer = infer
        self.df = df
        ids = self.df['id'].unique().tolist()
        
        X_ = []
        time_to_event_ = []  # Survival time (continuous or binned)
        event_indicator_ = []  # 1=event occurred, 0=censored
        times_ = []
        static_X_ = []
        ids_ = []
        
        for i in range(len(ids)):
            pid = ids[i]
            
            x = self.df[df['id']==pid][feature_cols].to_numpy()
            static_x = self.df[df['id']==pid][static_feature_cols].to_numpy()[0]
            
            time = self.df[df['id']==pid][time_col].to_numpy()
            
            # Survival data: time_to_event and event_indicator
            # Get the last record for this patient for survival info
            patient_df = self.df[df['id']==pid]
            time_to_event = patient_df[time_to_event_col].iloc[-1]
            event_indicator = patient_df[event_indicator_col].iloc[-1]
            
            if len(time) > 0 and time_to_event > time[-1]:
                time = np.append(time, time_to_event)
                # Append a zero-valued feature row.
                zero_features = np.zeros(len(feature_cols))
                x = np.vstack([x, zero_features])
            
            time = time[:, np.newaxis].astype(np.float32)
            
            # Handle NaN values
            if np.isnan(time_to_event):
                time_to_event = 0.0
            if np.isnan(event_indicator):
                event_indicator = 0.0
            event_indicator = np.clip(event_indicator, 0, 1)
            
            if incremental:
                for i in range(1, len(x)+1):
                    X_.append(x[:i])
                    static_X_.append(static_x)
                    times_.append(time[:i])
                    time_to_event_.append(time_to_event)
                    event_indicator_.append(event_indicator)
                    ids_.append(pid)
            else:
                X_.append(x)
                static_X_.append(static_x)
                times_.append(time)
                time_to_event_.append(time_to_event)
                event_indicator_.append(event_indicator)
                ids_.append(pid)
                
        if infer or not balance:
            self.X = X_
            self.time_to_event = time_to_event_
            self.event_indicator = event_indicator_
            self.times = times_
            self.static_X = static_X_
            self.ids = ids_
        else:
            # For test set, balance between event and censored samples
            num_events = int(np.sum(event_indicator_))
            X_e = []  # event occurred
            X_c = []  # censored
            times_e = []
            times_c = []
            static_X_e = []
            static_X_c = []
            tte_e = []
            tte_c = []
            ei_e = []
            ei_c = []
            
            for i in range(len(event_indicator_)):
                if event_indicator_[i] == 1:
                    X_e.append(X_[i])
                    times_e.append(times_[i])
                    static_X_e.append(static_X_[i])
                    tte_e.append(time_to_event_[i])
                    ei_e.append(event_indicator_[i])
                else:
                    X_c.append(X_[i])
                    times_c.append(times_[i])
                    static_X_c.append(static_X_[i])
                    tte_c.append(time_to_event_[i])
                    ei_c.append(event_indicator_[i])
            
            # Shuffle and balance censored samples
            if len(X_c) > 0:
                zipped = list(zip(X_c, static_X_c, times_c, tte_c, ei_c))
                np.random.shuffle(zipped)
                X_c, static_X_c, times_c, tte_c, ei_c = zip(*zipped)
                
                # Balance to match event count
                balance_n = min(len(X_c), num_events)
                X_c = list(X_c[:balance_n])
                static_X_c = list(static_X_c[:balance_n])
                times_c = list(times_c[:balance_n])
                tte_c = list(tte_c[:balance_n])
                ei_c = list(ei_c[:balance_n])
            
            self.X = X_e + X_c
            self.static_X = static_X_e + static_X_c
            self.time_to_event = tte_e + tte_c
            self.event_indicator = ei_e + ei_c
            self.times = times_e + times_c
                    
    def __len__(self):
        return len(self.time_to_event)

    def __getitem__(self, idx):
        """
        Returns:
            (x, static_x, time_to_event, event_indicator, times)
            or with id if infer=True
        """
        if self.infer:
            return (self.X[idx], self.static_X[idx], 
                   self.time_to_event[idx], self.event_indicator[idx], 
                   self.times[idx], self.ids[idx])
    
        return (self.X[idx], self.static_X[idx], 
               self.time_to_event[idx], self.event_indicator[idx], 
               self.times[idx])
