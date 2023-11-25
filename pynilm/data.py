import time

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from nilmtk import DataSet, MeterGroup
from pynilm.data_utils import chunkify

    
class DataWrapper:
    def __init__(
        self, 
        dataset_path,
        building,
        appliances,
        sample_period,
        start=None, 
        end=None,
        fillna=0,
        return_mode='dataframe',
        windows_size=None,
        windows_stride=None,
        get_activations=True,
        activations_type=int,
        mains_label='mains',
        debug=False
        ):
        """
        Args:
            dataset (nimltk.DataSet): The NILMTK dataset.
            building (int): The building to read the records from.
            appliances (list): The list of appliances used in the experiment.
            sample_period (int): The sample period of the records.
            start (str): The starting date in the format "{month}-{day of month}-{year}" e.g. "05-30-2012".
            end (str): The final date in the format "{month}-{day of month}-{year}" e.g. "08-30-2012".
            return_mode (str): The mode of data processing/storing output. If `'dataframe'` (default), only one 
                dataframe containing mains and all appliances power consumption as columns is stored in 
                `data` property (and no activation is calculated); else if `'dict'`, `data` is splitted in 
                appliance level as a dict object with structure like 
                    `{
                        appliance_label: pd.DataFrame(
                                            data={
                                                'mains': [], 
                                                appliance_label: []
                                            }
                                        )
                      }`.
            windows_size (None OR int OR dict): The windows size configurations; if `None`, original length 
                is assumed; else if `int`, all appliances will be the same size; else if `dict`, the 
                appliance has its specifc window size.
            windows_stride (None OR int): The windows stride configurations; if `None`, original length 
                is assumed; else if `int`, all appliances will be the same stride value.
            get_activations (bool): If activations is calculated in the process; if `True`, the activation
                of individual mensurements is calculated by appliance; else if `False` no activations is 
                calculated.
            activations_type (type):  The type of activatation used in the mensurements (cast).
            fillna (int): The default value to fill Null values in data.
        """
        self.dataset = DataSet(dataset_path)
        self.building = building
        self.appliances = appliances
        self.sample_period = sample_period
        self.start = start 
        self.end = end
        self.fillna = fillna
        self.return_mode = return_mode
        self.windows_size = windows_size      
        self.windows_stride = windows_stride
        self.get_activations = get_activations
        self.activations_type = activations_type
        self.mains_label = mains_label
        self.debug = debug        
        
        self.data = None
        self.activations = None
        self.metergroup = {}
        
        self.run()
        
        
    def run(self):
        """
        Loads mains and appliances data
        """
        df_mains, metegroup_mains = self.read_mains()
        self.data = {
            self.mains_label: df_mains
        }
        self.metergroup[self.mains_label] = metegroup_mains
        
        # Iterate over appliance list
        for a in self.appliances:
            df_appliance, metegroup_appliance = self.read_appliance(appliance_label=a)
            
            self.data[a] = df_appliance#[df_appliance.columns[0]].values
            self.metergroup[a] = metegroup_appliance
            
        # Combining multiple power consumptions (mains and appliances) in one dataframe
        self.data = self.join_dataframes()
        
        # Generating dict-like structure of data (by appliance)
        if self.return_mode == 'dict':
            data = {}
            for a in self.appliances:
                data[a] = pd.DataFrame(
                    data={
                        self.mains_label: self.data[self.mains_label].values,
                        a: self.data[a].values
                    },
                    index=self.data.index
                )   
            self.data = data
            
        # Postprocessing...
        
        # Generating windows, if needed
        if self.windows_size:
            
            # if data is a pd.DataFrame and Windows is int (fixed)
            if isinstance(self.data, pd.DataFrame) and \
                isinstance(self.windows_size, int):
                self.data = self.generate_windows(
                        data=self.data,
                        window_size=self.windows_size,
                        window_stride=self.windows_stride
                    )
                self.mains = np.array([data[self.mains_label].values for data in self.data])
                
            # if is dict (return_mode == 'appliance)    
            elif isinstance(self.data, dict):
                
                data = {}
                mains = {}
                
                for a in self.appliances:

                    if isinstance(self.windows_size, int):
                        windows_size = self.windows_size
                    else:
                        windows_size = self.windows_size[a]

                    data[a] = self.generate_windows(
                            data=self.data[a][[self.mains_label, a]],
                            window_size=self.windows_size,
                            window_stride=self.windows_stride
                        )
                    mains[a] = np.array([d[self.mains_label].values for d in data[a]])
                
                self.data = data
                self.mains = mains

            
        if self.get_activations:
            
            activations = {}
                
            for a in self.appliances:
                
                activations[a] = []
            
                # if data is a pd.DataFrame
                if isinstance(self.data, pd.DataFrame):
                    
                    df_windows = [self.data[[self.mains_label, a]]]
                
                # if is a dict (appliance-level)
                elif isinstance(self.data, dict):
                    
                    df_windows = self.data[a]
                    
                    
                elif isinstance(self.data, list):   
                    df_windows = self.data
                
                for df in df_windows:
                    status = self.is_on(df, a)
                    activations[a].append(status)
            
            self.activations = activations
            
#             # if is dict (return_mode == 'appliance)    
#             elif isinstance(self.data, dict):
            
#             if isinstance(self.windows_size, dict):
                
#             else:
#                 df = df_windows[i][[redd_train.mains_label, a]]
#                 status = redd_train.is_on(df, a)
#                 appliance_status[a].append(status)
        
        
    def read_mains(self):
        """
        Loads the data of the specified appliances.
        Args:
            Debug
        Returns:
            Returns a tuple containing the respective DataFrame and MeterGroup of the data that are read.
        """
        self.dataset.set_window(start=self.start, end=self.end)
        mains_meter = self.dataset.buildings[self.building].elec.mains()
        if isinstance(mains_meter, MeterGroup):
            metergroup = mains_meter
        else:
            metergroup = MeterGroup(meters=[mains_meter])
        start_time = time.time()
        df = next(metergroup.load(
            sample_period=self.sample_period, 
            physical_quantity='power', 
            ac_type='apparent'))
        if self.debug: print('NILMTK converting mains to dataframe: {}'.format(round(time.time() - start_time, 2)))

        df.fillna(self.fillna, inplace=True)
        df.set_axis(df.columns.map('_'.join), axis=1, inplace=True)
        
        if self.debug:
        
            figure = df.plot().get_figure()
            plt.title(f"{self.dataset.metadata['name'].upper()} | Building #{self.building} - Main\'s Power (sample period={self.sample_period} - from {self.start} to {self.end})")
            plt.show()

            # Summary
            print(df.describe())

            # Samples from main 1 (head and tail)
            print(df.head(10))
            print(df.tail(10))
        
        return df, metergroup
    
    def read_appliance(self, appliance_label):
        """
        Loads the data of the specified appliances.
        Args:
            appliance_label (str): appliance name in the current dataset.
            building (int): The building to read the records from.
            sample_period (int): The sample period of the records.
            start (str): The starting date in the format "{month}-{day of month}-{year}" e.g. "05-30-2012".
            end (str): The final date in the format "{month}-{day of month}-{year}" e.g. "08-30-2012".
        Returns:
            Returns a tuple containing the respective DataFrame and MeterGroup of the data that are read.
        """
        self.dataset.set_window(start=self.start, end=self.end)
        appliance_meter = self.dataset.buildings[self.building].elec.submeters()[appliance_label]
        if isinstance(appliance_meter, MeterGroup):
            metergroup = appliance_meter
        else:
            metergroup = MeterGroup(meters=[appliance_meter])
        start_time = time.time()
        df = next(metergroup.load(
            sample_period=self.sample_period, 
            physical_quantity='power', 
            ac_type='active'))
        if self.debug: print('NILMTK converting mains to dataframe: {}'.format(round(time.time() - start_time, 2)))

        df.fillna(self.fillna, inplace=True)#.set_axis(df.columns.map('_'.join), axis=1, inplace=True)
        df.set_axis(df.columns.map('_'.join), axis=1, inplace=True)
        
        # Synchronizing indexes with mains
        df = self.sync_data(df, self.data[self.mains_label])
        
        if self.debug:
        
            figure = df.plot().get_figure()
            plt.title(f"{self.dataset.metadata['name'].upper()} | Building #{self.building} - {appliance_label}\'s Power (sample period={self.sample_period} - from {self.start} to {self.end})")
            plt.show()

            # Summary
            print(df.describe())

            # Samples from main 1 (head and tail)
            print(df.head(10))
            print(df.tail(10))
        
        return df, metergroup
    
    def sync_data(self, df1, df2):
        """Synchronize data between dataframes based on indexes."""
        return df1.align(df2, fill_value=self.fillna)[0].iloc[:, :-1]
    
    def join_dataframes(self):
        data = {
            self.mains_label: self.data[self.mains_label][self.data[self.mains_label].columns[0]].values
        }
        for a in self.appliances:
            data[a] = self.data[a][self.data[a].columns[0]].values

        return pd.DataFrame(data, index=self.data[self.mains_label].index)
    
    def generate_windows(self, data: pd.DataFrame, window_size: int, window_stride: int=1):
        return chunkify(data, window_size, window_stride)
        
    
    def is_on(self, data: pd.DataFrame, appliance_label: str, threshold=None):
        """
        Returns the status ON/OFF (True/False, 1/0) of appliance in the current sample.
        """
        if not threshold:
            threshold = self.metergroup[appliance_label].on_power_threshold()
        
        status = data[appliance_label].values.mean() > threshold
        return np.array(status).astype(self.activations_type).item()
    
    def plot(self):
        pass