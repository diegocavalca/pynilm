import os
import random as rn
import cv2
import numpy as np
from sklearn.base import clone
from pyts.image import RecurrencePlot
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from xgboost.sklearn import XGBClassifier
import tensorflow as tf

from collections import OrderedDict
import numpy as np
import pandas as pd
from nilmtk.disaggregate import Disaggregator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten
from tensorflow.keras.models import Sequential

from matplotlib import pyplot as plt
import imageio

class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class DTLFE(Disaggregator):
    def __init__(self, params):
        """
        Parameters to be specified for the model
        """

        self.MODEL_NAME = params.get('model_name', 'DTLFE')
        self.models = OrderedDict()
        self.file_prefix = "{}-temp-weights".format(self.MODEL_NAME.lower())

        # Type of model (classification / regression)
        self.task = params.get('task', 'classification')
        
        # if classification, then binary (sigmoid) or Multilabel (softmax) classifier
        self.classifier_type = params.get('classifier_type', 'binary') 
        
        # Appliance-level power threshold
        self.on_power_threshold = params.get('on_power_threshold', {}) 
        
        # Scaling mains data
        self.scaling = params.get('scaling', False)

        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',99)
        self.n_epochs = params.get('n_epochs', 10 )
        self.batch_size = params.get('batch_size',512)
        self.appliance_params = params.get('appliance_params',{})
        # self.mains_mean = params.get('mains_mean', 1800)
        # self.mains_std = params.get('mains_std', 600)
        if self.sequence_length%2==0:
            print ("Sequence length should be odd!")
            raise (SequenceLengthError)
        
        # DTLFE params
        self.image_size = params.get('image_size', 32)
        IMG_SHAPE = (self.image_size, self.image_size, 3)
            
        self.feature_extractor = params.get('feature_extractor', None)
        if not self.feature_extractor:
            
            self.feature_extractor = VGG16(
                include_top = False, 
                weights = 'imagenet', 
                input_shape = IMG_SHAPE,
                pooling='avg'
            )
            for layer in self.feature_extractor.layers:
                layer.trainable = False
        
        self.preprocess_input = params.get('preprocess_input', None)
        if not self.preprocess_input:
            self.preprocess_input = preprocess_input
        
        self.classifier = params.get('classifier', None)
        if self.classifier == None:
            self.classifier = XGBClassifier()
            
        self.rp_params = params.get('rp_params', {
                "dimension": 1,
                "time_delay": 1,
                "threshold": None,
                "percentage": 10
            })

        self.base_model = BaseModel(
            feature_extractor = self.feature_extractor, 
            preprocess_input = self.preprocess_input,
            classifier=self.classifier,
            rp_params = self.rp_params,
            input_shape=IMG_SHAPE, # 224
            data_type=np.float32,
            normalize=False, 
            standardize=False, 
            rescale=False,
            seed=33
        )

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
        # If no appliance wise parameters are provided, then copmute them using the first chunk
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        print(f"...............{self.MODEL_NAME} partial_fit running...............")
        
        # print('>> mains original', [t.shape for t in train_main])
        # print(train_main[0].head())
        
        # Do the pre-processing, such as  windowing and scaling
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')
        
        # Combining multiple mains readings in only one
        train_main = pd.concat(train_main, axis=0).to_numpy()
        # train_main = train_main.values.reshape((-1, self.sequence_length, 1))
        print('>> mains processed', train_main.shape)
        
        # Combining different appliance readings individually
        print('>>', len(train_appliances), 'appliances processed') 
        new_train_appliances = []
        for app_name, app_readings in train_appliances:
            # app_df = pd.concat(app_df, axis=0)
            app_readings = np.concatenate(app_readings)
            # app_df_values = app_df.values#.reshape((-1, 1))
            # print('>>>>>', app_name, type(app_df), app_df.shape, app_df.columns)
            # print('   ->', app_df_values.shape)
            print('>>>>>', app_name, type(app_readings), app_readings.shape)
            new_train_appliances.append((app_name, app_readings))
        train_appliances = new_train_appliances
        
        # TODO: check classifier type / produce multilabel appliance data
        # if self.classifier_type == 'multilabel':
            
        
        for appliance_name, power in train_appliances:
            
            # Check if the appliance was already trained. If not then create a new model for it
            if appliance_name not in self.models:
                print("First model training for", appliance_name)
                self.models[appliance_name] = self.base_model
            # Retrain the particular appliance
            else:
                print("Started Retraining model for", appliance_name)

            model = self.models[appliance_name]
            if train_main.size > 0:
                # Sometimes chunks can be empty after dropping NANS
                if len(train_main) > 10:
                    # Do validation when you have sufficient samples
                    filepath = self.file_prefix + "-{}-epoch{}.h5".format(
                            "_".join(appliance_name.split()),
                            current_epoch,
                    )
                    #checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,save_best_only=True, mode='min')
                    model.fit(train_main, power)
                    # model.load_weights(filepath)


    def call_preprocessing(self, mains_lst, submeters_lst, method):
        """
        Preprocessing method aiming to prepare mains and appliance power data to the model. 
            The preparation include steps like windowing/padding original time series, 
            scaling and thresholding output (ON/OFF) status, for example.
        The main idea is to generate windows of length equal to 'sequence_length' (subseries), 
            so enabling further steps in the model pipeline (eg. convert to recurrence plot).
            The defaul padding to the windows generate is 1.
        
        Args:
            mains_lst (list): the mains power data of multiple sources, as necessary.
            submeters_lst (list): the N appliances power data.
            method (str): flag indicating if is 'train' or 'test' time.
        Return:
            mains_df_list (list): list of mains prepared data;
            appliance_list (list):             
        """
            

        if method == 'train':
            # Preprocessing for the train data           
            
            # # gif
            # filenames = [] 
            
            # Preparing mains data
            mains_df_list = []
            for mains in mains_lst:
                
                # Generating sliding windows
                windows_mains = self.__generate_windows(mains)
                
                # for i, window in enumerate(windows_mains[:1000]):
                #     # print('  -> window',i+1, '|', window.sum(), window.min(), window.max(), window.mean())
                #     plt.plot(window)
                #     plt.title(f'window {i+1}')
                #     # gif
                #     filename = f'images/window_{i+1}.png'
                #     filenames.append(filename)
                #     # gif - save frame
                #     plt.savefig(filename)
                #     plt.close()
                # # gif - build it
                # with imageio.get_writer('images/sample.gif', mode='I') as writer:
                #     for filename in filenames:
                #         image = imageio.imread(filename)
                #         writer.append_data(image)
                # # gif - Remove files
                # for filename in set(filenames):
                #     os.remove(filename)
                
                # Scaling windows values
                if self.scaling:
                    windows_mains = self.__scaling(windows_mains)
                
                # Appending new data
                mains_df_list.append(pd.DataFrame(windows_mains))
            
            # Preparing appliances data
            print('>> preprocessing appliances data')
            appliance_list = []
            
            # Iterate over appliances...
            for app_index, (app_name, app_df_list) in enumerate(submeters_lst):
                
                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                else:
                    print ("Parameters for ", app_name ," were not found!")
                    raise ApplianceNotFoundError()
                
                #  = params.get('task', 'classification')
                # self.model_type = params.get('model_type', 'binary')
                app_on_power_threshold = self.on_power_threshold[app_name]
                print('*** CHECK app_on_power_threshold:', app_on_power_threshold)

                processed_appliance_dfs = []
                for app_df in app_df_list:
                    
                    
                    # Appliance power consumption
                    app_power_consumption = app_df[app_df.columns[0]].values#.reshape((-1, 1))
                    # print("***", type(app_power_consumption), app_power_consumption.shape)
                    
                    if self.task == 'regression':
                        
                        # Return as a list of dataframe
                        processed_appliance_dfs.append(app_power_consumption)#pd.DataFrame(app_power_consumption))
                    
                    elif self.task == 'classification':
                        
                        # Converting consumption to active status (thresholding)
                        app_active_status = (app_power_consumption > app_on_power_threshold).astype(int)
                        
                        # Return as a list of dataframe
                        processed_appliance_dfs.append(app_active_status)#pd.DataFrame(app_active_status))
                        
                    else:
                        raise Exception ('Invalid task informed!!')
                        
                        
                appliance_list.append((app_name, processed_appliance_dfs))
                
            # print('Appliance list >>>', [(l[0], app_l.shape) for l in appliance_list for app_l in l[1]])
                
            return mains_df_list, appliance_list

        else:
            
            # Preprocessing for the test data
            mains_df_list = []

            for mains in mains_lst:
                
                # new_mains = mains.values.flatten()
                # n = self.sequence_length
                # units_to_pad = n // 2
                # new_mains = np.pad(new_mains,(units_to_pad,units_to_pad),'constant',constant_values=(0,0))
                # new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                # new_mains = (new_mains - self.mains_mean) / self.mains_std
                
                # Generating sliding windows
                windows_mains = self.__generate_windows(mains)
                
                # Scaling windows values
                if self.scaling:
                    windows_mains = self.__scaling(windows_mains)
                
                # Appending new data
                mains_df_list.append(pd.DataFrame(windows_mains))
                
            return mains_df_list
        
        
    def __scaling(self, data):
        """
        Scaling data using standarlization method.
        """
        return (data - data.mean()) / data.std()
        
    def __generate_windows(self, series):
        """
        Generating window subseries from contiguous series.
        
        Args:
            series (np.array): original series.
            
        Return:
            windows_series (np.array): windowed series.
        """
        
        # Converting series values into list
        series = series.values.flatten()

        # Calculating necessary padding 
        #  included in begin and end of the original series,
        #  enabling rolling windows (slide/padding = 1)
        n = self.sequence_length
        units_to_pad = n // 2
        padded_series = np.pad(
            series,
            (units_to_pad, units_to_pad),
            'constant',
            constant_values=(0, 0)
        )
        # Generating sliding windows
        windows_series = np.array([padded_series[i:i + n] for i in range(len(padded_series) - n + 1)])
        
        return windows_series
                    
    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        if model is not None:
            self.models = model

        # Preprocess the test mains such as windowing and normalizing

        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None, method='test')
            
        # # Combining multiple mains readings in only one
        # test_main = pd.concat(test_main, axis=0).to_numpy()
        # # train_main = train_main.values.reshape((-1, self.sequence_length, 1))
        
        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.to_numpy()
            print('>> test_main processed', test_main.shape)
            # test_main = test_main.reshape((-1, self.sequence_length, 1))
            disggregation_dict = {}
            for appliance in self.models:
                prediction = self.models[appliance].predict(test_main)#,batch_size=self.batch_size)
                # prediction = self.appliance_params[appliance]['mean'] + prediction * self.appliance_params[appliance]['std']
                valid_predictions = prediction.flatten()
                # valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    def set_appliance_params(self,train_appliances):
        
        # Find the parameters using the first
        for (app_name,df_list) in train_appliances:
            l = np.array(pd.concat(df_list,axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std<1:
                app_std = 100
            
            on_power_threshold = 10 # default value
            if app_name in self.on_power_threshold:
                on_power_threshold = self.on_power_threshold                
            
            self.appliance_params.update({
                app_name:{
                    'mean': app_mean,
                    'std': app_std,
                    'on_power_threshold': on_power_threshold
                }
            })
        print(self.appliance_params)

        
class BaseModel:
    def __init__(self,
                feature_extractor=None, 
                preprocess_input=None,
                classifier=None,
                rp_params={
                    "dimension": 1,
                    "time_delay": 1,
                    "threshold": None,
                    "percentage": 10
                },
                input_shape=(224, 224, 3),
                data_type=np.float32,
                normalize=False, 
                standardize=False, 
                rescale=False,
                seed=33
                ):
        self.model = None
        self.feature_extractor = feature_extractor
        self.preprocess_input = preprocess_input
        self.classifier = classifier
        self.rp_params = rp_params
        self.input_shape = input_shape
        self.data_type = data_type
        self.normalize = normalize
        self.standardize = standardize
        self.rescale = rescale
        self.seed = seed

        # Garantindo reprodutibilidade
        # Travar Seed's
        np.random.seed(self.seed)
        rn.seed(self.seed)
        os.environ['PYTHONHASHSEED']=str(self.seed)


    def seq_to_rp(self, X):
        
        X_rp = np.empty((len(X), *self.input_shape))
        
        for i, x in enumerate(X):
            
            img = RecurrencePlot(**self.rp_params).fit_transform([x])[0]
            img = cv2.resize(
                    img, 
                    dsize=self.input_shape[:2], 
                    interpolation=cv2.INTER_CUBIC
                ).astype(self.data_type)

            if np.sum(img) > 0:
                # TODO: improve fit/predict statistics
                # Normalizar
                if self.normalize:
                        img = (img - img.min()) / (img.max() - img.min()) # MinMax (0,1)
                    #img = (img - img.mean()) / np.max([img.std(), 1e-4])

            #     # centralizar
            #     if centralizar:
            #         img -= img.mean()

                # Padronizar
                elif self.standardize:
                    img = (img - img.mean())/img.std()#tf.image.per_image_standardization(img).numpy()
                    
                elif self.rescale:
                    img = (img - img.min()) / (img.max() - img.min())

            # N canais
            img = np.stack([img for i in range(self.input_shape[-1])],axis=-1).astype(self.data_type)     
            
            X_rp[i,] = img
            
        return X_rp
    
    def preprocessing(self, X):
        # Seq to RP image
        recurrence_plots = self.seq_to_rp(X)
        # Image preprocessing
        return self.preprocess_input(recurrence_plots).astype(self.data_type) # TODO: avaliar impacto na precisao
    
    def feature_extraction(self, X):
        # Seq -> RP -> Preproc TL
        images = self.preprocessing(X)
        print('images = ', type(images), images.shape)
        # Preprocessed Image Feat. Extract.
        features = self.feature_extractor.predict(images)#, batch_size=32)
        return features
                
    def fit(self,X,y):
        X_features = self.feature_extraction(X) # Transfer-learning
        self.classifier.fit(X_features, y)
    
    def predict(self,X):
        X_features = self.feature_extraction(X) # Transfer-learning
        y = self.classifier.predict(X_features)
        return y
    
    
