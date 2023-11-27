import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from tensorflow.keras import applications


class SequenceToImageTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            image_transformer=None,
            image_shape=(32, 32, 3),
            scaling=True,
            flatten_output=False,
        ):
        # Only store arguments in constructor
        if image_transformer is None:
            raise BaseException('An image_transformer is necessary (see `pyts` package)!')
        self.image_transformer = image_transformer
        self.image_shape = image_shape
        self.scaling = scaling
        self.flatten_output = flatten_output        

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        recurrence_plots = self.image_transformer.fit_transform(X)

        resized_rps = np.array([self._resize_image(rp) for rp in recurrence_plots])

        # Add extra channel through last axis (RGB)
        rgb_images = np.repeat(
            resized_rps[:, :, :, np.newaxis], 
            self.image_shape[-1], 
            axis=-1)

        # Normalize values between [0, 255]
        min_values = rgb_images.min(axis=(1, 2), keepdims=True)
        max_values = rgb_images.max(axis=(1, 2), keepdims=True)
        rgb_images = (rgb_images - min_values) / (max_values - min_values) * 255.0

        # Convert to uint8 (0 e 255)
        rgb_images = rgb_images.astype(np.uint8)
        
        # Scaling (standardization) 
        if self.scaling == True:
            rgb_images = rgb_images / 255.0
        
        # Flatten output (required by regular/not DL models)
        if self.flatten_output == True:
            rgb_images = rgb_images.reshape(rgb_images.shape[0], -1)
        
        return rgb_images

    def _build_transformer(self):
        pass
        
    def _resize_image(self, image):
        resized_image = cv2.resize(
            image, 
            self.image_shape[:2], 
            interpolation=cv2.INTER_LINEAR)
        return resized_image

class DeepLearningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model, weights='imagenet', pooling='avg'):
        self.model = model
        self.weights = weights
        self.pooling = pooling

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feature_extractor, preprocess_input = self._build_transformer()
        if preprocess_input:
            X = preprocess_input(X) # np.array([self.preprocess_input(x) for x in X])
        features = feature_extractor.predict(X)
        return features
    
    def _build_transformer(self):
        self._get_available_models()
        model_module = self._find_module_by_model()

        if model_module is None:
            raise ValueError(f'Modelo não suportado. Modelos disponíveis no momento: {self._list_model_names()}')
        
        return self._get_model_and_preprocessing(module_name=model_module)

    def _get_model_and_preprocessing(self, module_name):
        module = getattr(applications, module_name)
        model = getattr(module, self.model)(weights=self.weights, include_top=False, pooling=self.pooling)
        preprocess_input = None
        if 'preprocess_input' in dir(module):
            preprocess_input = getattr(module, 'preprocess_input')
        return model, preprocess_input
        
    def _get_available_models(self):
        available_models = defaultdict(list)
        for module in dir(applications):
            if not module.startswith('_') \
                and module.islower() \
                    and 'utils' not in module:
                for model_name in dir(getattr(applications, module)):
                    if not model_name.startswith('_') and model_name[0].isupper(): 
                        available_models[module].append(model_name)
        self.available_models = dict(available_models)

    def _find_module_by_model(self):
        for module, models in self.available_models.items():
            if self.model in models:
                return module
        return None

    def _list_model_names(self):
        return [n for m in self.available_models.values() for n in m]
    
    
class RQATransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            params={
                "dimension": 1,
                "time_delay": 1,
                "threshold": None,
                "percentage": 10
            },
            property_names=[
                # "Appliance", #"State",
                "Minimum diagonal line length (L_min)",
                "Minimum vertical line length (V_min)",
                "Minimum white vertical line length (W_min)",
                "Recurrence rate (RR)",
                "Determinism (DET)",
                "Average diagonal line length (L)",
                "Longest diagonal line length (L_max)",
                "Divergence (DIV)",
                "Entropy diagonal lines (L_entr)",
                "Laminarity (LAM)",
                "Trapping time (TT)",
                "Longest vertical line length (V_max)",
                "Entropy vertical lines (V_entr)",
                "Average white vertical line length (W)",
                "Longest white vertical line length (W_max)",
                "Longest white vertical line length inverse (W_div)",
                "Entropy white vertical lines (W_entr)",
                "Ratio determinism / recurrence rate (DET/RR)",
                "Ratio laminarity / determinism (LAM/DET)"
                ],
            model_columns=[
                "Recurrence rate (RR)",
                "Determinism (DET)"
                ]
        ):
        self.params = params
        self.property_names = property_names
        self.model_columns = model_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        rqa_data = []
        
        for x in tqdm(X, total=len(X)):

            # Calculating RQA attributes
            time_series = TimeSeries(x,
                        embedding_dimension=self.params["dimension"],
                        time_delay=self.params["time_delay"])
            settings = Settings(time_series,
                                analysis_type=Classic,
                                neighbourhood=FixedRadius(self.params["percentage"]/100), 
                                # PS.: Utilizando percentage ao inves de threshold 
                                # devido a semanticas distintas entre libs (pyts e pyrqa)
                                # bem como distincao entre RPs (cnn) e RQAs (supervisionado).
                                similarity_measure=EuclideanMetric)
            computation = RQAComputation.create(settings, verbose=False)
            rqa_result = computation.run()
            
            rqa_data.append(np.nan_to_num(rqa_result.to_array()))

        # Numpy to Pandas 
        df_rqa = pd.DataFrame(
            data=rqa_data,
            columns=self.property_names
        )
        
        # Select only specific attributes
        X_rqa = df_rqa[self.model_columns].values

        return X_rqa

    def _resize_image(self, image):
        resized_image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        return resized_image