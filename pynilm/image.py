import cv2
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class ImageTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 width=None, 
                 height=None, 
                 cv_interpolation = cv2.INTER_CUBIC,
                 dtype=np.float32):
        self.width = width
        self.height = height
        self.dtype = dtype
        self.cv_interpolation = cv_interpolation
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Perform arbitary transformation
        X_rp = np.empty((len(X), self.width, self.height))
        
        for i, img in enumerate(X):
            
            # Resize
            img = self.__resize(img)
            
            X_rp[i,] = img
            
        return X_rp
    
    def __resize(self, image):
        return cv2.resize(
            image, 
            dsize=(self.width, self.height), 
            interpolation=self.cv_interpolation
        ).astype(self.dtype)

def resize_batch(batch_images,
                 width=None, 
                 height=None, 
                 cv_interpolation = cv2.INTER_AREA,
                 dtype=np.float32):
    
    X = np.empty((len(batch_images), width, height))

    for i, img in enumerate(batch_images):

        # Resize
        img = cv2.resize(
            img, 
            dsize=(width, height), 
            interpolation=cv_interpolation
        ).astype(dtype)

        X[i,] = img

    return X

def flatten_batch(batch):
    return batch.reshape((batch.shape[0], -1))