from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from koregraph.tools.spectogram import save_images_log_power_spectogram


class LogPowerSpectrogramTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.image_folder = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = []
        for file_path in X:
            image_path = save_images_log_power_spectogram(file_path)
            transformed_X.append(str(image_path))
        return transformed_X


pipeline = Pipeline([("log_power_spectrogram", LogPowerSpectrogramTransformer())])
