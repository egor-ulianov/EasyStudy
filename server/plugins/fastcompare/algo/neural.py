from abc import ABC

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.applications.inception_v3 import preprocess_input
from keras.src.legacy.preprocessing import image
from keras.src.saving import load_model
from keras.src.utils import load_img, img_to_array

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


from plugins.fastcompare.algo.algorithm_base import AlgorithmBase, Parameter, ParameterType


class NeuralRecommender(AlgorithmBase, ABC):

    def __init__(self, loader, alpha=0.5, **kwargs):
        self.loader = loader
        self.alpha = alpha
        self.image_features = None
        self.thumbnail_similarity = None
        self.cf_similarity = None
        self.combined_similarity = None
        self.model = self._initialize_cnn_model()
        self._all_items = self.loader.ratings_df['item'].unique()

    def _initialize_cnn_model(self):
        feature_layer_model = load_model('/app/plugins/fastcompare/static/feature_extraction_films_current.keras')
        return feature_layer_model

    def _extract_features(self, img_path):
        img = load_img(f"/app{img_path}", target_size=(200, 200))
        img_data = img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = self.model.predict(img_data)
        return features

    def fit(self):
        print(f"Number of all items : {len(self._all_items)}")
        image_paths = [self.loader.get_item_index_image_url(item_index) for item_index in self._all_items]
        self.image_features = np.array([self._extract_features(img_path)[0] for img_path in image_paths])
        self.thumbnail_similarity = cosine_similarity(self.image_features)

        interaction_data = self.loader.ratings_df.pivot(index="user", columns="item", values="rating").fillna(0).values
        self.cf_similarity = cosine_similarity(interaction_data.T)

        self.combined_similarity = self.alpha * self.thumbnail_similarity + (1 - self.alpha) * self.cf_similarity

    def predict(self, selected_items, filter_out_items, k):
        print(f"Selected items: {selected_items}")
        print(f"Filtered items: {filter_out_items}")
        print(f"Scores form: {np.shape(self.combined_similarity)}")

        combined_scores = np.sum(self.combined_similarity[selected_items], axis=0)
        for item in filter_out_items:
            combined_scores[item] = -np.inf
        recommended_indices = np.argsort(-combined_scores)[:k]

        return recommended_indices

    @classmethod
    def name(cls):
        return "Combined Neural Recommender System"

    @classmethod
    def parameters(cls):
        return [
            Parameter(
                "alpha",
                "float",
                0.5,
                "Weight for combining thumbnail and CF similarities"
            )
        ]
