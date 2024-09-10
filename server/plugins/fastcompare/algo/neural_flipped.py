from abc import ABC

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.src.applications.resnet_v2 import ResNet50V2
from keras.src.applications.resnet_v2 import preprocess_input, decode_predictions
from keras.src.legacy.preprocessing import image
from keras.src.layers import GlobalMaxPooling2D
from keras.src.applications.inception_v3 import preprocess_input
from keras.src.legacy.preprocessing import image
from keras.src.saving import load_model
from keras.src.utils import load_img, img_to_array

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances


from plugins.fastcompare.algo.algorithm_base import AlgorithmBase, Parameter, ParameterType


class NeuralFlippedRecommender(AlgorithmBase, ABC):

    def __init__(self, loader, cf_select=50, **kwargs):
        self.loader = loader
        self.cf_select = cf_select
        self.image_features = None
        self.thumbnail_similarity = None
        self.cf_similarity = None
        self.combined_similarity = None
        self.model = self._initialize_cnn_model_v2()
        self._all_items = self.loader.ratings_df['item'].unique()

    def _initialize_cnn_model_v2(self):
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
        base_model.trainable = False
        model = keras.Sequential([base_model, GlobalMaxPooling2D()])
        return model

    def _extract_features_v2(self, img_path):
        img = load_img(f"/app{img_path}", target_size=(200, 200))
        img_data = img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        print(self.model.predict(img_data))
        features = self.model.predict(img_data).reshape(-1)
        return features

    def fit(self):
        print(f"Number of all items : {len(self._all_items)}")
        image_paths = [self.loader.get_item_index_image_url(item_index) for item_index in self._all_items]
        self.image_features = np.array([self._extract_features_v2(img_path) for img_path in image_paths])
        print(f"Shape of image features: {np.shape(self.image_features)}")
        self.thumbnail_similarity = 1 - pairwise_distances(self.image_features, metric='cosine')

        interaction_data = self.loader.ratings_df.pivot(index="user", columns="item", values="rating").fillna(0).values
        self.cf_similarity = cosine_similarity(interaction_data.T)

    def predict(self, selected_items, filter_out_items, k):
        print(f"Selected items: {selected_items}")
        print(f"Filtered items: {filter_out_items}")
        print(f"Scores form: {np.shape(self.combined_similarity)}")

        neural_scores = np.sum(self.thumbnail_similarity[selected_items], axis=0)
        cf_scores = np.sum(self.cf_similarity[selected_items], axis=0)

        for item in filter_out_items:
            neural_scores[item] = -np.inf
            cf_scores[item] = -np.inf

        print(f"cfscores: {cf_scores}")

        cf_selection = np.argsort(-cf_scores)[:self.cf_select]
        print(f"CF selection: {cf_selection}")

        filtered_neural_scores = neural_scores[cf_selection]
        sorted_k_indices = np.argsort(-filtered_neural_scores)[:k]
        print(f"Neural selection: {sorted_k_indices}")

        recommended_indices = cf_selection[sorted_k_indices[:k]]
        print(f"Recommended indices: {recommended_indices}")

        return recommended_indices

    @classmethod
    def name(cls):
        return "CNFRS"

    @classmethod
    def parameters(cls):
        return [
            Parameter(
                "cf_select",
                "float",
               100,
                "Weight for combining thumbnails and CF similarities"
            )
        ]
