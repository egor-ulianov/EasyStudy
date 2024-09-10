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


class NeuralDiversityFlippedRecommender(AlgorithmBase, ABC):

    def __init__(self, loader, genre_select=100, cf_select=30, genres_diversity_shift=0.1, **kwargs):
        self.genres_similarity = None
        self.loader = loader
        self.genre_select = genre_select
        self.cf_select = cf_select
        self.genres_diversity_shift = genres_diversity_shift
        self.image_features = None
        self.thumbnail_similarity = None
        self.cf_similarity = None
        self.combined_similarity = None
        self.model = self._initialize_cnn_model_v2()
        self._all_items = self.loader.ratings_df['item'].unique()
        self.genres = self._extract_all_genres()

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

    def _extract_all_genres(self):
        all_genres = set()
        self.loader.items_df['description'].apply(lambda x: all_genres.update(x.split(' ')[-1].split('|')))
        return list(all_genres)

    def _compute_genre_correlations(self):
        genre_list = self.loader.items_df['description'].apply(lambda x: x.split(' ')[-1].split('|'))
        genre_combinations = [tuple(sorted(g)) for g in genre_list]

        genre_counts = {}
        for genres in genre_combinations:
            for i in range(len(genres)):
                for j in range(i + 1, len(genres)):
                    pair = (genres[i], genres[j])
                    if pair not in genre_counts:
                        genre_counts[pair] = 0
                    genre_counts[pair] += 1

        self.genre_correlation = pd.DataFrame(0, index=self.genres, columns=self.genres)
        for (g1, g2), count in genre_counts.items():
            self.genre_correlation.loc[g1, g2] = count
            self.genre_correlation.loc[g2, g1] = count

        self.genre_correlation = self.genre_correlation.div(self.genre_correlation.sum(axis=1), axis=0)

    def fit(self):
        print(f"Number of all items : {len(self._all_items)}")
        image_paths = [self.loader.get_item_index_image_url(item_index) for item_index in self._all_items]
        self.image_features = np.array([self._extract_features_v2(img_path) for img_path in image_paths])
        print(f"Shape of image features: {np.shape(self.image_features)}")
        self.thumbnail_similarity = 1 - pairwise_distances(self.image_features, metric='cosine')

        interaction_data = self.loader.ratings_df.pivot(index="user", columns="item", values="rating").fillna(0).values
        self.cf_similarity = cosine_similarity(interaction_data.T)

        self._compute_genre_correlations()
        self.genres_similarity = self._compute_combined_genre_similarity()

    def _compute_combined_genre_similarity(self):
        genre_sim_matrix = np.zeros((len(self._all_items), len(self._all_items)))
        for i, item_i in enumerate(self._all_items):
            genres_i = self.loader.items_df.loc[item_i, 'description'].split(' ')[-1].split('|')
            for j, item_j in enumerate(self._all_items):
                if i == j:
                    continue
                genres_j = self.loader.items_df.loc[item_j, 'description'].split(' ')[-1].split('|')
                score = np.mean([self.genre_correlation.loc[g_i, g_j] for g_i in genres_i for g_j in genres_j])
                genre_sim_matrix[item_i, item_j] = 1 - score
        return genre_sim_matrix

    def predict(self, selected_items, filter_out_items, k):
        print(f"Selected items: {selected_items}")
        print(f"Filtered items: {filter_out_items}")
        print(f"Scores form: {np.shape(self.combined_similarity)}")

        neural_scores = np.sum(self.thumbnail_similarity[selected_items], axis=0)
        cf_scores = np.sum(self.cf_similarity[selected_items], axis=0)
        genre_scores = np.sum(self.genres_similarity[selected_items], axis=0)

        print(f"neural_scores: {neural_scores}")

        for item in filter_out_items:
            neural_scores[item] = -np.inf
            cf_scores[item] = -np.inf
            genre_scores[item] = -np.inf

        genres_selection = np.argsort(genre_scores)[int(len(genre_scores) * self.genres_diversity_shift):int(
            len(genre_scores) * self.genres_diversity_shift) + self.genre_select]

        filtered_cf_scores = cf_scores[genres_selection]
        cf_selection = genres_selection[np.argsort(-filtered_cf_scores)[:self.cf_select]]
        print(f"CF selection: {cf_selection}")

        filtered_neural_scores = neural_scores[cf_selection]
        neural_selection = cf_selection[np.argsort(-filtered_neural_scores)[:k]]

        recommended_indices = neural_selection
        print(f"Recommended indices: {recommended_indices}")

        return recommended_indices

    @classmethod
    def name(cls):
        return "CNDFRS"

    @classmethod
    def parameters(cls):
        return [
            Parameter(
                "genre_select",
                "float",
                100,
                "Weight top genres similarity select"
            ),
            Parameter(
                "cf_select",
                "float",
                30,
                "Weight for combining CF and genre similarities"
            ),
            Parameter(
                "genres_diversity_shift",
                "float",
                0.1,
                "Shift for genre diversity, from 0 to 0.5"
            )
        ]
