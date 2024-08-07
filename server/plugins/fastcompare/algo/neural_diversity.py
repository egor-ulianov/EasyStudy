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


class NeuralDiversityRecommender(AlgorithmBase, ABC):

    def __init__(self, loader, alpha=0.5, beta=0.3, gamma=0.2, **kwargs):
        self.loader = loader
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.image_features = None
        self.thumbnail_similarity = None
        self.cf_similarity = None
        self.genre_correlation = None
        self.combined_similarity = None
        print(self.loader.items_df['item_id'])
        print(self.loader.items_df.loc[self.loader.items_df['item_id'] == 8])
        self.model = self._initialize_cnn_model()
        self._all_items = self.loader.ratings_df['item'].unique()
        print(len(self._all_items))
        print(np.min(self._all_items))
        print(np.max(self._all_items))
        self.genres = self._extract_all_genres()

    def _initialize_cnn_model(self):
        feature_layer_model = load_model('/app/plugins/fastcompare/static/feature_extraction_films_current.keras')
        return feature_layer_model

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

        self._compute_genre_correlations()

        self.combined_similarity = (
                self.alpha * self.thumbnail_similarity +
                self.beta * self.cf_similarity +
                self.gamma * self._compute_combined_genre_similarity()
        )

    def _compute_combined_genre_similarity(self):
        genre_sim_matrix = np.zeros((len(self._all_items), len(self._all_items)))
        for i, item_i in enumerate(self._all_items):
            genres_i = self.loader.items_df.loc[item_i, 'description'].split(' ')[-1].split('|')
            for j, item_j in enumerate(self._all_items):
                if i == j:
                    continue
                genres_j = self.loader.items_df.loc[item_j, 'description'].split(' ')[-1].split('|')
                score = np.mean([self.genre_correlation.loc[g_i, g_j] for g_i in genres_i for g_j in genres_j])
                genre_sim_matrix[item_i, item_j] = score
        return genre_sim_matrix

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
        return "Combined Neural Diversity Recommender System"

    @classmethod
    def parameters(cls):
        return [
            Parameter(
                "alpha",
                "float",
                0.5,
                "Weight for combining thumbnail neural similarity"
            ),
            Parameter(
                "beta",
                "float",
                0.3,
                "Weight for combining CF similarity"
            ),
            Parameter(
                "gamma",
                "float",
                0.2,
                "Weight for combining genre correlation"
            )
        ]
