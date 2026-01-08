# Movie Recommendation System
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class MovieRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.user_movie_matrix = None
        self.movie_similarity_matrix = None
        
    def load_sample_data(self):
        """Create sample movie and rating data for demonstration"""
        
        # Sample movies dataset
        movies_data = {
            'movie_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'title': ['The Matrix', 'Titanic', 'Avatar', 'The Dark Knight', 
                     'Inception', 'Pulp Fiction', 'The Godfather', 'Forrest Gump',
                     'The Shawshank Redemption', 'Iron Man', 'Spider-Man', 
                     'The Avengers', 'Interstellar', 'The Lion King', 'Toy Story'],
            'genre': ['Action|Sci-Fi', 'Romance|Drama', 'Action|Adventure|Sci-Fi', 
                     'Action|Crime|Drama', 'Action|Sci-Fi|Thriller', 'Crime|Drama',
                     'Crime|Drama', 'Drama|Romance', 'Drama', 'Action|Adventure|Sci-Fi',
                     'Action|Adventure', 'Action|Adventure|Sci-Fi', 'Drama|Sci-Fi',
                     'Animation|Drama|Family', 'Animation|Comedy|Family'],
            'year': [1999, 1997, 2009, 2008, 2010, 1994, 1972, 1994, 1994, 2008,
                    2002, 2012, 2014, 1994, 1995]
        }
        
        # Sample ratings dataset (user_id, movie_id, rating)
        np.random.seed(42)  # For reproducible results
        ratings_data = []
        
        # Generate realistic ratings for 50 users
        for user_id in range(1, 51):
            # Each user rates 5-12 random movies
            num_ratings = np.random.randint(5, 13)
            movie_ids = np.random.choice(range(1, 16), num_ratings, replace=False)
            
            for movie_id in movie_ids:
                # Generate rating with some preference patterns
                if movie_id in [1, 4, 5, 10, 12]:  # Action/Sci-fi movies
                    rating = np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4])
                elif movie_id in [2, 8, 14, 15]:  # Drama/Family movies
                    rating = np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
                else:
                    rating = np.random.choice([2, 3, 4, 5], p=[0.1, 0.3, 0.4, 0.2])
                
                ratings_data.append([user_id, movie_id, rating])
        
        self.movies_df = pd.DataFrame(movies_data)
        self.ratings_df = pd.DataFrame(ratings_data, columns=['user_id', 'movie_id', 'rating'])
        
        print("✅ Sample data loaded successfully!")
        print(f"Movies: {len(self.movies_df)}")
        print(f"Ratings: {len(self.ratings_df)}")
        print(f"Users: {len(self.ratings_df['user_id'].unique())}")
    
    def explore_data(self):
        """Analyze and visualize the dataset"""
        print("\n" + "="*50)
        print("📊 DATA EXPLORATION")
        print("="*50)
        
        # Basic statistics
        print("\n🎬 Movies Dataset:")
        print(self.movies_df.head())
        
        print("\n⭐ Ratings Dataset:")
        print(self.ratings_df.head())
        
        print(f"\n📈 Rating Statistics:")
        print(self.ratings_df['rating'].describe())
        
        # Visualizations
        plt.figure(figsize=(15, 5))
        
        # Rating distribution
        plt.subplot(1, 3, 1)
        self.ratings_df['rating'].hist(bins=5, edgecolor='black')
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        
        # Top rated movies
        plt.subplot(1, 3, 2)
        top_movies = (self.ratings_df.groupby('movie_id')['rating']
                     .mean().sort_values(ascending=False).head(7))
        movie_names = [self.movies_df[self.movies_df['movie_id']==mid]['title'].iloc[0] 
                      for mid in top_movies.index]
        plt.bar(range(len(top_movies)), top_movies.values)
        plt.title('Top Rated Movies (Avg Rating)')
        plt.xticks(range(len(top_movies)), [name[:10] for name in movie_names], rotation=45)
        plt.ylabel('Average Rating')
        
        # Most rated movies
        plt.subplot(1, 3, 3)
        most_rated = (self.ratings_df.groupby('movie_id').size()
                     .sort_values(ascending=False).head(7))
        movie_names = [self.movies_df[self.movies_df['movie_id']==mid]['title'].iloc[0] 
                      for mid in most_rated.index]
        plt.bar(range(len(most_rated)), most_rated.values)
        plt.title('Most Rated Movies')
        plt.xticks(range(len(most_rated)), [name[:10] for name in movie_names], rotation=45)
        plt.ylabel('Number of Ratings')
        
        plt.tight_layout()
        plt.show()
    
    def create_user_movie_matrix(self):
        """Create user-item matrix for collaborative filtering"""
        self.user_movie_matrix = self.ratings_df.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating'
        ).fillna(0)
        
        print(f"\n📋 User-Movie Matrix created: {self.user_movie_matrix.shape}")
        print("Sample of the matrix:")
        print(self.user_movie_matrix.head())
    
    def collaborative_filtering_recommendations(self, user_id, num_recommendations=5):
        """Generate recommendations using collaborative filtering"""
        
        if self.user_movie_matrix is None:
            self.create_user_movie_matrix()
        
        if user_id not in self.user_movie_matrix.index:
            return f"User {user_id} not found in the dataset"
        
        # Calculate user similarity using cosine similarity
        user_similarity = cosine_similarity(self.user_movie_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index
        )
        
        # Get similar users (excluding the user themselves)
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]
        
        # Get movies rated by similar users that target user hasn't rated
        user_movies = self.user_movie_matrix.loc[user_id]
        unrated_movies = user_movies[user_movies == 0].index
        
        # Calculate weighted ratings for unrated movies
        recommendations = {}
        
        for movie_id in unrated_movies:
            weighted_rating = 0
            similarity_sum = 0
            
            for similar_user_id, similarity_score in similar_users.items():
                if self.user_movie_matrix.loc[similar_user_id, movie_id] > 0:
                    weighted_rating += (similarity_score * 
                                      self.user_movie_matrix.loc[similar_user_id, movie_id])
                    similarity_sum += similarity_score
            
            if similarity_sum > 0:
                recommendations[movie_id] = weighted_rating / similarity_sum
        
        # Sort and get top recommendations
        top_recommendations = sorted(recommendations.items(), 
                                   key=lambda x: x[1], reverse=True)[:num_recommendations]
        
        return top_recommendations
    
    def content_based_recommendations(self, movie_id, num_recommendations=5):
        """Generate recommendations using content-based filtering"""
        
        # Create feature matrix based on genres
        self.movies_df['genre_features'] = self.movies_df['genre'].str.replace('|', ' ')
        
        # Use TF-IDF to create feature vectors
        tfidf = TfidfVectorizer(stop_words='english')
        genre_matrix = tfidf.fit_transform(self.movies_df['genre_features'])
        
        # Calculate movie similarity
        movie_similarity = cosine_similarity(genre_matrix)
        
        # Get movie index
        movie_idx = self.movies_df[self.movies_df['movie_id'] == movie_id].index[0]
        
        # Get similarity scores
        similarity_scores = list(enumerate(movie_similarity[movie_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top similar movies (excluding the input movie)
        top_similar = similarity_scores[1:num_recommendations+1]
        
        recommendations = []
        for idx, score in top_similar:
            movie_info = self.movies_df.iloc[idx]
            recommendations.append((movie_info['movie_id'], movie_info['title'], score))
        
        return recommendations
    
    def display_recommendations(self, user_id=None, movie_id=None):
        """Display recommendations in a user-friendly format"""
        
        print("\n" + "="*50)
        print("🎯 MOVIE RECOMMENDATIONS")
        print("="*50)
        
        if user_id:
            print(f"\n👤 Collaborative Filtering Recommendations for User {user_id}:")
            print("-" * 40)
            
            recs = self.collaborative_filtering_recommendations(user_id)
            if isinstance(recs, str):
                print(recs)
            else:
                for i, (movie_id, predicted_rating) in enumerate(recs, 1):
                    movie_title = self.movies_df[self.movies_df['movie_id']==movie_id]['title'].iloc[0]
                    movie_genre = self.movies_df[self.movies_df['movie_id']==movie_id]['genre'].iloc[0]
                    print(f"{i}. {movie_title}")
                    print(f"   Genre: {movie_genre}")
                    print(f"   Predicted Rating: {predicted_rating:.2f}")
                    print()
        
        if movie_id:
            movie_title = self.movies_df[self.movies_df['movie_id']==movie_id]['title'].iloc[0]
            print(f"\n🎬 Content-Based Recommendations for '{movie_title}':")
            print("-" * 40)
            
            recs = self.content_based_recommendations(movie_id)
            for i, (rec_movie_id, title, similarity) in enumerate(recs, 1):
                genre = self.movies_df[self.movies_df['movie_id']==rec_movie_id]['genre'].iloc[0]
                year = self.movies_df[self.movies_df['movie_id']==rec_movie_id]['year'].iloc[0]
                print(f"{i}. {title} ({year})")
                print(f"   Genre: {genre}")
                print(f"   Similarity Score: {similarity:.3f}")
                print()
    
    def get_movie_list(self):
        """Display available movies for reference"""
        print("\n📚 Available Movies:")
        print("-" * 30)
        for _, movie in self.movies_df.iterrows():
            print(f"ID: {movie['movie_id']} - {movie['title']} ({movie['year']})")

# MAIN EXECUTION
def main():
    print("🎬 Welcome to the Movie Recommendation System!")
    print("=" * 50)
    
    # Initialize the system
    recommender = MovieRecommendationSystem()
    
    # Load and explore data
    recommender.load_sample_data()
    recommender.explore_data()
    
    # Create user-movie matrix
    recommender.create_user_movie_matrix()
    
    # Example recommendations
    print("\n🔍 Example Recommendations:")
    
    # Collaborative filtering example
    recommender.display_recommendations(user_id=5)
    
    # Content-based filtering example
    recommender.display_recommendations(movie_id=1)  # The Matrix
    
    # Show available movies
    recommender.get_movie_list()
    
    print("\n✨ Project completed successfully!")
    print("You can now:")
    print("- Try different user IDs (1-50)")
    print("- Try different movie IDs (1-15)")
    print("- Modify the code to add more movies")
    print("- Experiment with different similarity metrics")

if __name__ == "__main__":

    main()
