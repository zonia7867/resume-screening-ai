"""
Feature Extraction Module for Resume Screening
This module handles TF-IDF vectorization and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import pickle
import os

class FeatureExtractor:
    """Class to extract features from resume text using TF-IDF"""
    
    def __init__(self, max_features=1000, ngram_range=(1, 2)):
        """
        Initialize TF-IDF vectorizer
        
        Args:
            max_features (int): Maximum number of features to extract
            ngram_range (tuple): Range of n-grams to consider
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Minimum document frequency
            max_df=0.8,  # Maximum document frequency
            sublinear_tf=True  # Apply sublinear tf scaling
        )
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def fit_transform(self, texts, labels=None):
        """
        Fit vectorizer and transform texts to TF-IDF features
        
        Args:
            texts (list): List of resume texts
            labels (list): List of job categories (optional)
            
        Returns:
            np.array: TF-IDF feature matrix
            np.array: Encoded labels (if labels provided)
        """
        # Clean texts - remove NaN, None, and empty strings
        print("Cleaning input texts...")
        cleaned_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            # Convert to string and check if valid
            if pd.isna(text) or text is None or str(text).strip() == '':
                cleaned_texts.append('')  # Add empty string for invalid texts
            else:
                cleaned_texts.append(str(text))
            
            # Track valid indices
            if str(text).strip() != '':
                valid_indices.append(i)
        
        print(f"Valid texts: {len(valid_indices)}/{len(texts)}")
        
        # Transform texts to TF-IDF features
        print("Fitting TF-IDF vectorizer...")
        tfidf_features = self.vectorizer.fit_transform(cleaned_texts)
        self.is_fitted = True
        
        print(f"TF-IDF feature shape: {tfidf_features.shape}")
        
        # Encode labels if provided
        if labels is not None:
            # Clean labels to match valid texts
            cleaned_labels = [labels[i] if i < len(labels) else 'Unknown' for i in range(len(texts))]
            encoded_labels = self.label_encoder.fit_transform(cleaned_labels)
            return tfidf_features, encoded_labels
        
        return tfidf_features
    
    def transform(self, texts):
        """
        Transform texts to TF-IDF features using fitted vectorizer
        
        Args:
            texts (list): List of resume texts
            
        Returns:
            np.array: TF-IDF feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        # Clean texts - remove NaN, None, and empty strings
        cleaned_texts = []
        for text in texts:
            if pd.isna(text) or text is None or str(text).strip() == '':
                cleaned_texts.append('')
            else:
                cleaned_texts.append(str(text))
            
        return self.vectorizer.transform(cleaned_texts)
    
    def get_feature_names(self):
        """
        Get names of TF-IDF features
        
        Returns:
            list: List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted.")
            
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features(self, text, top_n=10):
        """
        Get top N TF-IDF features for a given text
        
        Args:
            text (str): Input text
            top_n (int): Number of top features to return
            
        Returns:
            list: List of (feature, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted.")
            
        # Transform text
        tfidf_vector = self.vectorizer.transform([text])
        
        # Get feature names and scores
        feature_names = self.get_feature_names()
        scores = tfidf_vector.toarray()[0]
        
        # Get top features
        top_indices = scores.argsort()[-top_n:][::-1]
        top_features = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
        
        return top_features
    
    def calculate_similarity(self, resume_text, job_description):
        """
        Calculate cosine similarity between resume and job description
        
        Args:
            resume_text (str): Resume text
            job_description (str): Job description text
            
        Returns:
            float: Similarity score (0-1)
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted.")
            
        # Transform both texts
        resume_vector = self.vectorizer.transform([resume_text])
        job_vector = self.vectorizer.transform([job_description])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(resume_vector, job_vector)[0][0]
        
        return similarity
    
    def batch_similarity(self, resume_texts, job_description):
        """
        Calculate similarity scores for multiple resumes against a job description
        
        Args:
            resume_texts (list): List of resume texts
            job_description (str): Job description text
            
        Returns:
            np.array: Array of similarity scores
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted.")
            
        # Transform resumes and job description
        resume_vectors = self.vectorizer.transform(resume_texts)
        job_vector = self.vectorizer.transform([job_description])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(resume_vectors, job_vector).flatten()
        
        return similarities
    
    def save_vectorizer(self, filepath):
        """
        Save fitted vectorizer to disk
        
        Args:
            filepath (str): Path to save the vectorizer
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted.")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"Vectorizer saved to {filepath}")
    
    def load_vectorizer(self, filepath):
        """
        Load fitted vectorizer from disk
        
        Args:
            filepath (str): Path to load the vectorizer from
        """
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        self.is_fitted = True
        print(f"Vectorizer loaded from {filepath}")
    
    def save_label_encoder(self, filepath):
        """
        Save fitted label encoder to disk
        
        Args:
            filepath (str): Path to save the label encoder
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Label encoder saved to {filepath}")
    
    def load_label_encoder(self, filepath):
        """
        Load fitted label encoder from disk
        
        Args:
            filepath (str): Path to load the label encoder from
        """
        with open(filepath, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print(f"Label encoder loaded from {filepath}")


class ResumeMatcher:
    """Class to match resumes with job descriptions"""
    
    def __init__(self, feature_extractor):
        """
        Initialize matcher with a feature extractor
        
        Args:
            feature_extractor (FeatureExtractor): Fitted feature extractor
        """
        self.feature_extractor = feature_extractor
        
    def rank_candidates(self, resumes_df, job_description, top_n=10):
        """
        Rank candidates based on match with job description
        
        Args:
            resumes_df (pd.DataFrame): DataFrame containing resumes
            job_description (str): Job description text
            top_n (int): Number of top candidates to return
            
        Returns:
            pd.DataFrame: Top N ranked candidates with similarity scores
        """
        # Calculate similarity scores
        similarities = self.feature_extractor.batch_similarity(
            resumes_df['cleaned_resume'].tolist(),
            job_description
        )
        
        # Add similarity scores to dataframe
        results_df = resumes_df.copy()
        results_df['similarity_score'] = similarities
        results_df['match_percentage'] = (similarities * 100).round(2)
        
        # Sort by similarity score
        results_df = results_df.sort_values('similarity_score', ascending=False)
        
        # Return top N candidates
        return results_df.head(top_n)
    
    def filter_by_category(self, resumes_df, category, job_description=None, top_n=10):
        """
        Filter resumes by category and optionally rank by job description
        
        Args:
            resumes_df (pd.DataFrame): DataFrame containing resumes
            category (str): Job category to filter
            job_description (str): Optional job description for ranking
            top_n (int): Number of top candidates to return
            
        Returns:
            pd.DataFrame: Filtered and ranked candidates
        """
        # Filter by category
        filtered_df = resumes_df[resumes_df['Category'] == category].copy()
        
        if len(filtered_df) == 0:
            print(f"No resumes found for category: {category}")
            return pd.DataFrame()
        
        # If job description provided, rank candidates
        if job_description:
            return self.rank_candidates(filtered_df, job_description, top_n)
        
        return filtered_df.head(top_n)
    
    def get_match_analysis(self, resume_text, job_description):
        """
        Provide detailed match analysis between resume and job description
        
        Args:
            resume_text (str): Resume text
            job_description (str): Job description text
            
        Returns:
            dict: Analysis results
        """
        # Calculate similarity
        similarity = self.feature_extractor.calculate_similarity(resume_text, job_description)
        
        # Get top features from both
        resume_features = self.feature_extractor.get_top_features(resume_text, top_n=15)
        job_features = self.feature_extractor.get_top_features(job_description, top_n=15)
        
        # Find matching keywords
        resume_keywords = set([feat[0] for feat in resume_features])
        job_keywords = set([feat[0] for feat in job_features])
        matching_keywords = resume_keywords.intersection(job_keywords)
        missing_keywords = job_keywords - resume_keywords
        
        return {
            'similarity_score': similarity,
            'match_percentage': round(similarity * 100, 2),
            'matching_keywords': list(matching_keywords),
            'missing_keywords': list(missing_keywords),
            'resume_top_keywords': resume_features[:10],
            'job_top_keywords': job_features[:10]
        }


if __name__ == "__main__":
    # Test the feature extractor
    sample_resumes = [
        "Python developer with experience in Django and Flask. Skills: Machine Learning, AWS, SQL",
        "Java developer experienced in Spring Boot and Microservices. Skills: Kubernetes, Docker, Jenkins",
        "Data Scientist with expertise in Machine Learning and Deep Learning. Python, TensorFlow, PyTorch"
    ]
    
    sample_job = "Looking for Python developer with Machine Learning experience. Must know TensorFlow and AWS."
    
    # Initialize and fit
    extractor = FeatureExtractor(max_features=100)
    tfidf_matrix = extractor.fit_transform(sample_resumes)
    
    print("TF-IDF Matrix shape:", tfidf_matrix.shape)
    
    # Test similarity
    for i, resume in enumerate(sample_resumes):
        similarity = extractor.calculate_similarity(resume, sample_job)
        print(f"\nResume {i+1} similarity: {similarity:.3f}")
        print(f"Top keywords: {extractor.get_top_features(resume, top_n=5)}")