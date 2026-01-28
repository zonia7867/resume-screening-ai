"""
Resume Matcher Module
Advanced resume matching and ranking functionality
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re


class AdvancedResumeMatcher:
    
    
    def __init__(self, feature_extractor, classifier=None):
        
        self.feature_extractor = feature_extractor
        self.classifier = classifier
    
    def calculate_similarity(self, resume_text, job_description):
       
        return self.feature_extractor.calculate_similarity(resume_text, job_description)
    
    def rank_by_tfidf(self, resumes_df, job_description, top_n=10):
       
        similarities = self.feature_extractor.batch_similarity(
            resumes_df['cleaned_resume'].tolist(),
            job_description
        )
        
        results_df = resumes_df.copy()
        results_df['tfidf_score'] = similarities
        results_df['tfidf_percentage'] = (similarities * 100).round(2)
        
        return results_df.sort_values('tfidf_score', ascending=False).head(top_n)
    
    def rank_by_skills(self, resumes_df, required_skills, top_n=10):
       
        required_skills_lower = [skill.lower() for skill in required_skills]
        
        def calculate_skill_score(resume_skills):
            """Calculate skill match percentage"""
            if not resume_skills or not required_skills:
                return 0.0
            
            resume_skills_lower = [skill.lower() for skill in resume_skills]
            matches = len(set(resume_skills_lower) & set(required_skills_lower))
            
            return (matches / len(required_skills)) * 100
        
        results_df = resumes_df.copy()
        results_df['skill_score'] = results_df['skills'].apply(calculate_skill_score)
        results_df['matched_skills'] = results_df['skills'].apply(
            lambda skills: [s for s in skills if s.lower() in required_skills_lower]
        )
        results_df['missing_skills'] = results_df['skills'].apply(
            lambda skills: [s for s in required_skills_lower 
                          if s not in [sk.lower() for sk in skills]]
        )
        
        return results_df.sort_values('skill_score', ascending=False).head(top_n)
    
    def hybrid_ranking(self, resumes_df, job_description, required_skills=None, 
                       tfidf_weight=0.6, skill_weight=0.4, top_n=10):
     
        results_df = resumes_df.copy()
        
        # Calculate TF-IDF similarity
        tfidf_scores = self.feature_extractor.batch_similarity(
            results_df['cleaned_resume'].tolist(),
            job_description
        )
        results_df['tfidf_score'] = tfidf_scores
        
        # Calculate skill scores if required skills provided
        if required_skills:
            required_skills_lower = [skill.lower() for skill in required_skills]
            
            results_df['skill_score'] = results_df['skills'].apply(
                lambda skills: len(set([s.lower() for s in skills]) & 
                                 set(required_skills_lower)) / len(required_skills)
                if skills else 0.0
            )
        else:
            # Extract skills from job description
            job_skills = self._extract_skills_from_text(job_description)
            
            results_df['skill_score'] = results_df['skills'].apply(
                lambda skills: len(set([s.lower() for s in skills]) & 
                                 set([s.lower() for s in job_skills])) / max(len(job_skills), 1)
                if skills else 0.0
            )
        
        # Calculate weighted combined score
        results_df['combined_score'] = (
            tfidf_weight * results_df['tfidf_score'] + 
            skill_weight * results_df['skill_score']
        )
        results_df['match_percentage'] = (results_df['combined_score'] * 100).round(2)
        
        return results_df.sort_values('combined_score', ascending=False).head(top_n)
    
    def _extract_skills_from_text(self, text):
      
        # Common technical skills
        skill_keywords = [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift',
            'html', 'css', 'react', 'angular', 'vue', 'node', 'django', 'flask',
            'sql', 'mysql', 'postgresql', 'mongodb', 'oracle',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes',
            'git', 'agile', 'scrum', 'excel', 'tableau', 'power bi'
        ]
        
        text_lower = text.lower()
        found_skills = [skill for skill in skill_keywords if skill in text_lower]
        
        return found_skills
    
    def filter_by_experience(self, resumes_df, min_years=0, max_years=100):
       
        # This would require extracting years from resume text
        # Simplified implementation
        return resumes_df
    
    def predict_category(self, resume_text):
        
        if self.classifier is None:
            raise ValueError("No classifier provided")
        
        # Transform text
        features = self.feature_extractor.transform([resume_text])
        
        # Predict
        prediction = self.classifier.predict(features)
        
        return prediction[0]
    
    def get_detailed_match_report(self, resume_text, job_description, required_skills=None):
       
        # TF-IDF similarity
        tfidf_similarity = self.calculate_similarity(resume_text, job_description)
        
        # Extract keywords
        resume_keywords = self.feature_extractor.get_top_features(resume_text, top_n=20)
        job_keywords = self.feature_extractor.get_top_features(job_description, top_n=20)
        
        resume_keyword_set = set([k[0] for k in resume_keywords])
        job_keyword_set = set([k[0] for k in job_keywords])
        
        matching_keywords = resume_keyword_set & job_keyword_set
        missing_keywords = job_keyword_set - resume_keyword_set
        
        # Skill analysis
        skill_analysis = None
        if required_skills:
            from src.data_preprocessing import ResumePreprocessor
            preprocessor = ResumePreprocessor()
            
            resume_skills = preprocessor.extract_skills(resume_text)
            resume_skills_lower = [s.lower() for s in resume_skills]
            required_skills_lower = [s.lower() for s in required_skills]
            
            matched_skills = set(resume_skills_lower) & set(required_skills_lower)
            missing_skills = set(required_skills_lower) - set(resume_skills_lower)
            
            skill_analysis = {
                'matched_skills': list(matched_skills),
                'missing_skills': list(missing_skills),
                'skill_match_percentage': (len(matched_skills) / len(required_skills) * 100) 
                                         if required_skills else 0
            }
        
        # Generate recommendation
        if tfidf_similarity >= 0.7:
            recommendation = "STRONG MATCH - Highly Recommended"
        elif tfidf_similarity >= 0.5:
            recommendation = "GOOD MATCH - Recommended for Interview"
        elif tfidf_similarity >= 0.3:
            recommendation = "MODERATE MATCH - Consider for Review"
        else:
            recommendation = "WEAK MATCH - Not Recommended"
        
        report = {
            'tfidf_similarity': tfidf_similarity,
            'match_percentage': round(tfidf_similarity * 100, 2),
            'recommendation': recommendation,
            'matching_keywords': list(matching_keywords),
            'missing_keywords': list(missing_keywords),
            'resume_top_keywords': resume_keywords[:10],
            'job_top_keywords': job_keywords[:10],
            'skill_analysis': skill_analysis
        }
        
        return report
    
    def batch_match_analysis(self, resumes_df, job_description, 
                            required_skills=None, threshold=0.3):
        
        # Calculate similarities
        similarities = self.feature_extractor.batch_similarity(
            resumes_df['cleaned_resume'].tolist(),
            job_description
        )
        
        results_df = resumes_df.copy()
        results_df['similarity'] = similarities
        results_df['match_percentage'] = (similarities * 100).round(2)
        
        # Categorize candidates
        strong_match = results_df[results_df['similarity'] >= 0.7]
        good_match = results_df[(results_df['similarity'] >= 0.5) & 
                                (results_df['similarity'] < 0.7)]
        moderate_match = results_df[(results_df['similarity'] >= threshold) & 
                                    (results_df['similarity'] < 0.5)]
        weak_match = results_df[results_df['similarity'] < threshold]
        
        return {
            'strong_match': strong_match.sort_values('similarity', ascending=False),
            'good_match': good_match.sort_values('similarity', ascending=False),
            'moderate_match': moderate_match.sort_values('similarity', ascending=False),
            'weak_match': weak_match.sort_values('similarity', ascending=False),
            'summary': {
                'total_candidates': len(results_df),
                'strong_match_count': len(strong_match),
                'good_match_count': len(good_match),
                'moderate_match_count': len(moderate_match),
                'weak_match_count': len(weak_match)
            }
        }


class JobDescriptionParser:
    """
    Parse job descriptions to extract requirements
    """
    
    @staticmethod
    def extract_required_skills(job_description):
       
        from src.data_preprocessing import ResumePreprocessor
        preprocessor = ResumePreprocessor()
        
        return preprocessor.extract_skills(job_description)
    
    @staticmethod
    def extract_experience_requirements(job_description):
       
        # Patterns for experience
        patterns = [
            r'(\d+)\+?\s*(?:to\s*(\d+))?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*(?:to\s*(\d+))?\s*yrs?\s*(?:of\s*)?experience',
            r'minimum\s*(\d+)\s*years?',
            r'at least\s*(\d+)\s*years?'
        ]
        
        min_years = 0
        max_years = 100
        
        for pattern in patterns:
            matches = re.findall(pattern, job_description.lower())
            if matches:
                if isinstance(matches[0], tuple):
                    min_years = int(matches[0][0])
                    if matches[0][1]:
                        max_years = int(matches[0][1])
                else:
                    min_years = int(matches[0])
                break
        
        return {
            'min_years': min_years,
            'max_years': max_years
        }
    
    @staticmethod
    def extract_education_requirements(job_description):
       
        education_keywords = [
            'bachelor', 'master', 'phd', 'mba', 'degree',
            'b.tech', 'm.tech', 'b.e.', 'm.e.', 'b.s.', 'm.s.'
        ]
        
        text_lower = job_description.lower()
        found_education = [edu for edu in education_keywords if edu in text_lower]
        
        return found_education


if __name__ == "__main__":
    """
    Example usage
    """
    print("Resume Matcher Module")
    print("="*80)
    print("\nThis module provides advanced resume matching functionality.")
    print("\nFeatures:")
    print("- TF-IDF based similarity matching")
    print("- Skill-based matching")
    print("- Hybrid ranking")
    print("- Detailed match reports")
    print("- Batch candidate analysis")
    print("\nSee documentation for usage examples.")