"""
Data Preprocessing Module for Resume Screening
This module handles cleaning and preprocessing of resume text data
"""

import re
import string
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load spaCy English model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("Please download spaCy model: python -m spacy download en_core_web_sm")
    nlp = None

class ResumePreprocessor:
    """Class to preprocess resume text data"""
    
    def __init__(self):
        self.nlp = nlp
        self.stop_words = STOP_WORDS
        
    def clean_text(self, text):
        """
        Clean resume text by removing URLs, special characters, extra spaces
        
        Args:
            text (str): Raw resume text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """
        Remove common stopwords from text
        
        Args:
            text (str): Cleaned text
            
        Returns:
            str: Text without stopwords
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text):
        """
        Lemmatize text using spaCy
        
        Args:
            text (str): Input text
            
        Returns:
            str: Lemmatized text
        """
        if self.nlp is None:
            return text
            
        doc = self.nlp(text)
        lemmatized = ' '.join([token.lemma_ for token in doc])
        return lemmatized
    
    def extract_entities(self, text):
        """
        Extract named entities from resume text
        
        Args:
            text (str): Resume text
            
        Returns:
            dict: Dictionary of entities by type
        """
        if self.nlp is None:
            return {}
            
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
            
        return entities
    
    def extract_skills(self, text):
        """
        Extract skills from resume using keyword matching
        
        Args:
            text (str): Resume text
            
        Returns:
            list: List of identified skills
        """
        # Comprehensive technical skills list
        skill_keywords = {
            # Programming Languages
            'programming': ['python', 'java', 'javascript', 'c\\+\\+', 'c#', 'ruby', 'php', 
                          'swift', 'kotlin', 'scala', 'go', 'rust', 'typescript', 'r programming',
                          'perl', 'matlab', 'vb\\.net', 'objective-c'],
            
            # Web Technologies
            'web_frontend': ['html', 'css', 'react', 'reactjs', 'angular', 'angularjs', 'vue', 
                           'vuejs', 'jquery', 'bootstrap', 'sass', 'less', 'webpack', 'redux'],
            
            'web_backend': ['node', 'nodejs', 'express', 'django', 'flask', 'spring boot', 
                          'spring', 'asp\\.net', 'laravel', 'rails', 'fastapi'],
            
            # Databases
            'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'sql server', 
                        'sqlite', 'redis', 'cassandra', 'dynamodb', 'mariadb', 'nosql',
                        'firebase', 'elasticsearch'],
            
            # Data Science & ML
            'data_science': ['machine learning', 'deep learning', 'data science', 'data analysis',
                           'data analytics', 'artificial intelligence', 'nlp', 'computer vision',
                           'neural networks'],
            
            'ml_frameworks': ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'scikit learn',
                            'pandas', 'numpy', 'scipy', 'opencv', 'nltk', 'spacy'],
            
            # Cloud & DevOps
            'cloud': ['aws', 'azure', 'gcp', 'google cloud', 'cloud computing', 'amazon web services',
                     'microsoft azure', 'heroku', 'digitalocean'],
            
            'devops': ['docker', 'kubernetes', 'jenkins', 'ci/cd', 'gitlab', 'github actions',
                      'terraform', 'ansible', 'linux', 'unix', 'bash', 'shell scripting'],
            
            # Mobile Development
            'mobile': ['android', 'ios', 'react native', 'flutter', 'xamarin', 'swift', 
                      'mobile development'],
            
            # Tools & Others
            'tools': ['git', 'github', 'bitbucket', 'jira', 'confluence', 'agile', 'scrum',
                     'kanban', 'excel', 'powerpoint', 'word'],
            
            # Data Visualization
            'visualization': ['tableau', 'power bi', 'powerbi', 'd3\\.js', 'd3', 'matplotlib',
                            'seaborn', 'plotly', 'excel'],
            
            # Big Data
            'big_data': ['hadoop', 'spark', 'apache spark', 'hive', 'kafka', 'airflow',
                        'big data', 'etl'],
            
            # Soft Skills
            'soft_skills': ['leadership', 'communication', 'teamwork', 'problem solving',
                          'analytical', 'project management', 'time management']
        }
        
        # Convert text to lowercase for matching
        text_lower = text.lower()
        found_skills = []
        
        # Search for each skill
        for category, skills in skill_keywords.items():
            for skill in skills:
                # Use regex for better matching (word boundaries)
                import re
                # Escape special regex characters except those we want
                pattern = r'\b' + skill.replace('\\', '') + r'\b'
                
                try:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        # Store the original skill name (not lemmatized)
                        found_skills.append(skill.replace('\\+\\+', '++').replace('\\.', '.'))
                except:
                    # Fallback to simple string matching
                    if skill.replace('\\', '') in text_lower:
                        found_skills.append(skill.replace('\\+\\+', '++').replace('\\.', '.'))
        
        # Remove duplicates and return
        return list(set(found_skills))
    
    def preprocess_resume(self, text, remove_stop=True, lemmatize=True):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Raw resume text
            remove_stop (bool): Whether to remove stopwords
            lemmatize (bool): Whether to lemmatize
            
        Returns:
            str: Fully preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Remove stopwords
        if remove_stop:
            text = self.remove_stopwords(text)
        
        # Lemmatize
        if lemmatize:
            text = self.lemmatize_text(text)
            
        return text
    
    def process_dataframe(self, df, text_column='Resume_str'):
        """
        Process entire dataframe of resumes
        
        Args:
            df (pd.DataFrame): DataFrame with resume data
            text_column (str): Name of column containing resume text
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        df = df.copy()
        
        # Remove rows with missing resume text
        print(f"Original dataset size: {len(df)}")
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].str.strip() != '']
        print(f"After removing empty resumes: {len(df)}")
        
        # Extract skills BEFORE preprocessing (use original text)
        print("Extracting skills from original text...")
        df['skills'] = df[text_column].apply(self.extract_skills)
        df['num_skills'] = df['skills'].apply(len)
        
        # Preprocess resume text (for TF-IDF)
        print("Cleaning resume text...")
        df['cleaned_resume'] = df[text_column].apply(self.preprocess_resume)
        
        # Remove rows where cleaned_resume is empty
        df = df[df['cleaned_resume'].str.strip() != '']
        print(f"After cleaning: {len(df)}")
        
        # Extract entities
        print("Extracting entities...")
        df['entities'] = df[text_column].apply(self.extract_entities)
        
        # Calculate resume length metrics
        df['resume_length'] = df[text_column].apply(lambda x: len(str(x).split()))
        df['cleaned_length'] = df['cleaned_resume'].apply(lambda x: len(str(x).split()))
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df


def load_and_preprocess_data(file_path):
    """
    Load and preprocess resume dataset
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Load data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Categories: {df['Category'].nunique()}")
    
    # Initialize preprocessor
    preprocessor = ResumePreprocessor()
    
    # Process dataframe
    df_processed = preprocessor.process_dataframe(df)
    
    print("Preprocessing completed!")
    return df_processed


if __name__ == "__main__":
    # Test the preprocessor
    sample_text = """
    John Doe
    Email: john.doe@email.com
    Phone: +1-234-567-8900
    
    EXPERIENCE:
    Software Engineer at Tech Corp (2020-2023)
    - Developed web applications using Python, Django, and React
    - Implemented machine learning models using TensorFlow
    - Worked with AWS cloud services
    
    SKILLS:
    Python, JavaScript, Machine Learning, AWS, SQL
    """
    
    preprocessor = ResumePreprocessor()
    cleaned = preprocessor.preprocess_resume(sample_text)
    skills = preprocessor.extract_skills(sample_text)
    
    print("Original Text:")
    print(sample_text[:200])
    print("\nCleaned Text:")
    print(cleaned[:200])
    print("\nExtracted Skills:")
    print(skills)