"""
Fix NaN Values in Processed Data
Run this script to clean your processed_resumes.csv file
"""

import pandas as pd
import os

print("="*80)
print("FIXING NaN VALUES IN PROCESSED DATA")
print("="*80)

# Check if processed file exists
processed_file = 'data/processed/processed_resumes.csv'

if not os.path.exists(processed_file):
    print("\n❌ Processed file not found!")
    print("Please run preprocessing first.")
    exit()

print(f"\n📂 Loading data from: {processed_file}")

# Load data
df = pd.read_csv(processed_file)
print(f"✅ Loaded {len(df)} rows")

# Show initial statistics
print("\n📊 Initial Statistics:")
print(f"   Total rows: {len(df)}")
print(f"   Columns: {list(df.columns)}")

# Check for NaN values
print("\n🔍 Checking for NaN values:")
nan_counts = df.isnull().sum()
for col, count in nan_counts.items():
    if count > 0:
        print(f"   ⚠️  {col}: {count} NaN values")

# Focus on cleaned_resume column
if 'cleaned_resume' in df.columns:
    print("\n🔍 Checking 'cleaned_resume' column:")
    
    # Count different types of invalid values
    nan_count = df['cleaned_resume'].isna().sum()
    empty_count = (df['cleaned_resume'].str.strip() == '').sum()
    
    print(f"   NaN values: {nan_count}")
    print(f"   Empty strings: {empty_count}")
    print(f"   Total invalid: {nan_count + empty_count}")
    
    # Show sample of problematic rows
    if nan_count > 0 or empty_count > 0:
        print("\n📋 Sample of problematic rows:")
        problematic = df[df['cleaned_resume'].isna() | (df['cleaned_resume'].str.strip() == '')]
        print(problematic[['Category', 'cleaned_resume']].head())

# Clean the data
print("\n🧹 Cleaning data...")

original_size = len(df)

# Remove rows with NaN or empty cleaned_resume
if 'cleaned_resume' in df.columns:
    df = df.dropna(subset=['cleaned_resume'])
    df = df[df['cleaned_resume'].str.strip() != '']

# Remove rows with NaN in Category
if 'Category' in df.columns:
    df = df.dropna(subset=['Category'])

# Reset index
df = df.reset_index(drop=True)

cleaned_size = len(df)
removed_count = original_size - cleaned_size

print(f"\n✅ Cleaning complete!")
print(f"   Original size: {original_size}")
print(f"   Cleaned size: {cleaned_size}")
print(f"   Removed: {removed_count} rows")

# Save cleaned data
if removed_count > 0:
    # Backup original file
    backup_file = 'data/processed/processed_resumes_backup.csv'
    print(f"\n💾 Creating backup: {backup_file}")
    
    original_df = pd.read_csv(processed_file)
    original_df.to_csv(backup_file, index=False)
    
    # Save cleaned data
    print(f"💾 Saving cleaned data: {processed_file}")
    df.to_csv(processed_file, index=False)
    
    print("\n✅ Data cleaned and saved!")
    print(f"\n📌 Summary:")
    print(f"   Valid resumes: {len(df)}")
    print(f"   Categories: {df['Category'].nunique()}")
    if 'num_skills' in df.columns:
        print(f"   Avg skills: {df['num_skills'].mean():.1f}")
    if 'resume_length' in df.columns:
        print(f"   Avg length: {df['resume_length'].mean():.0f} words")
else:
    print("\n✅ No cleaning needed! Data is already clean.")

print("\n" + "="*80)
print("🎉 DONE!")
print("="*80)
print("\nNext steps:")
print("1. Run Streamlit: streamlit run app.py")
print("2. Go to Settings page")
print("3. Click 'Train TF-IDF Model'")
print("="*80)