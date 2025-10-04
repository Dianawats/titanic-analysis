# Titanic EDA Complete Analysis Script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("🚀 STARTING TITANIC EDA ANALYSIS...")
print("=" * 60)

# Set up plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Step 1: Load Data
print("📊 LOADING DATASET...")
try:
    df = pd.read_csv('train.csv')
    print(f"Dataset loaded: {df.shape[0]} passengers, {df.shape[1]} features")
except FileNotFoundError:
    print("ERROR: train.csv file not found!")
    print("Make sure train.csv is in the same folder as this script")
    exit()

# Step 2: Data Cleaning
print("\n🛠️ CLEANING DATA...")
# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Deck'] = df['Cabin'].str[0].fillna('Unknown') if 'Cabin' in df.columns else 'Unknown'
if 'Cabin' in df.columns:
    df = df.drop('Cabin', axis=1)

print("✓ Missing values handled")

# Step 3: Feature Engineering
print("\n🎯 CREATING NEW FEATURES...")
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Fix for the escape sequence issue - use raw string
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Age groups (handle if Age column exists)
if 'Age' in df.columns:
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                           labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

# Fare groups (handle if Fare column exists)
if 'Fare' in df.columns:
    df['FareGroup'] = pd.qcut(df['Fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'])

print("✓ New features created")

# Step 4: Summary Statistics
print("\n" + "=" * 60)
print("📈 SUMMARY STATISTICS")
print("=" * 60)

overall_survival = df['Survived'].mean() * 100
print(f"\nOverall Survival Rate: {overall_survival:.2f}%")

print("\nSurvival by Gender:")
gender_survival = df.groupby('Sex')['Survived'].mean() * 100
print(gender_survival.round(2))

print("\nSurvival by Passenger Class:")
class_survival = df.groupby('Pclass')['Survived'].mean() * 100
print(class_survival.round(2))

# Step 5: Create Visualizations
print("\n📊 CREATING VISUALIZATIONS...")

# Create a directory for plots
if not os.path.exists('plots'):
    os.makedirs('plots')

try:
    # Plot 1: Survival by Gender and Class
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Pclass', y='Survived', hue='Sex', palette='viridis')
    plt.title('Survival Rate by Passenger Class and Gender')
    plt.ylabel('Survival Rate')
    plt.savefig('plots/survival_by_class_gender.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: survival_by_class_gender.png")

    # Plot 2: Survival by Age Groups
    if 'AgeGroup' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='AgeGroup', y='Survived', palette='coolwarm')
        plt.title('Survival Rate by Age Group')
        plt.ylabel('Survival Rate')
        plt.savefig('plots/survival_by_age.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: survival_by_age.png")

    # Plot 3: Fare distribution by survival
    if 'Fare' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Survived', y='Fare', palette='Set2')
        plt.title('Fare Distribution by Survival Status')
        plt.xticks([0, 1], ['Died', 'Survived'])
        plt.yscale('log')
        plt.savefig('plots/fare_by_survival.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: fare_by_survival.png")

    # Plot 4: Family Size impact
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='IsAlone', y='Survived', palette='pastel')
    plt.title('Survival Rate: Alone vs With Family')
    plt.xticks([0, 1], ['With Family', 'Alone'])
    plt.ylabel('Survival Rate')
    plt.savefig('plots/survival_by_family.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: survival_by_family.png")

    # Plot 5: Correlation Heatmap
    plt.figure(figsize=(12, 8))
    corr_df = df.copy()
    corr_df['Sex'] = corr_df['Sex'].map({'male': 0, 'female': 1})
    corr_df['Embarked'] = corr_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    numerical_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
    # Only use columns that exist
    numerical_cols = [col for col in numerical_cols if col in corr_df.columns]
    correlation_matrix = corr_df[numerical_cols].corr()

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Heatmap of Titanic Features')
    plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: correlation_heatmap.png")

except Exception as e:
    print(f"⚠️ Could not create some plots: {e}")

# Step 6: Key Insights
print("\n" + "=" * 60)
print("🔑 KEY INSIGHTS")
print("=" * 60)

try:
    insights = {
        'Female Survival': gender_survival['female'],
        'Male Survival': gender_survival['male'],
        '1st Class Survival': class_survival[1],
        '3rd Class Survival': class_survival[3],
        'Children Survival': df[df['Age'] < 18]['Survived'].mean() * 100,
        'Alone Survival': df[df['IsAlone'] == 1]['Survived'].mean() * 100
    }

    print(f"\n🎯 CRITICAL FINDINGS:")
    print(f"• Women had {insights['Female Survival']:.1f}% survival vs Men {insights['Male Survival']:.1f}%")
    print(f"• 1st Class: {insights['1st Class Survival']:.1f}% vs 3rd Class: {insights['3rd Class Survival']:.1f}%")
    print(f"• Children: {insights['Children Survival']:.1f}% survival rate")
    print(f"• Alone passengers: {insights['Alone Survival']:.1f}% survival")

    print(f"\n💡 OBSERVATIONS:")
    print("• 'Women and children first' policy clearly followed")
    print("• Strong wealth-survival correlation (higher class = better survival)")
    print("• Family ties provided survival advantage")

except Exception as e:
    print(f"Could not calculate all insights: {e}")

print(f"\n📁 Analysis complete! Check the 'plots' folder for visualizations.")
print("=" * 60)

# Show final plot
try:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Pclass', y='Survived', hue='Sex', palette='viridis')
    plt.title('SURVIVAL BY CLASS AND GENDER (Final Chart)')
    plt.ylabel('Survival Rate')
    plt.show()
except:
    print("Could not display final chart")