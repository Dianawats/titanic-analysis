# ENHANCED TITANIC EDA ANALYSIS WITH PDF REPORT (FIXED VERSION)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from fpdf import FPDF
import warnings
warnings.filterwarnings('ignore')

print("🚀 ENHANCED TITANIC EDA ANALYSIS STARTING...")
print("=" * 70)

# Set up plotting style for publication quality
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class TitanicAnalysis:
    def __init__(self):
        self.df = None
        self.insights = {}
        
    def load_and_clean_data(self):
        """Load and preprocess the Titanic dataset"""
        print("📊 LOADING AND CLEANING DATA...")
        
        # Load dataset
        self.df = pd.read_csv('train.csv')
        print(f"✓ Dataset loaded: {self.df.shape[0]} passengers, {self.df.shape[1]} features")
        
        # Advanced data cleaning
        self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())
        self.df['Embarked'] = self.df['Embarked'].fillna(self.df['Embarked'].mode()[0])
        self.df['Deck'] = self.df['Cabin'].str[0].fillna('Unknown')
        
        # Drop unnecessary columns
        self.df = self.df.drop('Cabin', axis=1)
        
        # Advanced feature engineering
        self._create_features()
        print("✓ Data cleaning and feature engineering completed")
    
    def _create_features(self):
        """Create advanced features for deeper analysis"""
        # Basic features
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        self.df['IsAlone'] = (self.df['FamilySize'] == 1).astype(int)
        
        # Age features
        self.df['AgeGroup'] = pd.cut(self.df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                   labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        self.df['IsChild'] = (self.df['Age'] < 18).astype(int)
        self.df['IsElderly'] = (self.df['Age'] > 60).astype(int)
        
        # Fare features
        self.df['FareGroup'] = pd.qcut(self.df['Fare'], 4, 
                                     labels=['Low', 'Medium', 'High', 'Very High'])
        self.df['FarePerPerson'] = self.df['Fare'] / self.df['FamilySize']
        
        # Name features
        self.df['Title'] = self.df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        self.df['Title'] = self.df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                                   'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        self.df['Title'] = self.df['Title'].replace(['Mlle', 'Ms'], 'Miss')
        self.df['Title'] = self.df['Title'].replace('Mme', 'Mrs')
        
        # Social class features
        self.df['WealthIndicator'] = self.df['Fare'] * self.df['Pclass']
        
        # Survival probability features
        self.df['HasFamily'] = ((self.df['SibSp'] > 0) | (self.df['Parch'] > 0)).astype(int)
    
    def comprehensive_statistical_analysis(self):
        """Perform comprehensive statistical analysis"""
        print("\n" + "=" * 70)
        print("📈 COMPREHENSIVE STATISTICAL ANALYSIS")
        print("=" * 70)
        
        # Basic survival rates
        overall_survival = self.df['Survived'].mean() * 100
        self.insights['overall_survival'] = overall_survival
        
        print(f"\nOVERALL SURVIVAL RATE: {overall_survival:.2f}%")
        
        # Detailed group analysis
        groups = ['Sex', 'Pclass', 'Embarked', 'AgeGroup', 'Title', 'IsAlone', 'Deck']
        
        for group in groups:
            if group in self.df.columns:
                survival_rates = self.df.groupby(group)['Survived'].agg(['mean', 'count'])
                survival_rates['mean'] = (survival_rates['mean'] * 100).round(2)
                print(f"\n{group.upper()} Analysis:")
                print(survival_rates.sort_values('mean', ascending=False))
        
        # Statistical tests
        self._perform_statistical_tests()
    
    def _perform_statistical_tests(self):
        """Perform statistical significance tests"""
        print("\nSTATISTICAL SIGNIFICANCE TESTS")
        
        # Chi-square test for categorical variables
        categorical_vars = ['Sex', 'Pclass', 'Embarked']
        for var in categorical_vars:
            contingency_table = pd.crosstab(self.df[var], self.df['Survived'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            print(f"{var} vs Survival: chi2={chi2:.3f}, p-value={p_value:.4f} {significance}")
        
        # T-test for numerical variables
        numerical_vars = ['Age', 'Fare', 'FamilySize']
        for var in numerical_vars:
            survived = self.df[self.df['Survived'] == 1][var]
            not_survived = self.df[self.df['Survived'] == 0][var]
            t_stat, p_value = stats.ttest_ind(survived, not_survived, nan_policy='omit')
            significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            print(f"{var}: t-stat={t_stat:.3f}, p-value={p_value:.4f} {significance}")
    
    def create_advanced_visualizations(self):
        """Create comprehensive advanced visualizations"""
        print("\nCREATING ADVANCED VISUALIZATIONS...")
        
        # Create plots directory
        if not os.path.exists('enhanced_plots'):
            os.makedirs('enhanced_plots')
        
        # 1. Survival Rate Dashboard
        self._create_survival_dashboard()
        
        # 2. Demographic Analysis
        self._create_demographic_analysis()
        
        # 3. Advanced Correlation Analysis
        self._create_advanced_correlation_analysis()
        
        # 4. Predictive Patterns
        self._create_predictive_patterns()
        
        print("✓ All advanced visualizations created")
    
    def _create_survival_dashboard(self):
        """Create a comprehensive survival dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('TITANIC SURVIVAL ANALYSIS DASHBOARD', fontsize=16, fontweight='bold')
        
        # Plot 1: Survival by Class and Gender
        sns.barplot(data=self.df, x='Pclass', y='Survived', hue='Sex', ax=axes[0,0])
        axes[0,0].set_title('Survival by Class & Gender')
        axes[0,0].set_ylabel('Survival Rate')
        
        # Plot 2: Survival by Age Group
        sns.barplot(data=self.df, x='AgeGroup', y='Survived', ax=axes[0,1])
        axes[0,1].set_title('Survival by Age Group')
        axes[0,1].set_ylabel('Survival Rate')
        
        # Plot 3: Survival by Embarkation Port
        sns.barplot(data=self.df, x='Embarked', y='Survived', ax=axes[0,2])
        axes[0,2].set_title('Survival by Embarkation Port')
        axes[0,2].set_ylabel('Survival Rate')
        
        # Plot 4: Fare Distribution by Survival
        sns.boxplot(data=self.df, x='Survived', y='Fare', ax=axes[1,0])
        axes[1,0].set_title('Fare Distribution by Survival')
        axes[1,0].set_xticks([0, 1])
        axes[1,0].set_xticklabels(['Died', 'Survived'])
        axes[1,0].set_yscale('log')
        
        # Plot 5: Family Size Impact
        sns.barplot(data=self.df, x='IsAlone', y='Survived', ax=axes[1,1])
        axes[1,1].set_title('Survival: Alone vs With Family')
        axes[1,1].set_xticks([0, 1])
        axes[1,1].set_xticklabels(['With Family', 'Alone'])
        axes[1,1].set_ylabel('Survival Rate')
        
        # Plot 6: Title-based Survival
        title_survival = self.df.groupby('Title')['Survived'].mean().sort_values(ascending=False)
        sns.barplot(x=title_survival.index, y=title_survival.values * 100, ax=axes[1,2])
        axes[1,2].set_title('Survival Rate by Title')
        axes[1,2].set_ylabel('Survival Rate (%)')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('enhanced_plots/survival_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_demographic_analysis(self):
        """Create demographic analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Age distribution by survival
        sns.histplot(data=self.df, x='Age', hue='Survived', multiple='layer', 
                    ax=axes[0,0], bins=30, alpha=0.7)
        axes[0,0].set_title('Age Distribution by Survival Status')
        axes[0,0].axvline(self.df['Age'].median(), color='red', linestyle='--', 
                         label=f'Median Age: {self.df["Age"].median():.1f}')
        axes[0,0].legend()
        
        # Fare distribution by class and survival
        sns.boxplot(data=self.df, x='Pclass', y='Fare', hue='Survived', ax=axes[0,1])
        axes[0,1].set_title('Fare Distribution by Class and Survival')
        axes[0,1].set_yscale('log')
        
        # Family composition
        family_data = self.df.groupby(['SibSp', 'Parch'])['Survived'].mean().reset_index()
        scatter = axes[1,0].scatter(family_data['SibSp'], family_data['Parch'], 
                                  c=family_data['Survived']*100, cmap='viridis', s=100)
        axes[1,0].set_xlabel('Number of Siblings/Spouses')
        axes[1,0].set_ylabel('Number of Parents/Children')
        axes[1,0].set_title('Family Composition vs Survival Rate')
        plt.colorbar(scatter, ax=axes[1,0], label='Survival Rate (%)')
        
        # Embarkation analysis
        embark_survival = self.df.groupby('Embarked').agg({
            'Survived': 'mean',
            'Fare': 'median',
            'Pclass': 'median'
        })
        sns.heatmap(embark_survival, annot=True, cmap='coolwarm', ax=axes[1,1])
        axes[1,1].set_title('Embarkation Port Characteristics')
        
        plt.tight_layout()
        plt.savefig('enhanced_plots/demographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_advanced_correlation_analysis(self):
        """Create advanced correlation analysis"""
        # Prepare data for correlation
        corr_df = self.df.copy()
        
        # Convert categorical to numerical
        corr_df['Sex'] = corr_df['Sex'].map({'male': 0, 'female': 1})
        corr_df['Embarked'] = corr_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
        corr_df['Title'] = corr_df['Title'].astype('category').cat.codes
        
        # Select features for correlation
        features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                   'FamilySize', 'IsAlone', 'IsChild', 'IsElderly', 'Title']
        
        correlation_matrix = corr_df[features].corr()
        
        # Create advanced correlation plot
        plt.figure(figsize=(14, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                   center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('ADVANCED CORRELATION MATRIX - Titanic Dataset Features', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('enhanced_plots/advanced_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_predictive_patterns(self):
        """Create visualizations showing predictive patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Wealth vs Survival
        sns.scatterplot(data=self.df, x='Fare', y='Age', hue='Survived', 
                       size='Pclass', sizes=(50, 200), alpha=0.7, ax=axes[0,0])
        axes[0,0].set_title('Wealth (Fare) vs Age by Survival')
        axes[0,0].set_xscale('log')
        
        # Family Size vs Age by Survival
        sns.scatterplot(data=self.df, x='Age', y='FamilySize', hue='Survived',
                       style='Sex', s=100, alpha=0.7, ax=axes[0,1])
        axes[0,1].set_title('Family Size vs Age by Survival and Gender')
        
        # Survival probability by multiple factors
        pivot_data = self.df.pivot_table(values='Survived', 
                                       index='Pclass', 
                                       columns='Sex', 
                                       aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', ax=axes[1,0])
        axes[1,0].set_title('Survival Probability by Class and Gender')
        
        # Age distribution by class and survival
        sns.violinplot(data=self.df, x='Pclass', y='Age', hue='Survived', 
                      split=True, inner='quart', ax=axes[1,1])
        axes[1,1].set_title('Age Distribution by Class and Survival')
        
        plt.tight_layout()
        plt.savefig('enhanced_plots/predictive_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE INSIGHTS REPORT")
        print("=" * 70)
        
        # Calculate key metrics
        self._calculate_key_metrics()
        
        print("\nEXECUTIVE SUMMARY:")
        print(f"Overall Survival Rate: {self.insights['overall_survival']:.1f}%")
        print(f"Female Advantage: {self.insights['gender_gap']:.1f}% higher survival for women")
        print(f"Class Disparity: {self.insights['class_gap']:.1f}% difference between 1st and 3rd class")
        print(f"Children Priority: {self.insights['child_survival']:.1f}% of children survived")
        
        print("\nKEY FINDINGS:")
        print("1. GENDER WAS THE STRONGEST PREDICTOR OF SURVIVAL")
        print(f"   Women: {self.insights['female_survival']:.1f}% survival")
        print(f"   Men: {self.insights['male_survival']:.1f}% survival")
        
        print("2. WEALTH AND SOCIAL CLASS DRASTICALLY IMPACTED SURVIVAL")
        print(f"   1st Class: {self.insights['class1_survival']:.1f}% survival")
        print(f"   2nd Class: {self.insights['class2_survival']:.1f}% survival") 
        print(f"   3rd Class: {self.insights['class3_survival']:.1f}% survival")
        
        print("3. FAMILY TIES PROVIDED SURVIVAL ADVANTAGE")
        print(f"   With Family: {self.insights['with_family_survival']:.1f}% survival")
        print(f"   Alone: {self.insights['alone_survival']:.1f}% survival")
        
        print("4. AGE PLAYED A SIGNIFICANT ROLE")
        print(f"   Children (<18): {self.insights['child_survival']:.1f}% survival")
        print(f"   Elderly (>60): {self.insights['elderly_survival']:.1f}% survival")
        
        print("\nRECOMMENDATIONS FOR FURTHER ANALYSIS:")
        print("Build machine learning models to predict survival")
        print("Analyze specific passenger stories and groups")
        print("Compare with other disaster datasets")
        print("Investigate regional and cultural factors")
    
    def _calculate_key_metrics(self):
        """Calculate key metrics for insights"""
        # Gender metrics
        gender_survival = self.df.groupby('Sex')['Survived'].mean() * 100
        self.insights['female_survival'] = gender_survival['female']
        self.insights['male_survival'] = gender_survival['male']
        self.insights['gender_gap'] = gender_survival['female'] - gender_survival['male']
        
        # Class metrics
        class_survival = self.df.groupby('Pclass')['Survived'].mean() * 100
        self.insights['class1_survival'] = class_survival[1]
        self.insights['class2_survival'] = class_survival[2]
        self.insights['class3_survival'] = class_survival[3]
        self.insights['class_gap'] = class_survival[1] - class_survival[3]
        
        # Age metrics
        self.insights['child_survival'] = self.df[self.df['Age'] < 18]['Survived'].mean() * 100
        self.insights['elderly_survival'] = self.df[self.df['Age'] > 60]['Survived'].mean() * 100
        
        # Family metrics
        self.insights['with_family_survival'] = self.df[self.df['IsAlone'] == 0]['Survived'].mean() * 100
        self.insights['alone_survival'] = self.df[self.df['IsAlone'] == 1]['Survived'].mean() * 100
    
    def create_pdf_report(self):
        """Create a professional PDF report without special characters"""
        print("\nGENERATING PROFESSIONAL PDF REPORT...")
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Title Page
        pdf.add_page()
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 40, 'TITANIC DISASTER ANALYSIS', 0, 1, 'C')
        pdf.set_font('Arial', 'I', 16)
        pdf.cell(0, 20, 'Comprehensive Exploratory Data Analysis Report', 0, 1, 'C')
        pdf.cell(0, 20, f'Dataset: {len(self.df)} passengers, {len(self.df.columns)} features', 0, 1, 'C')
        pdf.cell(0, 20, f'Overall Survival Rate: {self.insights["overall_survival"]:.1f}%', 0, 1, 'C')
        
        # Executive Summary
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Executive Summary', 0, 1)
        pdf.set_font('Arial', '', 12)
        summary_text = f"""This comprehensive analysis of the Titanic dataset reveals critical patterns in survival rates based on demographic and socio-economic factors. The overall survival rate was {self.insights["overall_survival"]:.1f}%, with significant disparities based on gender, passenger class, and age.

Key findings include a {self.insights["gender_gap"]:.1f}% survival advantage for women, a {self.insights["class_gap"]:.1f}% difference between first and third class passengers, and a {self.insights["child_survival"]:.1f}% survival rate for children.

The analysis demonstrates clear evidence of the women and children first protocol, while also highlighting the impact of wealth and social status on survival outcomes."""
        pdf.multi_cell(0, 8, summary_text)
        
        # Key Findings
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Key Findings', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        findings = [
            f"Gender Disparity: Women had {self.insights['female_survival']:.1f}% survival vs {self.insights['male_survival']:.1f}% for men",
            f"Class Impact: 1st Class: {self.insights['class1_survival']:.1f}%, 2nd: {self.insights['class2_survival']:.1f}%, 3rd: {self.insights['class3_survival']:.1f}%",
            f"Age Factors: Children: {self.insights['child_survival']:.1f}%, Elderly: {self.insights['elderly_survival']:.1f}%",
            f"Family Effect: With family: {self.insights['with_family_survival']:.1f}%, Alone: {self.insights['alone_survival']:.1f}%"
        ]
        
        for finding in findings:
            pdf.cell(0, 8, f"- {finding}", 0, 1)
        
        # Add images to PDF (only if they exist)
        image_files = [
            'enhanced_plots/survival_dashboard.png',
            'enhanced_plots/demographic_analysis.png',
            'enhanced_plots/advanced_correlation.png',
            'enhanced_plots/predictive_patterns.png'
        ]
        
        for img_file in image_files:
            if os.path.exists(img_file):
                pdf.add_page()
                pdf.set_font('Arial', 'B', 14)
                # Clean title without special characters
                title = os.path.basename(img_file).replace('_', ' ').replace('.png', '').title()
                pdf.cell(0, 10, title, 0, 1)
                try:
                    pdf.image(img_file, x=10, y=30, w=190)
                except:
                    pdf.cell(0, 10, f"Could not load image: {img_file}", 0, 1)
        
        # Save PDF
        try:
            pdf.output('titanic_analysis_report.pdf')
            print("Professional PDF report generated: 'titanic_analysis_report.pdf'")
        except Exception as e:
            print(f"Could not generate PDF: {e}")
            # Create a simple text report instead
            self._create_text_report()

    def _create_text_report(self):
        """Create a text report as fallback"""
        with open('titanic_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("TITANIC DISASTER ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Survival Rate: {self.insights['overall_survival']:.1f}%\n\n")
            
            f.write("KEY FINDINGS:\n")
            f.write(f"- Women: {self.insights['female_survival']:.1f}% survival\n")
            f.write(f"- Men: {self.insights['male_survival']:.1f}% survival\n")
            f.write(f"- 1st Class: {self.insights['class1_survival']:.1f}% survival\n")
            f.write(f"- 3rd Class: {self.insights['class3_survival']:.1f}% survival\n")
            f.write(f"- Children: {self.insights['child_survival']:.1f}% survival\n")
            f.write(f"- With Family: {self.insights['with_family_survival']:.1f}% survival\n")
            f.write(f"- Alone: {self.insights['alone_survival']:.1f}% survival\n")
        
        print("Text report generated: 'titanic_analysis_report.txt'")

def main():
    """Main execution function"""
    # Install required packages if missing
    try:
        from fpdf import FPDF
        from scipy import stats
    except ImportError:
        print("Installing required packages...")
        os.system('pip install fpdf scipy')
        from fpdf import FPDF
        from scipy import stats
    
    # Run enhanced analysis
    analyzer = TitanicAnalysis()
    analyzer.load_and_clean_data()
    analyzer.comprehensive_statistical_analysis()
    analyzer.create_advanced_visualizations()
    analyzer.generate_insights_report()
    analyzer.create_pdf_report()
    
    print("\n" + "=" * 70)
    print("ENHANCED ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("OUTPUTS GENERATED:")
    print("- Enhanced visualizations in 'enhanced_plots/' folder")
    print("- Professional PDF report: 'titanic_analysis_report.pdf'")
    print("- Comprehensive statistical analysis")
    print("- Advanced insights and findings")
    print("=" * 70)

if __name__ == "__main__":
    main()