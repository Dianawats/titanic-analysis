# 🚢 Titanic Survival Analysis - Enhanced EDA

![Titanic](https://img.shields.io/badge/Project-Titanic%20Analysis-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A comprehensive exploratory data analysis of the Titanic passenger dataset, revealing critical insights into survival patterns and factors that influenced life-and-death outcomes during the historic disaster.

## 📊 Project Overview

This enhanced analysis provides deep statistical insights into the Titanic disaster, examining how various factors like gender, class, age, and family connections influenced survival rates. The project combines rigorous statistical testing with advanced visualizations to uncover the hidden stories within the data.

## 🎯 Key Findings

### 🏆 Critical Survival Factors

| Factor | Survival Rate | Impact | Status |
|--------|---------------|---------|---------|
| **Female Passengers** | 74.20% | 🟢 **+55.3% advantage** | ✅ Highly Significant |
| **Male Passengers** | 18.89% | 🔴 Significant disadvantage | ✅ Highly Significant |
| **1st Class Passengers** | 62.96% | 🟢 Wealth provided protection | ✅ Highly Significant |
| **3rd Class Passengers** | 24.24% | 🔴 Highest casualty rate | ✅ Highly Significant |
| **Children (<18)** | 54.0% | 🟢 "Children first" policy evident | ⚠️ Borderline Significant |
| **Elderly (>60)** | 22.73% | 🔴 Vulnerable group | ⚠️ Borderline Significant |

### 📈 Statistical Significance Overview

- ✅ **Highly Significant**: Gender, Passenger Class, Fare
- ✅ **Significant**: Embarkation Point
- ⚠️ **Borderline**: Age (p=0.0528)
- ❌ **Not Significant**: Family Size

## ⚡ Quick Start

### Prerequisites
- Python 3.7+
- Required packages: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn

### Installation & Execution
```bash
# Clone the repository
git clone https://github.com/Dianawats/titanic-analysis.git
cd titanic-survival-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn

# Run the comprehensive analysis
python titanic_enhanced_analysis.py

🔍 Analysis Features
🛠️ Data Processing
Dataset: 891 passengers, 12 original features

Data Cleaning: Comprehensive missing value handling
Feature Engineering:
🎯 Age groups (Child, Teen, Adult, Senior)
👑 Passenger titles (Mr, Mrs, Miss, Master, Rare)
👨‍👩‍👧‍👦 Family size categorization
🚢 Deck assignments from cabin numbers
🏷️ Alone vs. with family status

📊 Statistical Analysis
Descriptive Statistics: Survival rates by all major factors
Hypothesis Testing: Chi-square and t-tests for significance
Correlation Analysis: Identifying key predictors
Visual Analytics: Comprehensive plot generation

🎨 Visualization Suite
The analysis generates multiple advanced visualizations including:
📈 Survival rate comparisons across categories
👥 Demographic distributions
🔥 Correlation heatmaps
📋 Statistical significance charts
🎯 Interactive survival probability plots

💡 Major Insights Breakdown
1. Gender Disparity 🚺 vs 🚹
Gender	Survival Rate	Advantage
Women	74.2%	🟢 +55.3% higher
Men	18.9%	🔴 Significant disadvantage
Interpretation: Clear evidence of "women and children first" protocol being followed during evacuation.

2. Class Inequality 💰
Class	Survival Rate	Difference from 1st Class
1st Class	62.96%	Baseline
2nd Class	47.28%	🔻 -15.68%
3rd Class	24.24%	🔻 -38.72%
Interpretation: Strong correlation between wealth/social status and survival chances, with nearly 40% difference between first and third class.

3. Age-Based Patterns 👶 → 👴
Age Group	Survival Rate	Priority Level
Children	57.97%	🟢 High Priority
Teens	42.86%	🟡 Medium Priority
Adults	35.33%	🟡 Medium Priority
Seniors	22.73%	🔴 Low Priority
Interpretation: Age was a significant factor in rescue priorities, with children receiving preferential treatment.

4. Social Connections 👨‍👩‍👧‍👦
Status	Survival Rate	Advantage
With Family	50.56%	🟢 +20.21% higher
Alone	30.35%	Baseline
Interpretation: Family ties provided significant survival advantage, possibly due to coordinated evacuation efforts.

📋 Outputs Generated
📄 Reports
📊 Professional PDF Report: Comprehensive analysis with visualizations

📈 Statistical Summary: Detailed numerical analysis

🔑 Key Insights Document: Executive summary of findings
🖼️ Visualizations
Survival rate comparisons
Demographic analysis charts
Statistical significance plots
Correlation matrices
Advanced EDA plots
