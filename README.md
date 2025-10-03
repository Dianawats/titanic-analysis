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

## 🗂️ Project Structure
titanic-analysis/
├── 📊 titanic_enhanced_analysis.py # Main analysis script
├── 📁 enhanced_plots/ # Generated visualizations
├── 📄 titanic_analysis_report.pdf # Professional PDF report
├── 🔑 KEY INSIGHTS.txt # Summary of findings
└── 📖 README.md # This file

## ⚡ Quick Start

### Prerequisites
- Python 3.7+
- Required packages: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn

### Installation & Execution

```bash
# Clone the repository
git clone https://github.com/yourusername/titanic-survival-analysis.git
cd titanic-survival-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn

# Run the comprehensive analysis
python titanic_enhanced_analysis.py
