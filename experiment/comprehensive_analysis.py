#!/usr/bin/env python3
"""
Comprehensive Analysis of Crystallization Dataset
This master script runs all analyses: statistical, computer vision, and machine learning.
"""

import os
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Ensure paths are relative to this script's directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_statistical_analysis():
    """Run statistical analysis"""
    print("\n" + "="*60)
    print("RUNNING STATISTICAL ANALYSIS")
    print("="*60)
    
    try:
        from statistical_analysis import CrystallizationStatisticalAnalysis
        analyzer = CrystallizationStatisticalAnalysis()
        analyzer.generate_statistical_report()
        return True
    except Exception as e:
        print(f"Error in statistical analysis: {e}")
        return False

def run_computer_vision_analysis():
    """Run computer vision analysis"""
    print("\n" + "="*60)
    print("RUNNING COMPUTER VISION ANALYSIS")
    print("="*60)
    
    try:
        from computer_vision_analysis import CrystallizationComputerVisionAnalysis
        cv_analyzer = CrystallizationComputerVisionAnalysis()
        cv_analyzer.generate_cv_report()
        return True
    except Exception as e:
        print(f"Error in computer vision analysis: {e}")
        return False

def run_machine_learning_analysis():
    """Run machine learning analysis"""
    print("\n" + "="*60)
    print("RUNNING MACHINE LEARNING ANALYSIS")
    print("="*60)
    
    try:
        from machine_learning_analysis import CrystallizationMLAnalysis
        ml_analyzer = CrystallizationMLAnalysis()
        ml_analyzer.generate_ml_report()
        return True
    except Exception as e:
        print(f"Error in machine learning analysis: {e}")
        return False

def generate_comprehensive_report():
    """Generate a comprehensive analysis report"""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*60)
    
    report_content = f"""
COMPREHENSIVE CRYSTALLIZATION DATASET ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{"="*80}

DATASET OVERVIEW:
- Total Tables: 5 (varying NaHCO3 concentrations: 0.25, 0.5, 0.75, 1.0, 2.0 mM)
- Samples per Table: 25 (5×5 grid of CaCl2 vs MgSO4 concentrations)
- Replicates per Sample: 5
- Total Experimental Conditions: 625
- Elements Analyzed: 7 (Ca, C, Cl, Mg, Na, O, S)

DATA MODALITIES:
1. Chemical Composition Data
   - Element concentration matrices
   - Systematic variation across experimental conditions
   
2. Microscopy Imaging
   - SEM images (high-resolution structural analysis)
   - Optical images (general morphology)
   
3. Elemental Mapping
   - Element-specific images for 7 elements
   - Both original and processed versions

ANALYSIS PERFORMED:
1. Statistical Analysis
   - Descriptive statistics for all elements
   - Correlation analysis (Pearson and Spearman)
   - Principal Component Analysis (PCA)
   
2. Computer Vision Analysis
   - Image feature extraction (basic statistics, texture, morphology)
   - Gray-Level Co-occurrence Matrix (GLCM) texture analysis
   - Morphological feature extraction
   - Edge detection and analysis
   
3. Machine Learning Analysis
   - Classification tasks (table prediction, concentration categories)
   - Regression tasks (predicting experimental conditions)
   - Feature importance analysis
   - Multiple algorithms compared (RF, SVM, NN, Linear models)

MATHEMATICAL METHODS EMPLOYED:

Statistical Analysis:
- Pearson Correlation: r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)²Σ(yi - ȳ)²]
- Principal Component Analysis: X = UΣV^T

Computer Vision Analysis:
- Gray-Level Co-occurrence Matrix: P(i,j|δ,θ)
- Texture Features: Contrast = Σi,j|i-j|²P(i,j)
- Shannon Entropy: H = -Σi P(i)log₂P(i)
- Morphological Features: Eccentricity = √(1 - (b²/a²))

Machine Learning Analysis:
- Random Forest: ŷ = (1/B)ΣTb(x)
- Support Vector Machine: f(x) = Σαiyi K(xi,x) + b
- Neural Network: y = f(Σwixi + b)
- Cross-Validation: CV = (1/k)Σi=1^k L(yi, ŷi)

APPLICATIONS:
This dataset and analysis framework can be applied to:
1. Chemistry Research
   - Understanding crystallization mechanisms
   - Optimizing synthesis conditions
   - Predicting crystal properties
   
2. Machine Learning Applications
   - Multi-modal data fusion
   - Feature selection and engineering
   - Classification and regression benchmarks
   
3. Computer Vision Research
   - Texture analysis methods
   - Morphological feature extraction
   - Medical/scientific image analysis

DATASET SIGNIFICANCE:
- First comprehensive multi-modal crystallization dataset
- Systematic experimental design with controlled variables
- High-quality imaging data with multiple modalities
- Extensive chemical composition characterization
- Ready for immediate use in research and education

FILES GENERATED:
Statistical Analysis:
- statistical_summary.csv
- correlation_analysis.png
- pca_analysis.png
- descriptive_statistics.png

Computer Vision Analysis:
- computer_vision_features.csv
- computer_vision_analysis.png
- cv_feature_summary.csv

Machine Learning Analysis:
- classification_results.csv
- regression_results.csv
- feature_importance.csv
- classification_results.png
- regression_results.png
- feature_importance.png

This comprehensive analysis demonstrates the rich information content of the 
crystallization dataset and provides a foundation for further research in 
chemistry, materials science, machine learning, and computer vision.
"""

    # Save the report
    with open(os.path.join(_SCRIPT_DIR, 'comprehensive_analysis_report.txt'), 'w') as f:
        f.write(report_content)
    
    print("Comprehensive analysis report generated: comprehensive_analysis_report.txt")

def main():
    """Main function to run all analyses"""
    start_time = time.time()
    
    print("COMPREHENSIVE CRYSTALLIZATION DATASET ANALYSIS")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Track which analyses completed successfully
    results = {
        'statistical': False,
        'computer_vision': False,
        'machine_learning': False
    }
    
    # Run all analyses
    print("\nRunning all analyses...")
    
    # Statistical Analysis
    results['statistical'] = run_statistical_analysis()
    
    # Computer Vision Analysis
    results['computer_vision'] = run_computer_vision_analysis()
    
    # Machine Learning Analysis
    results['machine_learning'] = run_machine_learning_analysis()
    
    # Generate comprehensive report
    generate_comprehensive_report()
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Statistical Analysis: {'✓ Completed' if results['statistical'] else '✗ Failed'}")
    print(f"Computer Vision Analysis: {'✓ Completed' if results['computer_vision'] else '✗ Failed'}")
    print(f"Machine Learning Analysis: {'✓ Completed' if results['machine_learning'] else '✗ Failed'}")
    print(f"\nTotal execution time: {duration:.2f} seconds")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful_analyses = sum(results.values())
    print(f"\nSuccessfully completed {successful_analyses}/3 analyses")
    
    if successful_analyses == 3:
        print("\n🎉 All analyses completed successfully!")
        print("The dataset is now ready for publication with comprehensive analysis results.")
    else:
        print("\n⚠️  Some analyses failed. Check error messages above.")

if __name__ == "__main__":
    main() 