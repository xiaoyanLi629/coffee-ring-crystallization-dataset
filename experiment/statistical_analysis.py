#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis of Crystallization Dataset
This script performs detailed statistical analysis of the crystallization experiment data
including chemical composition analysis, descriptive statistics, correlation analysis,
and multivariate analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Ensure paths are relative to this script's directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class CrystallizationStatisticalAnalysis:
    def __init__(self, data_path="data/"):
        self.data_path = data_path
        self.elements = ['Calcium', 'Carbon', 'Chlorine', 'Magnesium', 'Oxygen', 'Sodium', 'Sulphur']
        self.concentration_data = {}
        self.experimental_design = {}
        
    def load_data(self):
        """Load all concentration data and experimental design"""
        print("Loading experimental data...")
        
        # Load element concentration data
        for element in self.elements:
            try:
                self.concentration_data[element] = pd.read_csv(
                    os.path.join(_SCRIPT_DIR, "recipies", f"{element}.csv"), header=None
                )
                print(f"Loaded {element} data: {self.concentration_data[element].shape}")
            except Exception as e:
                print(f"Error loading {element}: {e}")
        
        # Create experimental design matrix
        self.create_experimental_design()
        
    def create_experimental_design(self):
        """Create experimental design matrix based on recipe.csv"""
        # Define concentration levels based on recipe.csv analysis
        nahco3_concentrations = [0.25, 0.5, 0.75, 1.0, 2.0]  # mM for each table
        cacl2_concentrations = [0.5, 0.75, 1.0, 1.5, 2.0]  # mM
        mgso4_concentrations = [0.25, 0.5, 0.75, 1.0, 2.0]  # mM
        
        design_data = []
        sample_id = 0
        
        for table_idx, nahco3 in enumerate(nahco3_concentrations):
            for mgso4_idx, mgso4 in enumerate(mgso4_concentrations):
                for cacl2_idx, cacl2 in enumerate(cacl2_concentrations):
                    sample_id += 1
                    design_data.append({
                        'Table': table_idx + 1,
                        'Sample': sample_id,
                        'NaHCO3_mM': nahco3,
                        'CaCl2_mM': cacl2,
                        'MgSO4_mM': mgso4,
                        'Row': mgso4_idx,
                        'Col': cacl2_idx
                    })
        
        self.experimental_design = pd.DataFrame(design_data)
        print(f"Created experimental design: {self.experimental_design.shape}")
        
    def calculate_descriptive_statistics(self):
        """Calculate comprehensive descriptive statistics"""
        print("\nCalculating descriptive statistics...")
        
        stats_results = {}
        
        for element in self.elements:
            data = self.concentration_data[element].values.flatten()
            stats_results[element] = {
                'mean': np.mean(data),
                'median': np.median(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'q25': np.percentile(data, 25),
                'q75': np.percentile(data, 75),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'cv': np.std(data) / np.mean(data) if np.mean(data) != 0 else 0  # Coefficient of variation
            }
        
        # Create summary dataframe
        stats_df = pd.DataFrame(stats_results).T
        stats_df.to_csv(os.path.join(_SCRIPT_DIR, 'statistical_summary.csv'))
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Distribution plots
        element_data = [self.concentration_data[elem].values.flatten() for elem in self.elements]
        axes[0, 0].boxplot(element_data, labels=self.elements)
        axes[0, 0].set_title('Element Concentration Distributions')
        axes[0, 0].set_ylabel('Concentration (mM)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Violin plots
        data_for_violin = []
        elements_for_violin = []
        for elem in self.elements:
            data_for_violin.extend(self.concentration_data[elem].values.flatten())
            elements_for_violin.extend([elem] * self.concentration_data[elem].size)
        
        violin_df = pd.DataFrame({'Element': elements_for_violin, 'Concentration': data_for_violin})
        sns.violinplot(data=violin_df, x='Element', y='Concentration', ax=axes[0, 1])
        axes[0, 1].set_title('Element Concentration Density Distributions')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Coefficient of variation
        cv_values = [stats_results[elem]['cv'] for elem in self.elements]
        axes[1, 0].bar(self.elements, cv_values)
        axes[1, 0].set_title('Coefficient of Variation by Element')
        axes[1, 0].set_ylabel('CV (σ/μ)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Skewness and Kurtosis
        skew_values = [stats_results[elem]['skewness'] for elem in self.elements]
        kurt_values = [stats_results[elem]['kurtosis'] for elem in self.elements]
        
        x_pos = np.arange(len(self.elements))
        width = 0.35
        axes[1, 1].bar(x_pos - width/2, skew_values, width, label='Skewness', alpha=0.8)
        axes[1, 1].bar(x_pos + width/2, kurt_values, width, label='Kurtosis', alpha=0.8)
        axes[1, 1].set_title('Distribution Shape Parameters')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(self.elements, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(_SCRIPT_DIR, 'descriptive_statistics.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return stats_df
    
    def correlation_analysis(self):
        """Perform comprehensive correlation analysis"""
        print("\nPerforming correlation analysis...")
        
        # Prepare data matrix
        data_matrix = []
        for element in self.elements:
            data_matrix.append(self.concentration_data[element].values.flatten())
        
        data_matrix = np.array(data_matrix).T
        correlation_df = pd.DataFrame(data_matrix, columns=self.elements)
        
        # Calculate different correlation measures
        pearson_corr = correlation_df.corr(method='pearson')
        spearman_corr = correlation_df.corr(method='spearman')
        
        # Create correlation heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Pearson correlation
        mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
        sns.heatmap(pearson_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=axes[0])
        axes[0].set_title('Pearson Correlation Matrix')
        
        # Spearman correlation
        mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
        sns.heatmap(spearman_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=axes[1])
        axes[1].set_title('Spearman Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(os.path.join(_SCRIPT_DIR, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save correlation matrices
        pearson_corr.to_csv(os.path.join(_SCRIPT_DIR, 'pearson_correlations.csv'))
        spearman_corr.to_csv(os.path.join(_SCRIPT_DIR, 'spearman_correlations.csv'))
        
        return pearson_corr, spearman_corr
    
    def principal_component_analysis(self):
        """Perform Principal Component Analysis"""
        print("\nPerforming Principal Component Analysis...")
        
        # Prepare data
        data_matrix = []
        for element in self.elements:
            data_matrix.append(self.concentration_data[element].values.flatten())
        
        data_matrix = np.array(data_matrix).T
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_matrix)
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(data_scaled)
        
        # Calculate cumulative explained variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        
        # Create PCA plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scree plot
        axes[0, 0].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                       pca.explained_variance_ratio_, 'bo-')
        axes[0, 0].set_title('Scree Plot')
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        
        # Cumulative explained variance
        axes[0, 1].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'ro-')
        axes[0, 1].axhline(y=0.8, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(y=0.95, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Cumulative Explained Variance')
        axes[0, 1].set_xlabel('Number of Components')
        axes[0, 1].set_ylabel('Cumulative Explained Variance')
        
        # PC1 vs PC2 scatter plot
        scatter = axes[1, 0].scatter(pca_result[:, 0], pca_result[:, 1], 
                                   c=range(len(pca_result)), cmap='viridis', alpha=0.6)
        axes[1, 0].set_title('PC1 vs PC2')
        axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # Component loadings heatmap
        loadings = pca.components_[:4, :]  # First 4 components
        loadings_df = pd.DataFrame(loadings, 
                                 columns=self.elements,
                                 index=[f'PC{i+1}' for i in range(4)])
        sns.heatmap(loadings_df, annot=True, cmap='RdBu_r', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Component Loadings')
        
        plt.tight_layout()
        plt.savefig(os.path.join(_SCRIPT_DIR, 'pca_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save PCA results
        pca_results_df = pd.DataFrame(pca_result, 
                                    columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
        pca_results_df.to_csv(os.path.join(_SCRIPT_DIR, 'pca_results.csv'), index=False)
        loadings_df.to_csv(os.path.join(_SCRIPT_DIR, 'pca_loadings.csv'))
        
        return pca, pca_result, loadings_df
    
    def generate_statistical_report(self):
        """Generate comprehensive statistical analysis report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Perform all analyses
        desc_stats = self.calculate_descriptive_statistics()
        pearson_corr, spearman_corr = self.correlation_analysis()
        pca, pca_result, loadings = self.principal_component_analysis()
        
        print("\nStatistical analysis complete!")
        print("Generated files:")
        print("- statistical_summary.csv")
        print("- correlation_analysis.png")
        print("- pca_analysis.png")

if __name__ == "__main__":
    # Run statistical analysis
    analyzer = CrystallizationStatisticalAnalysis()
    analyzer.generate_statistical_report() 