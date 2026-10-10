#!/usr/bin/env python3
"""
Comprehensive Computer Vision Analysis of Crystallization Dataset
This script performs detailed computer vision analysis including image feature extraction,
morphological analysis, texture analysis, and machine learning classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from pathlib import Path
from scipy import ndimage
from skimage import feature, measure, morphology, filters, segmentation
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Ensure paths are relative to this script's directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class CrystallizationComputerVisionAnalysis:
    def __init__(self, data_path=None):
        self.data_path = data_path if data_path else os.path.join(_SCRIPT_DIR, "data")
        self.tables = [f"Table_{i}" for i in range(1, 6)]
        self.samples = [f"Sample_{i}" for i in range(1, 26)]
        self.replicates = [f"{i}_rep" for i in range(1, 6)]
        self.elements = ['C', 'Ca', 'Cl', 'Mg', 'Na', 'O', 'S']
        
        self.image_features = {}
        self.morphological_features = {}
        self.texture_features = {}
        
    def calculate_entropy(self, image):
        """Calculate Shannon entropy of an image"""
        # Calculate histogram
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        # Normalize histogram to get probabilities
        hist = hist / hist.sum()
        # Remove zero probabilities
        hist = hist[hist > 0]
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
        
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess an image"""
        try:
            if image_path.endswith('.jpg') or image_path.endswith('.jpeg'):
                img = cv2.imread(image_path)
                if img is not None:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Convert to grayscale for analysis
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    return img, gray
            return None, None
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None
    
    def extract_basic_image_features(self, image):
        """Extract basic statistical features from an image"""
        if image is None:
            return {}
        
        features = {
            'mean_intensity': np.mean(image),
            'std_intensity': np.std(image),
            'min_intensity': np.min(image),
            'max_intensity': np.max(image),
            'median_intensity': np.median(image),
            'q25_intensity': np.percentile(image, 25),
            'q75_intensity': np.percentile(image, 75),
            'skewness': float(np.nan_to_num(np.mean(((image - np.mean(image)) / np.std(image))**3))),
            'kurtosis': float(np.nan_to_num(np.mean(((image - np.mean(image)) / np.std(image))**4) - 3)),  # excess kurtosis
            'entropy': float(np.nan_to_num(self.calculate_entropy(image))),
            'contrast': np.max(image) - np.min(image)
        }
        
        return features
    
    def extract_texture_features(self, image):
        """Extract texture features using GLCM"""
        if image is None:
            return {}
        
        # Normalize image to 8-bit
        value_range = image.max() - image.min()
        if value_range == 0:
            image_8bit = np.zeros_like(image, dtype=np.uint8)
        else:
            image_8bit = ((image - image.min()) / value_range * 255).astype(np.uint8)
        
        # Calculate GLCM
        distances = [1, 2]
        angles = [0, 45, 90, 135]
        
        texture_features = {}
        
        for distance in distances:
            for angle in angles:
                glcm = graycomatrix(image_8bit, [distance], [np.radians(angle)], 
                                  levels=256, symmetric=True, normed=True)
                
                # Extract texture properties
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                
                texture_features.update({
                    f'contrast_d{distance}_a{angle}': float(np.nan_to_num(contrast)),
                    f'homogeneity_d{distance}_a{angle}': float(np.nan_to_num(homogeneity)),
                    f'energy_d{distance}_a{angle}': float(np.nan_to_num(energy))
                })
        
        return texture_features
    
    def extract_morphological_features(self, image):
        """Extract morphological features from segmented image"""
        if image is None:
            return {}
        
        # Apply threshold to segment the image
        threshold = filters.threshold_otsu(image)
        binary = image > threshold
        
        # Remove small objects
        cleaned = morphology.remove_small_objects(binary, min_size=50)
        
        # Label connected components
        labeled = measure.label(cleaned)
        regions = measure.regionprops(labeled)
        
        if len(regions) == 0:
            return {
                'num_objects': 0,
                'total_area': 0,
                'mean_area': 0,
                'mean_perimeter': 0,
                'mean_eccentricity': 0,
                'area_coverage': 0
            }
        
        # Extract features from regions
        areas = [region.area for region in regions]
        perimeters = [region.perimeter for region in regions]
        eccentricities = [region.eccentricity for region in regions]
        
        morphological_features = {
            'num_objects': len(regions),
            'total_area': sum(areas),
            'mean_area': np.mean(areas),
            'mean_perimeter': np.mean(perimeters),
            'mean_eccentricity': np.mean(eccentricities),
            'area_coverage': sum(areas) / (image.shape[0] * image.shape[1])
        }
        
        return morphological_features
    
    def extract_edge_features(self, image):
        """Extract edge-based features"""
        if image is None:
            return {}
        
        # Apply edge detection
        edges_canny = feature.canny(image)
        edges_sobel = filters.sobel(image)
        
        edge_features = {
            'edge_density_canny': np.sum(edges_canny) / edges_canny.size,
            'edge_mean_sobel': np.mean(edges_sobel),
            'edge_std_sobel': np.std(edges_sobel),
            'edge_max_sobel': np.max(edges_sobel)
        }
        
        return edge_features
    
    def analyze_single_sample(self, table, sample, replicate):
        """Analyze all images for a single sample"""
        sample_path = Path(self.data_path) / table / sample / replicate
        
        if not sample_path.exists():
            print(f"Path not found: {sample_path}")
            return {}
        
        sample_features = {
            'table': table,
            'sample': sample,
            'replicate': replicate
        }
        
        # Analyze SEM image
        sem_path = sample_path / "sem.jpg"
        if sem_path.exists():
            sem_img, sem_gray = self.load_and_preprocess_image(str(sem_path))
            if sem_gray is not None:
                sem_features = self.extract_basic_image_features(sem_gray)
                sem_texture = self.extract_texture_features(sem_gray)
                sem_morphology = self.extract_morphological_features(sem_gray)
                sem_edges = self.extract_edge_features(sem_gray)
                
                # Add prefix to distinguish SEM features
                for key, value in sem_features.items():
                    sample_features[f'sem_{key}'] = value
                for key, value in sem_texture.items():
                    sample_features[f'sem_texture_{key}'] = value
                for key, value in sem_morphology.items():
                    sample_features[f'sem_morph_{key}'] = value
                for key, value in sem_edges.items():
                    sample_features[f'sem_{key}'] = value
        
        # Analyze regular optical image
        img_path = sample_path / "image.jpg"
        if img_path.exists():
            opt_img, opt_gray = self.load_and_preprocess_image(str(img_path))
            if opt_gray is not None:
                opt_features = self.extract_basic_image_features(opt_gray)
                opt_texture = self.extract_texture_features(opt_gray)
                opt_morphology = self.extract_morphological_features(opt_gray)
                opt_edges = self.extract_edge_features(opt_gray)
                
                # Add prefix to distinguish optical features
                for key, value in opt_features.items():
                    sample_features[f'optical_{key}'] = value
                for key, value in opt_texture.items():
                    sample_features[f'optical_texture_{key}'] = value
                for key, value in opt_morphology.items():
                    sample_features[f'optical_morph_{key}'] = value
                for key, value in opt_edges.items():
                    sample_features[f'optical_{key}'] = value
        
        # Analyze element-specific images
        for element in self.elements:
            element_path = sample_path / f"{element}.jpeg"
            if element_path.exists():
                elem_img, elem_gray = self.load_and_preprocess_image(str(element_path))
                if elem_gray is not None:
                    elem_features = self.extract_basic_image_features(elem_gray)
                    elem_texture = self.extract_texture_features(elem_gray)
                    elem_morphology = self.extract_morphological_features(elem_gray)
                    
                    # Add element prefix
                    for key, value in elem_features.items():
                        sample_features[f'{element}_{key}'] = value
                    for key, value in elem_texture.items():
                        sample_features[f'{element}_texture_{key}'] = value
                    for key, value in elem_morphology.items():
                        sample_features[f'{element}_morph_{key}'] = value
        
        return sample_features
    
    def extract_all_features(self):
        """Extract features from all samples"""
        print("Extracting computer vision features from all samples...")
        
        all_features = []
        total_samples = len(self.tables) * len(self.samples) * len(self.replicates)
        processed = 0
        
        for table in self.tables:
            for sample in self.samples:
                for replicate in self.replicates:
                    sample_features = self.analyze_single_sample(table, sample, replicate)
                    if sample_features:
                        all_features.append(sample_features)
                    
                    processed += 1
                    if processed % 50 == 0:
                        print(f"Processed {processed}/{total_samples} samples")
        
        self.features_df = pd.DataFrame(all_features)
        print(f"Extracted features from {len(all_features)} samples")
        
        # Save features
        self.features_df.to_csv(os.path.join(_SCRIPT_DIR, 'computer_vision_features.csv'), index=False)
        
        return self.features_df
    
    def perform_feature_analysis(self):
        """Perform analysis on extracted features"""
        if not hasattr(self, 'features_df') or self.features_df.empty:
            print("No features available. Run extract_all_features() first.")
            return
        
        print("\nPerforming computer vision feature analysis...")
        
        # Get numeric columns only
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        numeric_data = self.features_df[numeric_cols]
        
        # Remove columns with all NaN or constant values
        numeric_data = numeric_data.dropna(axis=1, how='all')
        numeric_data = numeric_data.loc[:, numeric_data.std() != 0]
        
        if numeric_data.empty:
            print("No valid numeric features found.")
            return
        
        # Create feature importance plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Feature variance analysis
        feature_vars = numeric_data.var().sort_values(ascending=False)
        top_var_features = feature_vars.head(20)
        
        axes[0, 0].barh(range(len(top_var_features)), top_var_features.values)
        axes[0, 0].set_yticks(range(len(top_var_features)))
        axes[0, 0].set_yticklabels(top_var_features.index, fontsize=8)
        axes[0, 0].set_title('Top 20 Features by Variance')
        axes[0, 0].set_xlabel('Variance')
        
        # Feature correlation heatmap (subset)
        corr_matrix = numeric_data.corr()
        # Select subset for visualization
        subset_features = feature_vars.head(15).index
        subset_corr = corr_matrix.loc[subset_features, subset_features]
        
        sns.heatmap(subset_corr, annot=False, cmap='RdBu_r', center=0, ax=axes[0, 1])
        axes[0, 1].set_title('Feature Correlation Matrix (Top 15 Features)')
        axes[0, 1].tick_params(axis='both', labelsize=8)
        
        # Feature distribution by table
        if 'table' in self.features_df.columns:
            # Select a representative feature for analysis
            representative_feature = feature_vars.index[0] if len(feature_vars) > 0 else numeric_cols[0]
            
            table_data = []
            table_labels = []
            for table in self.features_df['table'].unique():
                if pd.notna(table):
                    table_subset = self.features_df[self.features_df['table'] == table]
                    if representative_feature in table_subset.columns:
                        feature_values = table_subset[representative_feature].dropna()
                        if len(feature_values) > 0:
                            table_data.append(feature_values)
                            table_labels.append(table)
            
            if table_data:
                axes[1, 0].boxplot(table_data, labels=table_labels)
                axes[1, 0].set_title(f'Feature Distribution by Table\n({representative_feature})')
                axes[1, 0].set_ylabel('Feature Value')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Feature summary statistics
        summary_stats = numeric_data.describe().T
        summary_stats = summary_stats.sort_values('std', ascending=False).head(10)
        
        x_pos = np.arange(len(summary_stats))
        axes[1, 1].bar(x_pos, summary_stats['mean'], yerr=summary_stats['std'], 
                      capsize=5, alpha=0.7)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(summary_stats.index, rotation=45, ha='right', fontsize=8)
        axes[1, 1].set_title('Top 10 Features by Standard Deviation')
        axes[1, 1].set_ylabel('Mean ± Std')
        
        plt.tight_layout()
        plt.savefig(os.path.join(_SCRIPT_DIR, 'computer_vision_feature_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save summary statistics
        summary_stats.to_csv(os.path.join(_SCRIPT_DIR, 'cv_feature_summary.csv'))
        
        return numeric_data
    
    def clustering_analysis(self, features_data):
        """Perform clustering analysis on CV features"""
        print("\nPerforming clustering analysis...")
        
        if features_data is None or features_data.empty:
            print("No features data available.")
            return
        
        # Prepare data for clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_data.fillna(0))
        
        # Determine optimal number of clusters
        inertias = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features_scaled)
            inertias.append(kmeans.inertia_)
        
        # Perform clustering with optimal k (e.g., k=4)
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Create clustering visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Elbow curve
        axes[0].plot(k_range, inertias, 'bo-')
        axes[0].set_title('Elbow Method for Optimal k')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        
        # PCA visualization
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        scatter = axes[1].scatter(features_pca[:, 0], features_pca[:, 1], 
                                c=cluster_labels, cmap='tab10', alpha=0.6)
        axes[1].set_title(f'Clusters in PCA Space (k={optimal_k})')
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        
        # Cluster centers in PCA space
        centers_pca = pca.transform(kmeans.cluster_centers_)
        axes[1].scatter(centers_pca[:, 0], centers_pca[:, 1], 
                       c='red', marker='x', s=200, linewidths=3)
        
        # Cluster size distribution
        unique, counts = np.unique(cluster_labels, return_counts=True)
        axes[2].bar(unique, counts)
        axes[2].set_title('Cluster Size Distribution')
        axes[2].set_xlabel('Cluster ID')
        axes[2].set_ylabel('Number of Samples')
        
        plt.tight_layout()
        plt.savefig(os.path.join(_SCRIPT_DIR, 'cv_clustering_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save clustering results
        cluster_df = features_data.copy()
        cluster_df['cluster'] = cluster_labels
        cluster_df.to_csv(os.path.join(_SCRIPT_DIR, 'cv_clustering_results.csv'), index=False)
        
        return cluster_labels, kmeans
    
    def generate_cv_report(self):
        """Generate comprehensive computer vision analysis report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE COMPUTER VISION ANALYSIS REPORT")
        print("="*60)
        
        # Extract features
        features_df = self.extract_all_features()
        
        # Perform feature analysis
        numeric_features = self.perform_feature_analysis()
        
        # Perform clustering
        if numeric_features is not None and not numeric_features.empty:
            clusters, kmeans = self.clustering_analysis(numeric_features)
        
        print("\nComputer vision analysis complete!")
        print("Generated files:")
        print("- computer_vision_features.csv")
        print("- computer_vision_feature_analysis.png")
        print("- cv_clustering_analysis.png")
        print("- cv_feature_summary.csv")

if __name__ == "__main__":
    # Run computer vision analysis
    cv_analyzer = CrystallizationComputerVisionAnalysis()
    cv_analyzer.generate_cv_report() 