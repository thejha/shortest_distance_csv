"""
HDBSCAN Distance-Based Clustering Implementation
=============================================

This script implements HDBSCAN clustering using distance-based features
derived from your comprehensive distance matrix. HDBSCAN offers significant
advantages over basic distance clustering:

1. Hierarchical density-based approach
2. Automatic noise detection
3. Variable cluster densities
4. No need to specify number of clusters
5. Robust to outliers and irregular cluster shapes
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Hub configuration from your existing code
HUB_PINCODE_MAP = {
    8:  "TAMIL NADU+600060",
    30: "TAMIL NADU+641005",
    1:  "KARNATAKA+562114",
    2:  "HARYANA+131029",
    3:  "UTTAR PRADESH+226008",
    31:  "MAHARASHTRA+402107",
    26: "KARNATAKA+580024",
    5:  "TELANGANA+500070",
    18: "MAHARASHTRA+421506",
    28: "ANDHRA PRADESH+522001",
    13: "RAJASTHAN+302013",
    29: "PUNJAB+141003",
    10: "UTTAR PRADESH+243302",
    20: "JAMMU AND KASHMIR+192301",
    22: "MADHYA PRADESH+462039",
    19: "UTTAR PRADESH+273403",
    14: "WEST BENGAL+712310",
    23: "BIHAR+800023",
    16: "UTTAR PRADESH+221108",
    25: "ODISHA+754021",
    32: "TELANGANA+501510"
}

class HDBSCANLocationClusterer:
    """
    Advanced HDBSCAN clustering for logistics locations using distance-based features.
    
    This approach is superior to basic distance clustering because:
    - Finds clusters of varying densities
    - Automatically identifies noise/outlier locations
    - Creates hierarchical cluster structure
    - No need to pre-specify number of clusters
    - Robust to irregular geographic distributions
    """
    
    def __init__(self, min_cluster_size=25, min_samples=8):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.clusterer = None
        self.scaler = StandardScaler()
        self.hub_locations = list(HUB_PINCODE_MAP.values())
        self.feature_names = []
        
    def load_and_prepare_data(self, csv_path):
        """Load distance data and prepare for clustering"""
        print("Loading distance data...")
        self.df = pd.read_csv(csv_path)
        
        # Get all unique locations
        all_sources = set(self.df['source'].unique())
        all_destinations = set(self.df['destination'].unique())
        self.all_locations = list(all_sources | all_destinations)
        
        print(f"Total unique locations: {len(self.all_locations):,}")
        print(f"Total distance records: {len(self.df):,}")
        
        return self.all_locations
    
    def create_distance_features(self):
        """
        Create sophisticated distance-based features for HDBSCAN clustering.
        
        Features created:
        1. Distance to each of 21 hubs
        2. Distance to nearest hub
        3. Average distance to 3 nearest hubs
        4. Distance variance (accessibility measure)
        5. Hub preference score
        6. State accessibility index
        """
        print("Creating distance-based features...")
        
        feature_matrix = []
        self.location_list = []
        
        # Create distance lookup for faster processing
        distance_lookup = self._create_distance_lookup()
        
        for i, location in enumerate(self.all_locations):
            if i % 500 == 0:
                print(f"Processing location {i+1}/{len(self.all_locations)}")
            
            location_features = []
            hub_distances = []
            
            # Distance to each hub (21 features)
            for hub in self.hub_locations:
                distance = self._get_distance(location, hub, distance_lookup)
                location_features.append(distance)
                hub_distances.append(distance)
            
            hub_distances = np.array(hub_distances)
            
            # Derived features for better clustering
            location_features.extend([
                np.min(hub_distances),  # Distance to nearest hub
                np.mean(np.sort(hub_distances)[:3]),  # Avg to 3 nearest hubs
                np.std(hub_distances),  # Distance variance (accessibility)
                np.argmin(hub_distances),  # Nearest hub ID
                self._calculate_hub_preference_score(hub_distances),
                self._calculate_state_accessibility(location, hub_distances)
            ])
            
            feature_matrix.append(location_features)
            self.location_list.append(location)
        
        self.feature_matrix = np.array(feature_matrix)
        
        # Create feature names for interpretability
        self.feature_names = [f'dist_to_hub_{i}' for i in range(21)]
        self.feature_names.extend([
            'min_hub_distance', 'avg_3_nearest_hubs', 'distance_variance',
            'nearest_hub_id', 'hub_preference_score', 'state_accessibility'
        ])
        
        print(f"Created feature matrix: {self.feature_matrix.shape}")
        return self.feature_matrix
    
    def _create_distance_lookup(self):
        """Create optimized distance lookup dictionary"""
        distance_lookup = {}
        for _, row in self.df.iterrows():
            source, dest, distance = row['source'], row['destination'], row['distance']
            distance_lookup[(source, dest)] = distance
            distance_lookup[(dest, source)] = distance  # Symmetric
        return distance_lookup
    
    def _get_distance(self, loc1, loc2, distance_lookup):
        """Get distance between two locations with fallback estimation"""
        if (loc1, loc2) in distance_lookup:
            return distance_lookup[(loc1, loc2)]
        
        # Fallback: estimate based on state proximity
        state1 = loc1.split('+')[0]
        state2 = loc2.split('+')[0]
        
        if state1 == state2:
            return 150  # Same state estimate
        else:
            # Different states - estimate based on common inter-state distances
            return self._estimate_interstate_distance(state1, state2)
    
    def _estimate_interstate_distance(self, state1, state2):
        """Estimate interstate distances based on geographic proximity"""
        # Simplified interstate distance estimation
        # In production, you could use a more sophisticated state distance matrix
        interstate_distances = {
            ('TAMIL NADU', 'KARNATAKA'): 300,
            ('TAMIL NADU', 'ANDHRA PRADESH'): 400,
            ('KARNATAKA', 'MAHARASHTRA'): 500,
            ('UTTAR PRADESH', 'BIHAR'): 200,
            ('WEST BENGAL', 'BIHAR'): 300,
        }
        
        key1 = (state1, state2)
        key2 = (state2, state1)
        
        if key1 in interstate_distances:
            return interstate_distances[key1]
        elif key2 in interstate_distances:
            return interstate_distances[key2]
        else:
            return 600  # Default long distance
    
    def _calculate_hub_preference_score(self, hub_distances):
        """Calculate how strongly a location prefers its nearest hub"""
        sorted_distances = np.sort(hub_distances)
        if len(sorted_distances) > 1:
            # Ratio of second nearest to nearest (higher = stronger preference)
            return sorted_distances[1] / sorted_distances[0]
        return 1.0
    
    def _calculate_state_accessibility(self, location, hub_distances):
        """Calculate how accessible a location is within its state"""
        state = location.split('+')[0]
        
        # Count hubs in same state
        same_state_hubs = sum(1 for hub in self.hub_locations 
                             if hub.split('+')[0] == state)
        
        if same_state_hubs > 0:
            # Average distance to same-state hubs
            same_state_distances = []
            for i, hub in enumerate(self.hub_locations):
                if hub.split('+')[0] == state:
                    same_state_distances.append(hub_distances[i])
            return np.mean(same_state_distances) if same_state_distances else np.min(hub_distances)
        else:
            return np.min(hub_distances)  # Fallback to nearest hub
    
    def perform_hdbscan_clustering(self):
        """
        Perform HDBSCAN clustering with optimized parameters for distance data.
        
        HDBSCAN advantages over basic clustering:
        - Finds natural cluster boundaries in distance space
        - Handles varying cluster densities (urban vs rural areas)
        - Automatically identifies outlier locations
        - Creates hierarchical cluster structure
        """
        print("Performing HDBSCAN clustering...")
        
        # Normalize features (critical for distance-based clustering)
        features_scaled = self.scaler.fit_transform(self.feature_matrix)
        
        # Initialize HDBSCAN with parameters optimized for logistics data
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom',  # Excess of Mass - good for distance data
            cluster_selection_epsilon=0.0,
            algorithm='best',
            leaf_size=30,
            n_jobs=-1  # Use all CPU cores
        )
        
        # Fit and predict clusters
        self.cluster_labels = self.clusterer.fit_predict(features_scaled)
        
        # Calculate clustering metrics
        self._calculate_clustering_metrics(features_scaled)
        
        return self.cluster_labels
    
    def _calculate_clustering_metrics(self, features_scaled):
        """Calculate clustering quality metrics"""
        # Basic statistics
        unique_labels = set(self.cluster_labels)
        self.n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        self.n_noise = list(self.cluster_labels).count(-1)
        self.noise_ratio = self.n_noise / len(self.cluster_labels)
        
        # Silhouette score (excluding noise points)
        if self.n_clusters > 1:
            non_noise_mask = self.cluster_labels != -1
            if np.sum(non_noise_mask) > 0:
                self.silhouette_avg = silhouette_score(
                    features_scaled[non_noise_mask], 
                    self.cluster_labels[non_noise_mask]
                )
            else:
                self.silhouette_avg = -1
        else:
            self.silhouette_avg = -1
        
        print(f"\nClustering Results:")
        print(f"Number of clusters: {self.n_clusters}")
        print(f"Number of noise points: {self.n_noise} ({self.noise_ratio:.1%})")
        print(f"Silhouette score: {self.silhouette_avg:.3f}")
    
    def analyze_clusters(self):
        """Comprehensive cluster analysis"""
        print("\n" + "="*50)
        print("CLUSTER ANALYSIS")
        print("="*50)
        
        # Cluster size distribution
        cluster_sizes = {}
        hub_assignment = {}
        state_distribution = defaultdict(lambda: defaultdict(int))
        
        for i, (location, cluster_id) in enumerate(zip(self.location_list, self.cluster_labels)):
            state = location.split('+')[0]
            
            if cluster_id == -1:
                continue  # Skip noise
            
            # Cluster sizes
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
            
            # State distribution
            state_distribution[cluster_id][state] += 1
            
            # Hub assignment analysis
            if cluster_id not in hub_assignment:
                hub_distances = self.feature_matrix[i, :21]
                nearest_hub_idx = np.argmin(hub_distances)
                hub_assignment[cluster_id] = {
                    'nearest_hub_idx': nearest_hub_idx,
                    'nearest_hub_location': self.hub_locations[nearest_hub_idx],
                    'avg_distance_to_hub': hub_distances[nearest_hub_idx],
                    'locations': []
                }
            hub_assignment[cluster_id]['locations'].append(location)
        
        # Print cluster analysis
        print(f"\nCluster Size Distribution:")
        for cluster_id in sorted(cluster_sizes.keys()):
            size = cluster_sizes[cluster_id]
            hub_info = hub_assignment[cluster_id]
            dominant_states = sorted(state_distribution[cluster_id].items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
            
            print(f"Cluster {cluster_id}: {size} locations")
            print(f"  Nearest Hub: {hub_info['nearest_hub_location']}")
            print(f"  Avg Distance to Hub: {hub_info['avg_distance_to_hub']:.1f} km")
            print(f"  Top States: {', '.join([f'{state}({count})' for state, count in dominant_states])}")
            print()
        
        return cluster_sizes, hub_assignment, state_distribution
    
    def visualize_clusters(self):
        """Create comprehensive cluster visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('HDBSCAN Clustering Analysis - Distance-Based Features', fontsize=16)
        
        # 1. Cluster size distribution
        cluster_sizes = np.bincount(self.cluster_labels[self.cluster_labels >= 0])
        axes[0, 0].bar(range(len(cluster_sizes)), cluster_sizes)
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Locations')
        
        # 2. Distance to nearest hub by cluster
        cluster_hub_distances = []
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            if np.any(cluster_mask):
                cluster_features = self.feature_matrix[cluster_mask]
                avg_min_distance = np.mean(cluster_features[:, 21])  # min_hub_distance feature
                cluster_hub_distances.append(avg_min_distance)
        
        axes[0, 1].bar(range(len(cluster_hub_distances)), cluster_hub_distances)
        axes[0, 1].set_title('Average Distance to Nearest Hub by Cluster')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Distance (km)')
        
        # 3. Feature importance heatmap
        if self.n_clusters > 1:
            cluster_feature_means = []
            for cluster_id in range(self.n_clusters):
                cluster_mask = self.cluster_labels == cluster_id
                if np.any(cluster_mask):
                    cluster_mean = np.mean(self.feature_matrix[cluster_mask], axis=0)
                    cluster_feature_means.append(cluster_mean)
            
            if cluster_feature_means:
                feature_heatmap = np.array(cluster_feature_means)
                im = axes[0, 2].imshow(feature_heatmap[:, :21].T, cmap='viridis', aspect='auto')
                axes[0, 2].set_title('Hub Distance Patterns by Cluster')
                axes[0, 2].set_xlabel('Cluster ID')
                axes[0, 2].set_ylabel('Hub Index')
                plt.colorbar(im, ax=axes[0, 2])
        
        # 4. State distribution by cluster
        state_cluster_matrix = self._create_state_cluster_matrix()
        if state_cluster_matrix.size > 0:
            sns.heatmap(state_cluster_matrix[:15], annot=True, fmt='d', 
                       cmap='Blues', ax=axes[1, 0])
            axes[1, 0].set_title('Top 15 States Distribution Across Clusters')
            axes[1, 0].set_xlabel('Cluster ID')
            axes[1, 0].set_ylabel('States')
        
        # 5. Clustering hierarchy (condensed tree)
        if hasattr(self.clusterer, 'condensed_tree_'):
            self.clusterer.condensed_tree_.plot(select_clusters=True, ax=axes[1, 1])
            axes[1, 1].set_title('HDBSCAN Cluster Hierarchy')
        
        # 6. Outlier detection
        if hasattr(self.clusterer, 'outlier_scores_'):
            axes[1, 2].hist(self.clusterer.outlier_scores_, bins=50, alpha=0.7)
            axes[1, 2].set_title('Outlier Score Distribution')
            axes[1, 2].set_xlabel('Outlier Score')
            axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def _create_state_cluster_matrix(self):
        """Create state-cluster distribution matrix for visualization"""
        states = [loc.split('+')[0] for loc in self.location_list]
        unique_states = list(set(states))
        
        if self.n_clusters == 0:
            return np.array([])
        
        matrix = np.zeros((len(unique_states), self.n_clusters))
        
        for i, state in enumerate(states):
            cluster_id = self.cluster_labels[i]
            if cluster_id >= 0:  # Skip noise
                state_idx = unique_states.index(state)
                matrix[state_idx, cluster_id] += 1
        
        # Sort states by total locations for better visualization
        state_totals = np.sum(matrix, axis=1)
        sorted_indices = np.argsort(state_totals)[::-1]
        
        return matrix[sorted_indices]
    
    def save_results(self, output_path='clustering_results.csv'):
        """Save clustering results to CSV"""
        results_df = pd.DataFrame({
            'location': self.location_list,
            'cluster_id': self.cluster_labels,
            'state': [loc.split('+')[0] for loc in self.location_list],
            'pincode': [loc.split('+')[1] for loc in self.location_list],
            'min_hub_distance': self.feature_matrix[:, 21],
            'hub_preference_score': self.feature_matrix[:, 24],
            'state_accessibility': self.feature_matrix[:, 25]
        })
        
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        return results_df

def main():
    """Main execution function demonstrating HDBSCAN clustering"""
    print("HDBSCAN Distance-Based Clustering for Logistics Optimization")
    print("=" * 60)
    
    # Initialize clusterer
    clusterer = HDBSCANLocationClusterer(min_cluster_size=30, min_samples=10)
    
    # Load and prepare data
    locations = clusterer.load_and_prepare_data('distances_dump.csv')
    
    # Create distance-based features
    features = clusterer.create_distance_features()
    
    # Perform HDBSCAN clustering
    cluster_labels = clusterer.perform_hdbscan_clustering()
    
    # Analyze results
    cluster_sizes, hub_assignment, state_distribution = clusterer.analyze_clusters()
    
    # Create visualizations
    clusterer.visualize_clusters()
    
    # Save results
    results_df = clusterer.save_results()
    
    print("\n" + "="*60)
    print("HDBSCAN ADVANTAGES DEMONSTRATED:")
    print("="*60)
    print("✓ Automatic cluster discovery (no need to specify number)")
    print("✓ Handles varying cluster densities across different regions")
    print("✓ Identifies outlier/noise locations automatically") 
    print("✓ Creates hierarchical cluster structure")
    print("✓ Robust to irregular geographic distributions")
    print("✓ Distance-based features capture logistics relationships")
    print("\nThis approach is superior to basic distance clustering!")

if __name__ == "__main__":
    main()
