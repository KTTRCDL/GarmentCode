#!/usr/bin/env python3
"""
3D Human Body Segmentation Tool

This tool provides multiple approaches for segmenting 3D human body meshes:
1. Geometric-based segmentation using height and position
2. Manual refinement with interactive tools
3. Transfer segmentation from reference models
4. Export segmentation to JSON format

Usage:
    python body_segmentation_tool.py --input /path/to/body.obj --output /path/to/segmentation.json
"""

import argparse
import json
import numpy as np
import trimesh
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import open3d as o3d

class BodySegmentationTool:
    def __init__(self, mesh_path):
        """Initialize the segmentation tool with a mesh"""
        self.mesh_path = Path(mesh_path)
        self.mesh = trimesh.load(str(self.mesh_path))
        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces
        self.segmentation = {}
        
        print(f"Loaded mesh: {self.mesh_path}")
        print(f"Vertices: {len(self.vertices)}")
        print(f"Faces: {len(self.faces)}")
        
    def geometric_segmentation(self):
        """Perform geometric-based segmentation using height and position analysis"""
        print("Performing geometric segmentation...")
        
        # Normalize the mesh to standard orientation
        self._normalize_mesh()
        
        # Get bounding box
        bbox_min = self.vertices.min(axis=0)
        bbox_max = self.vertices.max(axis=0)
        height = bbox_max[1] - bbox_min[1]
        
        print(f"Body height: {height:.3f}")
        print(f"Bounding box: {bbox_min} to {bbox_max}")
        
        # Define body part height ranges (normalized) - only 6 parts as requested
        height_ranges = {
            'body': (0.45, 0.80),
            'left_arm': (0.45, 0.80),
            'right_arm': (0.45, 0.80),
            'left_leg': (0.0, 0.45),
            'right_leg': (0.0, 0.45),
            'face_internal': (0.85, 1.0)
        }
        
        # Normalize vertex heights
        normalized_heights = (self.vertices[:, 1] - bbox_min[1]) / height
        
        # Initialize segmentation
        for part_name in height_ranges.keys():
            self.segmentation[part_name] = []
        
        # Assign vertices to body parts based on height
        for i, height_norm in enumerate(normalized_heights):
            assigned = False
            
            for part_name, (min_h, max_h) in height_ranges.items():
                if min_h <= height_norm <= max_h:
                    # Additional position-based filtering
                    if self._is_valid_for_part(i, part_name, height_norm):
                        self.segmentation[part_name].append(i)
                        assigned = True
                        break
            
            if not assigned:
                # Assign to closest part
                closest_part = min(height_ranges.keys(), 
                                 key=lambda x: abs(height_norm - np.mean(height_ranges[x])))
                self.segmentation[closest_part].append(i)
        
        # Post-process to improve segmentation
        self._post_process_segmentation()
        
        return self.segmentation
    
    def _normalize_mesh(self):
        """Normalize mesh to standard orientation (Y-up, centered)"""
        # Center the mesh
        center = self.vertices.mean(axis=0)
        self.vertices = self.vertices - center
        
        # Ensure Y is the up direction (height)
        # This assumes the mesh is already roughly oriented
        # You might need to add rotation logic here
        
        # Update mesh
        self.mesh.vertices = self.vertices
    
    def _is_valid_for_part(self, vertex_idx, part_name, height_norm):
        """Check if a vertex is valid for a specific body part"""
        vertex = self.vertices[vertex_idx]
        
        # Basic position-based rules for 6 parts
        if part_name == 'left_arm':
            return vertex[0] < 0  # Left side
        elif part_name == 'right_arm':
            return vertex[0] > 0  # Right side
        elif part_name == 'left_leg':
            return vertex[0] < 0 and height_norm < 0.5  # Left side, lower body
        elif part_name == 'right_leg':
            return vertex[0] > 0 and height_norm < 0.5  # Right side, lower body
        elif part_name == 'face_internal':
            return height_norm > 0.85
        elif part_name == 'body':
            return 0.45 <= height_norm <= 0.80 and abs(vertex[0]) < 0.3
        
        return True
    
    def _post_process_segmentation(self):
        """Post-process segmentation to improve quality"""
        # Remove small isolated regions
        min_vertices = 10
        for part_name in list(self.segmentation.keys()):
            if len(self.segmentation[part_name]) < min_vertices:
                print(f"Removing small part: {part_name} ({len(self.segmentation[part_name])} vertices)")
                del self.segmentation[part_name]
        
        # Ensure all vertices are assigned
        assigned_vertices = set()
        for part_name, vertices in self.segmentation.items():
            assigned_vertices.update(vertices)
        
        unassigned = set(range(len(self.vertices))) - assigned_vertices
        if unassigned:
            print(f"Found {len(unassigned)} unassigned vertices, assigning to body")
            self.segmentation['body'].extend(list(unassigned))
    
    def cluster_based_segmentation(self, n_clusters=8):
        """Perform segmentation using clustering"""
        print(f"Performing clustering-based segmentation with {n_clusters} clusters...")
        
        # Prepare features for clustering
        features = np.column_stack([
            self.vertices,  # 3D position
            self.mesh.vertex_normals,  # Normal vectors
        ])
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Assign clusters to body parts based on position
        cluster_centers = kmeans.cluster_centers_
        
        # Map clusters to body parts
        cluster_to_part = self._map_clusters_to_parts(cluster_centers, scaler)
        
        # Create segmentation
        for i, label in enumerate(cluster_labels):
            part_name = cluster_to_part[label]
            if part_name not in self.segmentation:
                self.segmentation[part_name] = []
            self.segmentation[part_name].append(i)
        
        return self.segmentation
    
    def _map_clusters_to_parts(self, cluster_centers, scaler):
        """Map cluster centers to body parts"""
        # Inverse transform to get original scale
        centers_original = scaler.inverse_transform(cluster_centers)
        
        # Use Y coordinate (height) to determine body parts
        heights = centers_original[:, 1]
        sorted_indices = np.argsort(heights)
        
        # Define body parts from bottom to top (6 parts as requested)
        body_parts = ['left_leg', 'right_leg', 'body', 'left_arm', 'right_arm', 'face_internal']
        
        cluster_to_part = {}
        for i, cluster_idx in enumerate(sorted_indices):
            if i < len(body_parts):
                cluster_to_part[cluster_idx] = body_parts[i]
            else:
                cluster_to_part[cluster_idx] = 'body'  # Default
        
        return cluster_to_part
    
    def transfer_segmentation(self, reference_mesh_path, reference_seg_path):
        """Transfer segmentation from a reference mesh"""
        print(f"Transferring segmentation from {reference_mesh_path}")
        
        # Load reference mesh and segmentation
        ref_mesh = trimesh.load(reference_mesh_path)
        with open(reference_seg_path, 'r') as f:
            ref_seg = json.load(f)
        
        # Find closest vertices between meshes
        from scipy.spatial.distance import cdist
        
        # For each vertex in target mesh, find closest in reference
        distances = cdist(self.vertices, ref_mesh.vertices)
        closest_indices = np.argmin(distances, axis=1)
        
        # Transfer segmentation
        for part_name, ref_vertices in ref_seg.items():
            self.segmentation[part_name] = []
            
            for i, closest_idx in enumerate(closest_indices):
                if closest_idx in ref_vertices:
                    self.segmentation[part_name].append(i)
        
        return self.segmentation
    
    def save_segmentation(self, output_path):
        """Save segmentation to JSON file"""
        output_path = Path(output_path)
        
        # Convert numpy arrays to lists for JSON serialization
        seg_dict = {}
        for part_name, vertices in self.segmentation.items():
            seg_dict[part_name] = vertices.tolist() if isinstance(vertices, np.ndarray) else vertices
        
        with open(output_path, 'w') as f:
            json.dump(seg_dict, f, indent=2)
        
        print(f"Segmentation saved to: {output_path}")
        
        # Print statistics
        total_vertices = sum(len(vertices) for vertices in self.segmentation.values())
        print(f"\nSegmentation statistics:")
        print(f"Total vertices: {total_vertices}")
        print(f"Mesh vertices: {len(self.vertices)}")
        
        for part_name, vertices in self.segmentation.items():
            percentage = (len(vertices) / len(self.vertices)) * 100
            print(f"  {part_name}: {len(vertices)} vertices ({percentage:.1f}%)")
    
    def visualize_segmentation(self, save_path=None):
        """Visualize the segmentation with colors"""
        if not self.segmentation:
            print("No segmentation to visualize. Run segmentation first.")
            return
        
        # Create colors for each part
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.segmentation)))
        
        # Create vertex colors
        vertex_colors = np.ones((len(self.vertices), 4)) * 0.5  # Default gray
        vertex_colors[:, 3] = 1.0  # Alpha
        
        for i, (part_name, vertices) in enumerate(self.segmentation.items()):
            color = colors[i]
            vertex_colors[vertices] = color
        
        # Apply colors to mesh
        self.mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
        
        # Save colored mesh
        if save_path:
            self.mesh.export(save_path)
            print(f"Colored mesh saved to: {save_path}")
        
        # Show mesh (if possible)
        try:
            self.mesh.show()
        except Exception as e:
            print(f"Could not display mesh: {e}")
            print("Try opening the saved colored mesh file in a 3D viewer.")

def main():
    parser = argparse.ArgumentParser(description='3D Human Body Segmentation Tool')
    parser.add_argument('--input', required=True, help='Input OBJ file path')
    parser.add_argument('--output', required=True, help='Output JSON segmentation file path')
    parser.add_argument('--method', choices=['geometric', 'clustering', 'transfer'], 
                       default='geometric', help='Segmentation method')
    parser.add_argument('--reference-mesh', help='Reference mesh for transfer method')
    parser.add_argument('--reference-seg', help='Reference segmentation for transfer method')
    parser.add_argument('--clusters', type=int, default=8, help='Number of clusters for clustering method')
    parser.add_argument('--visualize', action='store_true', help='Visualize segmentation')
    parser.add_argument('--colored-output', help='Save colored mesh for visualization')
    
    args = parser.parse_args()
    
    # Create segmentation tool
    tool = BodySegmentationTool(args.input)
    
    # Perform segmentation
    if args.method == 'geometric':
        tool.geometric_segmentation()
    elif args.method == 'clustering':
        tool.cluster_based_segmentation(args.clusters)
    elif args.method == 'transfer':
        if not args.reference_mesh or not args.reference_seg:
            print("Error: --reference-mesh and --reference-seg are required for transfer method")
            return
        tool.transfer_segmentation(args.reference_mesh, args.reference_seg)
    
    # Save segmentation
    tool.save_segmentation(args.output)
    
    # Visualize if requested
    if args.visualize or args.colored_output:
        tool.visualize_segmentation(args.colored_output)

if __name__ == "__main__":
    main()
