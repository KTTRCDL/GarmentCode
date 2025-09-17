#!/usr/bin/env python3
"""
Test script to generate body segmentation with exactly 6 parts:
- body
- left_arm  
- right_arm
- left_leg
- right_leg
- face_internal
"""

from body_segmentation_tool import BodySegmentationTool

def main():
    # Input and output paths
    input_obj = "/data/lixinag/project/GarmentCode/A_RESULT/bodies/mean_all.obj"
    output_json = "/data/lixinag/project/GarmentCode/A_RESULT/bodies/mean_all_six_parts.json"
    colored_output = "/data/lixinag/project/GarmentCode/A_RESULT/mean_all_six_parts_colored.obj"
    
    print("=== Generating 6-Part Body Segmentation ===")
    print(f"Input: {input_obj}")
    print(f"Output: {output_json}")
    print()
    
    # Create segmentation tool
    tool = BodySegmentationTool(input_obj)
    
    # Perform geometric segmentation (will output exactly 6 parts)
    print("Performing geometric segmentation for 6 body parts...")
    tool.geometric_segmentation()
    
    # Save segmentation
    tool.save_segmentation(output_json)
    
    # Create colored visualization
    print(f"\nCreating colored visualization: {colored_output}")
    tool.visualize_segmentation(colored_output)
    
    print("\n=== Complete ===")
    print("The segmentation JSON file contains exactly 6 keys:")
    print("- body")
    print("- left_arm") 
    print("- right_arm")
    print("- left_leg")
    print("- right_leg")
    print("- face_internal")

if __name__ == "__main__":
    main()
