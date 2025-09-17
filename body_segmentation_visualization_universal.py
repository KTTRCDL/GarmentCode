# 导入必要的库
import trimesh
import json
import numpy as np
import matplotlib.pyplot as plt
import os

def color_body_mesh(obj_path, seg_path, output_path):
    """
    通用的身体网格着色函数，可以处理不同格式的OBJ文件
    
    Args:
        obj_path: OBJ文件路径
        seg_path: 分割数据JSON文件路径
        output_path: 输出彩色OBJ文件路径
    """
    
    # 加载人体网格
    print(f"正在加载人体网格: {obj_path}")
    mesh = trimesh.load(obj_path)
    print(f"网格信息: {mesh.vertices.shape[0]} 个顶点, {mesh.faces.shape[0]} 个面")
    
    # 检查网格属性
    print(f"原始网格材质状态: Has material = {hasattr(mesh.visual, 'material')}")
    print(f"原始网格顶点颜色状态: Has vertex_colors = {hasattr(mesh.visual, 'vertex_colors')}")
    
    # 加载身体分割数据
    print(f"正在加载身体分割数据: {seg_path}")
    with open(seg_path, 'r') as f:
        body_seg = json.load(f)
    
    print("分割部位:")
    for part_name, vertex_indices in body_seg.items():
        print(f"  {part_name}: {len(vertex_indices)} 个顶点")
    
    # 根据分割文件类型选择颜色方案
    if 'ggg_body_segmentation' in seg_path:
        colors = {
            'body': [1.0, 0.0, 0.0, 1.0],        # 红色
            'left_arm': [0.0, 1.0, 0.0, 1.0],    # 绿色
            'right_arm': [0.0, 0.0, 1.0, 1.0],   # 蓝色
            'left_leg': [1.0, 1.0, 0.0, 1.0],    # 黄色
            'right_leg': [1.0, 0.0, 1.0, 1.0],   # 紫色
            'face_internal': [0.0, 1.0, 1.0, 1.0] # 青色
        }
    else:  # smpl_vert_segmentation
        colors = {
            'rightHand': [1.0, 0.0, 0.0, 0.5],
            'rightUpLeg': [0.0, 1.0, 0.0, 0.5],
            'leftArm': [0.0, 0.0, 1.0, 0.5],
            'leftLeg': [1.0, 1.0, 0.0, 0.5],
            'leftToeBase': [1.0, 0.0, 1.0, 0.5],
            'leftFoot': [0.0, 1.0, 1.0, 0.5],
            'spine1': [1.0, 0.0, 0.0, 0.5],
            'spine2': [0.0, 1.0, 0.0, 0.5],
            'leftShoulder': [0.0, 0.0, 1.0, 0.5],
            'rightShoulder': [1.0, 1.0, 0.0, 0.5],
            'rightFoot': [1.0, 0.0, 1.0, 0.5],
            'head': [0.0, 1.0, 1.0, 0.5],
            'rightArm': [1.0, 0.0, 0.0, 0.5],
            'leftHandIndex1': [0.0, 1.0, 0.0, 0.5],
            'rightLeg': [0.0, 0.0, 1.0, 0.5],
            'rightHandIndex1': [1.0, 1.0, 0.0, 0.5],
            'leftForeArm': [1.0, 0.0, 1.0, 0.5],
            'rightForeArm': [0.0, 1.0, 1.0, 0.5],
            'leftUpLeg': [1.0, 0.0, 0.0, 0.5],
            "neck": [0.0, 1.0, 0.0, 0.5],
            "rightToeBase": [1.0, 0.0, 1.0, 0.5],
            "spine": [0.0, 1.0, 1.0, 0.5],
            "leftUpLeg": [1.0, 0.0, 0.0, 0.5],
            "leftHand": [0.0, 1.0, 0.0, 0.5],
            "hips": [0.0, 0.0, 1.0, 0.5]
        }
    
    # 安全地处理材质
    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        print("删除现有材质...")
        del mesh.visual.material
    
    # 创建顶点颜色数组，默认为灰色
    vertex_colors = np.ones((len(mesh.vertices), 4)) * 0.5
    vertex_colors[:, 3] = 1.0  # 设置透明度为1
    
    # 为每个部位分配颜色，添加边界检查
    max_vertex_index = len(mesh.vertices) - 1
    print(f"最大顶点索引: {max_vertex_index}")
    
    for part_name, vertex_indices in body_seg.items():
        if part_name in colors:
            color = colors[part_name]
            # 过滤掉超出边界的索引
            valid_indices = [idx for idx in vertex_indices if 0 <= idx <= max_vertex_index]
            invalid_count = len(vertex_indices) - len(valid_indices)
            
            if invalid_count > 0:
                print(f"警告: 部位 '{part_name}' 有 {invalid_count} 个无效索引被忽略")
            
            print(f"为部位 '{part_name}' 分配颜色 {color[:3]}，包含 {len(valid_indices)} 个有效顶点")
            vertex_colors[valid_indices] = color
    
    # 设置网格的顶点颜色
    mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
    print(f"设置后网格顶点颜色状态: Has vertex_colors = {hasattr(mesh.visual, 'vertex_colors')}")
    
    # 创建分割统计信息
    stats = {}
    total_vertices = sum(len(indices) for indices in body_seg.values())
    
    for part_name, vertex_indices in body_seg.items():
        valid_indices = [idx for idx in vertex_indices if 0 <= idx <= max_vertex_index]
        stats[part_name] = {
            'vertex_count': len(valid_indices),
            'percentage': (len(valid_indices) / total_vertices) * 100 if total_vertices > 0 else 0
        }
    
    # 打印统计信息
    print("\n=== 身体分割统计信息 ===")
    print(f"{'部位':<15} {'顶点数':<10} {'百分比':<10}")
    print("-" * 35)
    
    for part_name, info in stats.items():
        print(f"{part_name:<15} {info['vertex_count']:<10} {info['percentage']:.2f}%")
    
    print("-" * 35)
    total_valid_vertices = sum(info['vertex_count'] for info in stats.values())
    print(f"{'总计':<15} {total_valid_vertices:<10} 100.00%")
    
    # 保存带颜色的OBJ文件
    print(f"\n正在保存带颜色的OBJ文件到: {output_path}")
    mesh.export(output_path)
    print(f"带颜色的OBJ文件已保存到: {output_path}")
    
    # 验证保存的文件
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"保存的文件大小: {file_size} 字节")
        
        # 尝试重新加载验证
        try:
            test_mesh = trimesh.load(output_path)
            print(f"验证: 重新加载的网格有 {test_mesh.vertices.shape[0]} 个顶点")
            print(f"验证: 重新加载的网格有顶点颜色: {hasattr(test_mesh.visual, 'vertex_colors')}")
            return True
        except Exception as e:
            print(f"验证失败: {e}")
            return False
    else:
        print("错误: 文件保存失败")
        return False

# 测试用例1: mean_female.obj (工作正常的情况)
print("=" * 60)
print("测试用例1: mean_female.obj")
print("=" * 60)
obj_path1 = "/data/lixinag/project/GarmentCode/assets/bodies/mean_female.obj"
seg_path1 = "/data/lixinag/project/GarmentCode/A_RESULT/bodies/ggg_body_segmentation.json"
output_path1 = "/data/lixinag/project/GarmentCode/A_RESULT/colored_body1.obj"

success1 = color_body_mesh(obj_path1, seg_path1, output_path1)

print("\n" + "=" * 60)
print("测试用例2: f_smpl_average_A40_modify.obj")
print("=" * 60)
obj_path2 = "/data/lixinag/project/GarmentCode/A_RESULT/bodies/f_smpl_average_A40_modify.obj"
seg_path2 = "/data/lixinag/project/GarmentCode/A_RESULT/bodies/smpl_vert_segmentation.json"
output_path2 = "/data/lixinag/project/GarmentCode/A_RESULT/colored_body2.obj"

success2 = color_body_mesh(obj_path2, seg_path2, output_path2)

print("\n" + "=" * 60)
print("测试结果总结")
print("=" * 60)
print(f"mean_female.obj 处理结果: {'成功' if success1 else '失败'}")
print(f"f_smpl_average_A40_modify.obj 处理结果: {'成功' if success2 else '失败'}")
