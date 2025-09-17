# 导入必要的库
import trimesh
import json
import numpy as np
import matplotlib.pyplot as plt
import os

# 文件路径
obj_path = "/data/lixinag/project/GarmentCode/A_RESULT/bodies/f_smpl_average_A40_modify.obj"
seg_path = "/data/lixinag/project/GarmentCode/A_RESULT/bodies/smpl_vert_segmentation.json"

# 加载人体网格
print(f"正在加载人体网格: {obj_path}")
mesh = trimesh.load(obj_path)
print(f"网格信息: {mesh.vertices.shape[0]} 个顶点, {mesh.faces.shape[0]} 个面")

# 检查原始网格属性
print(f"原始网格材质状态: Has material = {hasattr(mesh.visual, 'material')}")
print(f"原始网格顶点颜色状态: Has vertex_colors = {hasattr(mesh.visual, 'vertex_colors')}")

# 加载身体分割数据
print(f"正在加载身体分割数据: {seg_path}")
with open(seg_path, 'r') as f:
    body_seg = json.load(f)

print("分割部位:")
for part_name, vertex_indices in body_seg.items():
    print(f"  {part_name}: {len(vertex_indices)} 个顶点")

# 定义不同部位的颜色
colors = {
    'rightHand': [1.0, 0.0, 0.0, 1.0],        # 红色
    'rightUpLeg': [0.0, 1.0, 0.0, 1.0],       # 绿色
    'leftArm': [0.0, 0.0, 1.0, 1.0],          # 蓝色
    'leftLeg': [1.0, 1.0, 0.0, 1.0],          # 黄色
    'rightLeg': [1.0, 0.0, 1.0, 1.0],         # 紫色
    'leftToeBase': [0.0, 1.0, 1.0, 1.0],      # 青色
    'leftFoot': [1.0, 0.5, 0.0, 1.0],         # 橙色
    'spine1': [0.5, 0.0, 1.0, 1.0],           # 紫罗兰
    'spine2': [0.0, 0.5, 1.0, 1.0],           # 天蓝
    'leftShoulder': [1.0, 0.0, 0.5, 1.0],     # 粉红
    'rightShoulder': [0.5, 1.0, 0.0, 1.0],    # 青绿
    'rightFoot': [1.0, 0.5, 0.5, 1.0],        # 浅粉
    'head': [0.5, 0.5, 0.5, 1.0],             # 灰色
    'rightArm': [0.8, 0.2, 0.2, 1.0],         # 深红
    'leftHandIndex1': [0.2, 0.8, 0.2, 1.0],   # 深绿
    'rightHandIndex1': [0.2, 0.2, 0.8, 1.0],  # 深蓝
    'leftForeArm': [0.8, 0.8, 0.2, 1.0],      # 深黄
    'rightForeArm': [0.8, 0.2, 0.8, 1.0],     # 深紫
    'neck': [0.2, 0.8, 0.8, 1.0],             # 深青
    'rightToeBase': [0.8, 0.5, 0.2, 1.0],     # 深橙
    'spine': [0.5, 0.2, 0.8, 1.0],            # 深紫罗兰
    'leftUpLeg': [0.2, 0.5, 0.8, 1.0],        # 深天蓝
    'leftHand': [0.8, 0.2, 0.5, 1.0],         # 深粉红
    'hips': [0.5, 0.8, 0.2, 1.0]              # 深青绿
}

# 安全地处理材质
if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
    print("删除现有材质...")
    del mesh.visual.material
if hasattr(mesh.visual, 'material_map') and mesh.visual.material_map is not None:
    print("删除现有材质映射...")
    del mesh.visual.material_map

# 创建顶点颜色数组，默认为白色
vertex_colors = np.ones((len(mesh.vertices), 4)) * [1.0, 1.0, 1.0, 1.0]

# 为每个部位分配颜色，添加边界检查
max_vertex_index = len(mesh.vertices) - 1
print(f"最大顶点索引: {max_vertex_index}")

colored_vertices = 0
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
        colored_vertices += len(valid_indices)

print(f"总共为 {colored_vertices} 个顶点分配了颜色")

# 设置网格的顶点颜色
mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
print(f"设置后网格顶点颜色状态: Has vertex_colors = {hasattr(mesh.visual, 'vertex_colors')}")

# 验证颜色设置
if hasattr(mesh.visual, 'vertex_colors'):
    unique_colors = np.unique(mesh.visual.vertex_colors, axis=0)
    print(f"网格中的唯一颜色数量: {len(unique_colors)}")
    print("前5个唯一颜色:")
    for i, color in enumerate(unique_colors[:5]):
        print(f"  {i+1}: {color}")

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

# 方法1: 使用trimesh的PLY导出（应该更好地支持顶点颜色）
print(f"\n正在保存PLY文件...")
ply_path = "/data/lixinag/project/GarmentCode/A_RESULT/colored_body.ply"
mesh.export(ply_path)
print(f"PLY文件已保存: {ply_path}")

# 方法2: 手动导出带顶点颜色的OBJ文件
print(f"\n正在手动导出带顶点颜色的OBJ文件...")
obj_path_with_colors = "/data/lixinag/project/GarmentCode/A_RESULT/colored_body_with_colors.obj"

with open(obj_path_with_colors, 'w') as f:
    f.write("# Colored body mesh with vertex colors\n")
    f.write(f"# Generated from {obj_path}\n\n")
    
    # 写入顶点
    for i, vertex in enumerate(mesh.vertices):
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            color = mesh.visual.vertex_colors[i] / 255.0  # 转换回0-1范围
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n")
        else:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
    
    # 写入面
    for face in mesh.faces:
        # OBJ文件索引从1开始
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

print(f"手动导出的OBJ文件已保存: {obj_path_with_colors}")

# 方法3: 使用trimesh的标准OBJ导出
print(f"\n正在使用trimesh导出标准OBJ文件...")
standard_obj_path = "/data/lixinag/project/GarmentCode/A_RESULT/colored_body_standard.obj"
mesh.export(standard_obj_path)
print(f"标准OBJ文件已保存: {standard_obj_path}")

# 验证所有保存的文件
files_to_check = [
    (ply_path, "PLY"),
    (obj_path_with_colors, "手动OBJ"),
    (standard_obj_path, "标准OBJ")
]

for file_path, file_type in files_to_check:
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f"\n{file_type}文件验证:")
        print(f"  文件大小: {file_size} 字节")
        
        try:
            test_mesh = trimesh.load(file_path)
            print(f"  顶点数: {test_mesh.vertices.shape[0]}")
            print(f"  面数: {test_mesh.faces.shape[0]}")
            print(f"  有顶点颜色: {hasattr(test_mesh.visual, 'vertex_colors')}")
            
            if hasattr(test_mesh.visual, 'vertex_colors') and test_mesh.visual.vertex_colors is not None:
                unique_colors_test = np.unique(test_mesh.visual.vertex_colors, axis=0)
                print(f"  唯一颜色数量: {len(unique_colors_test)}")
                if len(unique_colors_test) > 1:
                    print(f"  颜色保存成功！")
                else:
                    print(f"  颜色保存失败（只有一种颜色）")
            else:
                print(f"  没有顶点颜色信息")
                
        except Exception as e:
            print(f"  验证失败: {e}")
    else:
        print(f"\n{file_type}文件不存在")

print("\n处理完成！")
print("建议:")
print("1. 如果OBJ文件中的颜色显示不正确，请使用PLY格式文件")
print("2. 某些3D查看器可能不支持OBJ文件中的顶点颜色")
print("3. 可以尝试使用支持顶点颜色的专业3D软件查看PLY文件")
