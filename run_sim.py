import json
import os
import pygarment.data_config as data_config
from pathlib import Path
from pygarment.meshgen.sim_config import PathCofig
from pygarment.meshgen.boxmeshgen import BoxMesh
from pygarment.meshgen.simulation import run_sim
from tqdm import tqdm

PROJECT_ROOT = '/data/lixinag/project/GarmentCode'
DATA_ROOT = f'{PROJECT_ROOT}/A_DATA/GarmentCodeData'
SPLIT_FILE_PATH = f'{DATA_ROOT}/train_test_split.json'
SIM_CONFIG_PATH = f'{PROJECT_ROOT}/assets/Sim_props/default_sim_props_modified.yaml'
OUTPUT_PATH = f'{PROJECT_ROOT}/simulation_output/'
BODY_DEFAULT_DIR = f'{PROJECT_ROOT}/A_RESULT/bodies'
BODY_NAME = 'SMPL_FEMALE_A_POSE_110_ORI'


def get_input_file_path_pool(split_file_path, end_with='_specification.json'):
    input_file_path_pool = []
    with open(split_file_path, 'r') as f:
        data = json.load(f)
        for key_name in data.keys():
            for file_path in data[key_name]:
                file_id = file_path.split('/')[-1]
                input_file_path_pool.append(Path(os.path.join(file_path, file_id + end_with)))
    return input_file_path_pool


if __name__ == "__main__":
    input_file_path_pool = get_input_file_path_pool(SPLIT_FILE_PATH)
    props = data_config.Properties(SIM_CONFIG_PATH)
    props.set_section_stats('sim', fails={}, sim_time={}, spf={}, fin_frame={}, body_collisions={}, self_collisions={})
    props.set_section_stats('render', render_time={})


    for spec_path in tqdm(input_file_path_pool):

        if os.path.exists(f'{OUTPUT_PATH}/{spec_path}'):
            continue

        with open(f'{PROJECT_ROOT}/record.txt', 'a') as f:
            f.write(f'{spec_path}\n')
        
        garment_name, _, _ = spec_path.stem.rpartition('_')
        paths = PathCofig(
            data_root_path=DATA_ROOT,
            in_element_path=spec_path.parent,  
            out_path=OUTPUT_PATH, 
            in_name=garment_name,
            out_name=spec_path.parent,
            body_default_dir=BODY_DEFAULT_DIR,
            body_name=BODY_NAME,
            smpl_body=True)

        garment_box_mesh = BoxMesh(paths.in_g_spec, props['sim']['config']['resolution_scale'])
        garment_box_mesh.load()
        garment_box_mesh.serialize(
            paths, store_panels=False, uv_config=props['render']['config']['uv_texture'])

        props.serialize(paths.element_sim_props)

        run_sim(
            garment_box_mesh.name, 
            props, 
            paths,
            save_v_norms=False,
            store_usd=False,  # NOTE: False for fast simulation!
            optimize_storage=False,
            verbose=False
        )
