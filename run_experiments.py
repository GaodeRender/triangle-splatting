from src.diff_recon import VanillaTSTrainer, loadConfig, run_exp_with_args
import os
import argparse
import torch


def exp(config_path: str, dataset_path: str, scene_id: str, resolution: int, device: int, point_count: int = None):
    config = loadConfig(config_path)
    config.dataset.local_dir = dataset_path
    config.dataset.scene_id = scene_id
    config.dataset.train_target_res = resolution
    config.dataset.test_target_res = resolution
    if point_count is not None:
        config.model.model_update.contribution_pruning.target_point_num = point_count
    trainer = VanillaTSTrainer(config, exp_name=scene_id, device=device)
    trainer.train()


def train_MipNerf360_VanillaTS(dataset_path: str, num_workers: int):
    config_path = "config/MipNerf360_VanillaTS.yaml"
    device_count = min(torch.cuda.device_count(), num_workers) if num_workers > 0 else 1

    args_list = []
    outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
    indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
    for i, scene in enumerate(outdoor_scenes):
        device = i % device_count
        args_list.append((config_path, dataset_path, scene, 4, device))
    for i, scene in enumerate(indoor_scenes):
        device = (i + len(outdoor_scenes)) % device_count
        args_list.append((config_path, dataset_path, scene, 2, device))

    run_exp_with_args(exp, args_list, num_workers=num_workers)


def train_NerfSynthetic_VanillaTS(dataset_path: str, num_workers: int):
    config_path = "config/NerfSynthetic_VanillaTS.yaml"
    device_count = min(torch.cuda.device_count(), num_workers) if num_workers > 0 else 1

    args_list = []
    scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
    for i, scene in enumerate(scenes):
        device = i % device_count
        args_list.append((config_path, dataset_path, scene, 1, device))

    run_exp_with_args(exp, args_list, num_workers=num_workers)


def train_NerfSynthetic_VanillaTS_mesh(dataset_path: str, num_workers: int):
    config_path = "config/NerfSynthetic_VanillaTS_mesh.yaml"
    device_count = min(torch.cuda.device_count(), num_workers) if num_workers > 0 else 1

    args_list = []
    scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
    point_counts = [89, 82, 41, 58, 112, 78, 83, 93]
    for i, (scene, point_count) in enumerate(zip(scenes, point_counts)):
        device = i % device_count
        args_list.append((config_path, dataset_path, scene, 1, device, point_count*1000))

    run_exp_with_args(exp, args_list, num_workers=num_workers)


def train_TanksAndBlending_VanillaTS(dataset_path: str, num_workers: int):
    config_path = "config/TanksAndBlending_VanillaTS.yaml"
    device_count = min(torch.cuda.device_count(), num_workers) if num_workers > 0 else 1

    args_list = []
    scenes = ["tandt/truck", "tandt/train", "db/drjohnson", "db/playroom"]
    for i, scene in enumerate(scenes):
        device = i % device_count
        args_list.append((config_path, dataset_path, scene, 1, device))

    run_exp_with_args(exp, args_list, num_workers=num_workers)


def train_MatrixCity_VanillaTS_mesh(dataset_path: str, num_workers: int):
    config_path = "config/MatrixCity_VanillaTS_mesh.yaml"
    device_count = min(torch.cuda.device_count(), num_workers) if num_workers > 0 else 1

    args_list = []
    scenes = ["small_city/aerial"]
    for i, scene in enumerate(scenes):
        device = i % device_count
        args_list.append((config_path, dataset_path, scene, 1, device))

    run_exp_with_args(exp, args_list, num_workers=num_workers)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, help="One of MipNerf360, NerfSynthetic, TanksAndBlending, NerfSynthetic_mesh, MatrixCity_mesh")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--num_workers", type=int, default=0, help="Number of concurrent training jobs")
    args = parser.parse_args()
    dataset_path = args.dataset_path
    num_workers = args.num_workers

    match args.type:
        case "MipNerf360":
            train_MipNerf360_VanillaTS(dataset_path, num_workers)
        case "NerfSynthetic":
            train_NerfSynthetic_VanillaTS(dataset_path, num_workers)
        case "NerfSynthetic_mesh":
            train_NerfSynthetic_VanillaTS_mesh(dataset_path, num_workers)
        case "TanksAndBlending":
            train_TanksAndBlending_VanillaTS(dataset_path, num_workers)
        case "MatrixCity_mesh":
            train_MatrixCity_VanillaTS_mesh(dataset_path, num_workers)
        case _:
            raise ValueError(f"Unknown type: {args.type}")
