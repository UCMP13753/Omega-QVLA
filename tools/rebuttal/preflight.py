"""Environment and data checks, to run inside the target model's GPU allocation."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,choices=['groot','pi05'])
    p.add_argument('--checkpoints',required=True)
    p.add_argument('--pi-checkpoint',default='')
    args=p.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('Preflight requires a GPU allocation')
    for module in ('libero.libero.benchmark','robosuite','mujoco','cv2','tyro'):
        importlib.import_module(module)
    from libero.libero import benchmark
    for suite in ('libero_goal','libero_spatial','libero_object','libero_10'):
        tasks=benchmark.get_benchmark_dict()[suite]()
        for i in range(tasks.n_tasks):
            if len(tasks.get_task_init_states(i))<30:
                raise RuntimeError(f'{suite}/{i}: fewer than 30 initial states')
    if args.model=='groot':
        from gr00t.model.policy import Gr00tPolicy
        from examples.Libero.custom_data_config import LiberoDataConfig
        for suite in ('goal','spatial','object','long'):
            folder=Path(args.checkpoints)/f'gr00t-n1.5-libero-{suite}-posttrain'
            if not folder.is_dir() or not (folder/'config.json').exists():
                raise FileNotFoundError(folder)
    else:
        from openpi.policies import policy_config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        import openpi_client
        if not hasattr(PI0Pytorch,'denoise_step'):
            raise RuntimeError('Unsupported OpenPI version: missing denoise_step')
        if not (Path(args.pi_checkpoint)/'model.safetensors').exists():
            raise FileNotFoundError('Need a converted PyTorch pi0.5 checkpoint with model.safetensors')
    print(json.dumps({'status':'imports_and_data_ok','model':args.model,'python':sys.version,
                      'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
                      'execution':'No native INT4 backend in this runner'},indent=2))
    print('Run a smoke manifest next; import checks do not validate full policy execution.')

if __name__=='__main__':
    main()
