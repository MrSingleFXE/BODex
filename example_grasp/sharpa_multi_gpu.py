"""Shard the full tabletop scene list across GPUs using the V3/S34 seed."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--scene-limit',type=int)
    p.add_argument('--parallel-worlds',type=int,default=20)
    a=p.parse_args(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load((ROOT/'src/curobo/content/configs/manip/sharpa_right/tabletop.yml').read_text())
    scenes=sorted((ROOT/'src/curobo/content/assets/object/DGN_2k/scene_cfg').glob('**/tabletop_ur10e/*.npy'))
    count=len(scenes) if a.scene_limit is None else min(a.scene_limit,len(scenes))
    if count < 1: raise ValueError('No scenes; install DGN_2k object assets first')
    gpus=a.gpus.split(','); children=[]; logs=[]
    try:
        for i,gpu in enumerate(gpus):
            start,end=count*i//len(gpus),count*(i+1)//len(gpus)
            if start==end: continue
            cfg['world'].update(start=start,end=end)
            config=out/f'gpu_{gpu}.yml'; config.write_text(yaml.safe_dump(cfg,sort_keys=False))
            log=(out/f'gpu_{gpu}.log').open('w'); logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,SPHERE_SEED_AUDIT=str(out/f'seed_gpu_{gpu}.jsonl'),SPHERE_EXPECTED_COUNT='34')
            children.append(subprocess.Popen([sys.executable,str(ROOT/'example_grasp/sharpa_synthesis.py'),'-c',str(config),'-f',str(out),'-m','npy','-w',str(a.parallel_worlds)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
        codes=[child.wait() for child in children]
        if any(codes): raise RuntimeError(f'Synthesis workers failed: {codes}')
        print(f'Workers completed for {count} scenes. Run Bench coverage checks before training.')
    finally:
        for child in children:
            if child.poll() is None: child.terminate()
        for log in logs: log.close()


if __name__ == '__main__': main()
