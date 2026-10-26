"""Generate an explicit immutable dependency graph; standard-library only."""
import argparse
import hashlib
import json
from pathlib import Path


def source_digest(repo):
    digest=hashlib.sha256()
    # Include imported runtime, model helpers and evaluator, not only new runner files.
    for directory in ('tools','gr00t','examples','scripts'):
        for p in sorted((repo/directory).rglob('*')):
            if p.is_file() and p.suffix in ('.py','.sh','.sbatch'):
                digest.update(str(p.relative_to(repo)).encode())
                digest.update(p.read_bytes())
    return digest.hexdigest()


def make_plan(root,paths,models=('groot','pi05'),suites=('goal','spatial','object','long'),
              seeds=(42,43,44),trials=10,extras=False,smoke=False):
    jobs=[]
    captures={}
    def add(stage,cfg,output,**fields):
        index=len(jobs)
        job={'index':index,'id':f'{index:04}-{stage}','stage':stage,'config':cfg.copy(),'output':output,**fields}
        jobs.append(job)
        return job
    for model in models:
        for suite in suites:
            for seed in seeds:
                base={'model':model,'suite':suite,'cal_seed':seed,'eval_seed':seed,'steps':8,
                      'samples':10,'token_cap':64,'trials':trials,'eval_offset':20,'rotation':'svd_hadamard',
                      'block':64,'permute':True,'scaling':'perstep','scope':'both',
                      'wbits':4,'abits':4,'solver':'gptq','head_solver':'rtn'}
                variants=[('fp16',{}),('hadamard',{'rotation':'hadamard'}),('svd_hadamard',{}),
                          ('backbone_only',{'scope':'backbone'}),('head_only',{'scope':'head'}),
                          ('gptq_w4a16',{'rotation':'none','permute':False,'abits':0,'head_solver':'gptq'}),
                          ('rtn_w4a16',{'rotation':'none','permute':False,'abits':0,'solver':'rtn'})]
                if extras and suite=='long':
                    variants += [('svd_only',{'rotation':'svd'}),('permutation_only',{'rotation':'none'}),
                                 ('pooled_scaling',{'scaling':'pooled'}),('w4a8',{'abits':8}),
                                 ('cal2',{'samples':2}),('block32',{'block':32}),('block128',{'block':128}),
                                 ('fp16_steps4',{'steps':4}),('svdh_steps4',{'steps':4}),
                                 ('fp16_steps10',{'steps':10}),('svdh_steps10',{'steps':10})]
                if smoke:
                    variants=[('fp16',{}),('hadamard',{'rotation':'hadamard'}),('svd_hadamard',{})]
                    base['trials']=1
                    # Enough samples to exercise all suites; fewer input tokens only.
                    base['token_cap']=16
                for label,change in variants:
                    cfg=dict(base,**change)
                    cfg['variant']=label
                    cap_cfg={k:cfg[k] for k in ('model','suite','cal_seed','steps','samples','token_cap')}
                    capkey=json.dumps(cap_cfg,sort_keys=True)
                    if capkey not in captures:
                        capid=f'{model}/{suite}/c{seed}-t{cfg["steps"]}-n{cfg["samples"]}-cap{cfg["token_cap"]}'
                        captures[capkey]=add('capture',cap_cfg,f'captures/{capid}',dependencies=[])
                    cap=captures[capkey]
                    key=f'{model}/{suite}/{label}/c{seed}-e{seed}'
                    pack=None
                    dep=cap
                    if not label.startswith('fp16'):
                        pack=f'packs/{key}.pt'
                        dep=add('build',cfg,pack,capture=cap['output'],dependencies=[cap['index']])
                    add('evaluate',cfg,f'eval/{key}.json',pack=pack,dependencies=[dep['index']])
                    if seed==seeds[0] and suite==suites[-1]:
                        add('profile',cfg,f'profile/{key}.json',pack=pack,dependencies=[dep['index']])
    return {'format':1,'root':str(Path(root).resolve()),'paths':paths,'jobs':jobs,
            'protocol':{'design':'paired variants, independent calibration/evaluation seeds',
                        'calibration':'one reset observation per task; initial-state index 0',
                        'evaluation':'initial-state indices 20..20+trials-1; shared across variants',
                        'output_rotation':'none for every variant; controlled new study, not exact paper reproduction',
                        'backbone_solver':'GPTQ, damping 0.01, solver block 128',
                        'head_solver':'per-output minmax RTN, except explicit GPTQ baseline',
                        'activation':'per-token backbone; per-input-channel head, p99.9',
                        'execution':'fake quantization + floating GEMM, NO native INT4 kernel',
                        'calibration_inputs_are_not_full_trajectories':True}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--checkpoints',required=True)
    p.add_argument('--pi-checkpoint',default='')
    p.add_argument('--models',nargs='+',choices=['groot','pi05'],default=['groot','pi05'])
    p.add_argument('--suites',nargs='+',choices=['goal','spatial','object','long'],default=['goal','spatial','object','long'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--trials',type=int,default=10)
    p.add_argument('--extras',action='store_true')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.trials<1 or args.trials>30:
        p.error('Use 1..30 trials with held-out initial states 20..49')
    if 'pi05' in args.models and not args.pi_checkpoint:
        p.error('--pi-checkpoint is required for pi05')
    for values in (args.models,args.suites,args.seeds):
        if len(values)!=len(set(values)):
            p.error('Duplicate models/suites/seeds are not allowed')
    root=Path(args.root).resolve()
    manifest=root/'manifest.json'
    if manifest.exists():
        p.error('Run manifest already exists; use a new run directory')
    repo=Path(__file__).resolve().parents[2]
    paths={'checkpoints':str(Path(args.checkpoints).resolve()),
           'pi_checkpoint':str(Path(args.pi_checkpoint).resolve()) if args.pi_checkpoint else ''}
    result=make_plan(root,paths,args.models,args.suites,args.seeds,args.trials,args.extras,args.smoke)
    result['source_digest']=source_digest(repo)
    root.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps(result,indent=2)+'\n')
    counts={s:sum(j['stage']==s for j in result['jobs']) for s in ('capture','build','evaluate','profile')}
    print(json.dumps({'manifest':str(manifest),'jobs':counts,
                      'expected_eval_episodes':sum(j['config']['trials']*10 for j in result['jobs'] if j['stage']=='evaluate')},indent=2))

if __name__=='__main__':
    main()
