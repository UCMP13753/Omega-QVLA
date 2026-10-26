"""Submit dependency-aware single-GPU jobs. Dry run by default; resumable submission ledger."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--account',required=True)
    p.add_argument('--submit',action='store_true')
    p.add_argument('--retry-failed',action='store_true',help='Skip only done files; resubmit all remaining jobs after checking old jobs have stopped')
    args=p.parse_args()
    if args.account=='YOUR_COMPUTE_CANADA_ACCOUNT':
        p.error('Set your actual Slurm account')
    manifest=Path(args.manifest).resolve()
    data=json.loads(manifest.read_text())
    root=Path(data['root'])
    (root/'slurm').mkdir(exist_ok=True)
    repo=Path(__file__).resolve().parents[2]
    from tools.rebuttal.plan import source_digest
    if source_digest(repo)!=data['source_digest']:
        p.error('Source changed after plan generation; generate a new run directory')
    ledger_path=root/'submitted.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    if args.retry_failed and args.submit and ledger:
        ids=[v['slurm_id'] for v in ledger.values()]
        active=subprocess.check_output(['squeue','--noheader','--jobs',','.join(ids),'--format=%i'],text=True).strip()
        if active:
            p.error('Old jobs are still active; do not duplicate a live run')
        ledger={}
    live=[]
    limit=int(os.environ.get('REBUTTAL_MAX_IN_FLIGHT','8'))
    if limit<1:
        p.error('REBUTTAL_MAX_IN_FLIGHT must be positive')
    for job in data['jobs']:
        if (root/'status'/f'{job["id"]}.done.json').exists():
            continue
        key=str(job['index'])
        if key in ledger:
            live.append(ledger[key]['slurm_id'])
            continue
        afterok=[]
        for index in job['dependencies']:
            dependency=data['jobs'][index]
            if (root/'status'/f'{dependency["id"]}.done.json').exists():
                continue
            if str(index) not in ledger:
                raise RuntimeError(f'Unsubmitted dependency {index}')
            afterok.append(ledger[str(index)]['slurm_id'])
        dependencies=[]
        if afterok:
            dependencies.append('afterok:'+':'.join(afterok))
        # Lane throttle by submit order, afterany so one unrelated failure does not block all work.
        if len(live)>=limit:
            dependencies.append('afterany:'+live[-limit])
        cmd=['sbatch','--parsable','--account='+args.account,
             '--time='+os.environ.get('REBUTTAL_TIME','12:00:00'),
             '--mem='+os.environ.get('REBUTTAL_MEM','128G'),
             '--cpus-per-task='+os.environ.get('REBUTTAL_CPUS','8'),
             '--output='+str(root/'slurm'/f'{job["id"]}-%j.out'),
             '--error='+str(root/'slurm'/f'{job["id"]}-%j.err')]
        if dependencies:
            cmd.append('--dependency='+','.join(dependencies))
        env=os.environ.copy()
        env.update(REBUTTAL_REPO=str(repo),MANIFEST=str(manifest),JOB_INDEX=key)
        cmd += ['--export=ALL',str(repo/'scripts/narval_rebuttal/job.sbatch')]
        if args.submit:
            response=subprocess.check_output(cmd,env=env,text=True).strip()
            slurm_id=response.split(';',1)[0]
            if not re.fullmatch(r'\d+',slurm_id):
                raise RuntimeError(f'Unexpected sbatch response: {response}')
        else:
            slurm_id=f'DRY{job["index"]}'
        ledger[key]={'slurm_id':slurm_id,'stage':job['stage'],'config':job['config']}
        live.append(slurm_id)
        if args.submit:
            temp=ledger_path.with_suffix('.tmp')
            temp.write_text(json.dumps(ledger,indent=2)+'\n')
            temp.replace(ledger_path)
        print(f'{slurm_id} {job["stage"]} {job["config"]["model"]}/{job["config"]["suite"]} '+job['config'].get('variant','capture'),flush=True)
    if not args.submit:
        print('Dry run only; add --submit to call sbatch.')

if __name__=='__main__':
    main()
