"""Summarize complete runs without silently treating missing jobs as zero accuracy."""
import argparse
import collections
import json
import math
from pathlib import Path
import statistics


def wilson(successes,total):
    if not total:
        return None
    z=1.95996398454
    mean=successes/total
    d=1+z*z/total
    c=(mean+z*z/(2*total))/d
    h=z*math.sqrt(mean*(1-mean)/total+z*z/(4*total*total))/d
    return [100*(c-h),100*(c+h)]


def summarize(manifest):
    root=Path(manifest['root'])
    groups=collections.defaultdict(list)
    expected=collections.Counter()
    missing=[]
    runs={}
    for job in manifest['jobs']:
        if job['stage']!='evaluate':
            continue
        cfg=job['config']
        key=(cfg['model'],cfg['suite'],cfg['variant'])
        expected[key]+=1
        path=root/job['output']
        if not path.exists() or not (root/'status'/f'{job["id"]}.done.json').exists():
            missing.append(job['id'])
            continue
        row=json.loads(path.read_text())
        if not row.get('complete') or row['config']!=cfg:
            raise ValueError(f'Incomplete or mismatched result: {path}')
        episodes=row['episodes']
        keys={(e['task_id'],e['init_index']) for e in episodes}
        if len(keys)!=len(episodes) or len(episodes)!=cfg['trials']*10:
            raise ValueError(f'Wrong denominator or duplicate episodes: {path}')
        if any(e.get('error') for e in episodes):
            raise ValueError(f'Implementation error in {path}')
        n=len(episodes); k=sum(e['success'] for e in episodes)
        groups[key].append((cfg['cal_seed'],100*k/n,k,n))
        runs[(cfg['model'],cfg['suite'],cfg['variant'],cfg['cal_seed'])]=row
    rows=[]
    for key,count in sorted(expected.items()):
        data=groups[key]
        vals=[r[1] for r in data]
        k=sum(r[2] for r in data); n=sum(r[3] for r in data)
        rows.append({'model':key[0],'suite':key[1],'variant':key[2],'complete_runs':len(data),'expected_runs':count,
                     'mean':statistics.mean(vals) if vals else None,
                     'std':statistics.stdev(vals) if len(vals)>1 else None,
                     'successes':k,'episodes':n,'wilson95_descriptive':wilson(k,n)})
    pairs=[]
    for (model,suite,variant,seed),row in sorted(runs.items()):
        if variant!='svd_hadamard':
            continue
        for reference in ('hadamard','fp16'):
            other=runs.get((model,suite,reference,seed))
            if not other:
                continue
            a={(e['task_id'],e['init_index'],e['seed']):e['success'] for e in row['episodes']}
            b={(e['task_id'],e['init_index'],e['seed']):e['success'] for e in other['episodes']}
            if a.keys()!=b.keys():
                raise ValueError('Paired comparisons require identical episode identities')
            saved=sum(a[k] and not b[k] for k in a)
            spoiled=sum(b[k] and not a[k] for k in a)
            pairs.append({'model':model,'suite':suite,'cal_seed':seed,'reference':reference,
                          'saved':saved,'spoiled':spoiled,'delta_pp':100*(saved-spoiled)/len(a)})
    # Four-suite macro average per independently calibrated seed, only if all suites exist.
    macro=[]
    for model in sorted({k[0] for k in expected}):
        for variant in sorted({k[2] for k in expected if k[0]==model}):
            seeds=sorted({k[3] for k in runs if k[0]==model and k[2]==variant})
            values=[]
            for seed in seeds:
                four=[runs.get((model,s,variant,seed)) for s in ('goal','spatial','object','long')]
                if all(four):
                    values.append(statistics.mean(100*r['total_successes']/r['total_episodes'] for r in four))
            if values:
                macro.append({'model':model,'variant':variant,'complete_four_suite_runs':len(values),
                              'mean':statistics.mean(values),'std':statistics.stdev(values) if len(values)>1 else None})
    return {'rows':rows,'four_suite_macro':macro,'paired':pairs,'missing_jobs':missing,
            'uncertainty_note':'Std is across independent calibration/evaluation runs. Wilson intervals are descriptive pooled-binomial intervals; episodes share tasks/checkpoints and are not independent replicates.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);args=ap.parse_args()
    manifest=json.loads(Path(args.manifest).read_text())
    result=summarize(manifest)
    root=Path(manifest['root'])
    (root/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# AAAI rebuttal results','',result['uncertainty_note'],'',
           f'Missing evaluation jobs: {len(result["missing_jobs"])}','',
           '| Model | Suite | Variant | Runs | Mean ± std (%) | Successes / episodes |',
           '| --- | --- | --- | --- | --- | --- |']
    for row in result['rows']:
        mean='—' if row['mean'] is None else f'{row["mean"]:.2f}'
        std='—' if row['std'] is None else f'{row["std"]:.2f}'
        lines.append(f'| {row["model"]} | {row["suite"]} | {row["variant"]} | {row["complete_runs"]}/{row["expected_runs"]} | {mean} ± {std} | {row["successes"]}/{row["episodes"]} |')
    lines += ['','## Complete four-suite macro averages','',
              '| Model | Variant | Complete seeds | Mean ± std (%) |','| --- | --- | --- | --- |']
    for row in result['four_suite_macro']:
        std='—' if row['std'] is None else f'{row["std"]:.2f}'
        lines.append(f'| {row["model"]} | {row["variant"]} | {row["complete_four_suite_runs"]} | {row["mean"]:.2f} ± {std} |')
    lines += ['','Paired outcome changes are saved in report.json. Diagnostics, per-layer inventories, latency and traces remain under packs/ and profile/.',
              '', 'Execution is fake quantization followed by floating-point GEMM. No results in this run establish native INT4 acceleration.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(root/'report.md')

if __name__=='__main__':
    main()
