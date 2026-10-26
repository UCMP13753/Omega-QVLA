"""CPU tests for scientific controls, strict runtime dispatch, plans and reporting.
Run: python -m unittest discover -s tests/rebuttal -v
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.rebuttal.plan import make_plan
from tools.rebuttal.report import summarize


class PlanTests(unittest.TestCase):
    def test_full_core_is_paired_and_acyclic(self):
        plan=make_plan('/tmp/run',{'checkpoints':'/tmp/ckpt','pi_checkpoint':'/tmp/pi'})
        jobs=plan['jobs']
        self.assertEqual(sum(j['stage']=='capture' for j in jobs),24)
        self.assertEqual(sum(j['stage']=='evaluate' for j in jobs),168)
        self.assertEqual(sum(j['stage']=='build' for j in jobs),144)
        self.assertEqual(sum(j['stage']=='profile' for j in jobs),14)
        for job in jobs:
            self.assertTrue(all(i<job['index'] for i in job['dependencies']))
        for model in ('groot','pi05'):
            a=next(j for j in jobs if j['stage']=='build' and j['config']['model']==model and j['config']['variant']=='hadamard')
            b=next(j for j in jobs if j['stage']=='build' and j['config']['model']==model and j['config']['variant']=='svd_hadamard')
            self.assertEqual(a['capture'],b['capture'])
            ca=dict(a['config']);cb=dict(b['config'])
            for cfg in (ca,cb):
                cfg.pop('variant');cfg.pop('rotation')
            self.assertEqual(ca,cb)

    def test_step_sensitivity_recalibrates_and_has_fp_controls(self):
        plan=make_plan('/tmp/run',{},models=('pi05',),suites=('long',),seeds=(42,),extras=True)
        builds={j['config']['variant']:j for j in plan['jobs'] if j['stage']=='build'}
        self.assertNotEqual(builds['svdh_steps4']['capture'],builds['svdh_steps10']['capture'])
        evaluations={j['config']['variant'] for j in plan['jobs'] if j['stage']=='evaluate'}
        self.assertTrue({'fp16_steps4','fp16_steps10'}<=evaluations)


class ReportTests(unittest.TestCase):
    def test_denominators_pairing_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            plan=make_plan(d,{},models=('groot',),suites=('long',),seeds=(42,),trials=1,smoke=True)
            for job in plan['jobs']:
                if job['stage']!='evaluate' or job['config']['variant']=='fp16':
                    continue
                path=Path(d)/job['output'];path.parent.mkdir(parents=True,exist_ok=True)
                good=job['config']['variant']=='svd_hadamard'
                eps=[{'task_id':i,'init_index':20,'seed':42+i*1000,'success':good,'error':None} for i in range(10)]
                path.write_text(json.dumps({'config':job['config'],'complete':True,'episodes':eps,'total_successes':sum(e['success'] for e in eps),'total_episodes':10}))
                status=Path(d)/'status';status.mkdir(exist_ok=True)
                (status/f'{job["id"]}.done.json').write_text('{}')
            result=summarize(plan)
            self.assertEqual(len(result['missing_jobs']),1)
            self.assertEqual(result['paired'][0]['saved'],10)
            self.assertEqual(result['paired'][0]['spoiled'],0)
            self.assertEqual(result['four_suite_macro'],[])
            row=next(r for r in result['rows'] if r['variant']=='fp16')
            self.assertIsNone(row['mean'])
            self.assertIsNone(row['std'])


try:
    import torch
    from tools.rebuttal.core import make_rotation, block_transform
    from tools.rebuttal.runtime import RebuttalLinear
    from gr00t.quantization.dit_step_context import set_dit_quant_step, get_current_dit_step
    HAS_TORCH=True
except ImportError:
    HAS_TORCH=False


@unittest.skipUnless(HAS_TORCH,'PyTorch not installed')
class NumericalTests(unittest.TestCase):
    def test_all_rotations_preserve_linear_map_and_hadamard_never_uses_svd(self):
        torch.manual_seed(7)
        w=torch.randn(12,16);x=torch.randn(7,16)
        for mode in ('none','svd','hadamard','svd_hadamard'):
            b,p=make_rotation(w,mode,8)
            torch.testing.assert_close(block_transform(x,b,p)@block_transform(w,b,p).T,x@w.T,rtol=1e-5,atol=1e-5)
        with patch('gr00t.quantization.duquant_preprocess.compute_block_rotation',side_effect=AssertionError('Hidden SVD')):
            make_rotation(w,'hadamard',8)
        with self.assertRaises(ValueError):
            make_rotation(w,'hadamard',7)

    def test_runtime_no_quant_equivalence_with_bias(self):
        base=torch.nn.Linear(16,12)
        b,p=make_rotation(base.weight,'svd_hadamard',8)
        rec={'weight_q':block_transform(base.weight,b,p).detach(),'blocks':b,'perm':p,
             'scales':torch.empty(0,16),'side':'backbone'}
        cfg={'abits':0,'steps':8,'scaling':'perstep'}
        linear=RebuttalLinear(base,rec,cfg,'test')
        x=torch.randn(5,16)
        torch.testing.assert_close(linear(x),base(x),rtol=1e-5,atol=1e-5)

    def test_head_requires_valid_step_context_and_uses_requested_scale(self):
        base=torch.nn.Linear(4,2,bias=False)
        b,p=make_rotation(base.weight,'none',4,False)
        rec={'weight_q':base.weight.detach(),'blocks':b,'perm':p,
             'scales':torch.tensor([[.1]*4,[.3]*4]),'side':'head'}
        cfg={'abits':4,'steps':2,'scaling':'perstep'}
        linear=RebuttalLinear(base,rec,cfg,'head')
        x=torch.tensor([[.16,.26,.36,.46]])
        with self.assertRaises(RuntimeError):
            linear(x)
        for step in (0,1):
            with set_dit_quant_step(step,total=2):
                scale=rec['scales'][step]
                expected=torch.nn.functional.linear(torch.round(x/scale).clamp(-7,7)*scale,base.weight)
                torch.testing.assert_close(linear(x),expected)
        with set_dit_quant_step(0,total=8):
            with self.assertRaises(RuntimeError):
                linear(x)

    def test_pack_build_and_runtime_install_with_shared_capture(self):
        from tools.rebuttal.worker import build
        from tools.rebuttal.runtime import install
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/'capture';source.mkdir()
            model=torch.nn.Sequential(torch.nn.Linear(8,6,bias=False))
            torch.save({'name':'0','side':'head','weight':model[0].weight.detach(),
                        'x':{0:torch.randn(5,8),1:torch.randn(5,8)}},source/'0.pt')
            (source/'index.json').write_text(json.dumps({'layers':[{'name':'0','file':'0.pt','side':'head'}],
                'inventory':[{'name':'0','weight_bytes':192}],'parameter_bytes':192}))
            cfg={'model':'groot','scope':'both','rotation':'hadamard','block':4,'permute':True,
                 'wbits':4,'abits':4,'steps':2,'scaling':'perstep','solver':'gptq','head_solver':'rtn'}
            job={'config':cfg,'capture':'capture','output':'pack.pt'}
            with patch.object(torch.Tensor,'cuda',lambda tensor,*a,**kw:tensor):
                build(job,{},root)
            pack=torch.load(root/'pack.pt',weights_only=False)
            self.assertEqual(install(model,pack,cfg),['0'])
            with set_dit_quant_step(1,total=2):
                self.assertTrue(torch.isfinite(model(torch.randn(2,8))).all())
            inventory=json.loads((root/'pack.inventory.json').read_text())
            self.assertEqual(inventory['target_layers'],1)
            self.assertIn('no native INT4',inventory['execution'])
            self.assertEqual(inventory['unquantized_parameter_bytes'],0)

    def test_openpi_context_resets_for_every_inference_and_policy_cache(self):
        from tools.rebuttal.pi05_context import configure_policy
        seen=[]
        class Model:
            def denoise_step(self):
                seen.append(get_current_dit_step())
            def sample_actions(self,*args,num_steps=10,**kwargs):
                for _ in range(num_steps):
                    self.denoise_step()
                return 1
        class Policy:
            def __init__(self):
                self._model=Model();self._sample_actions=self._model.sample_actions;self._sample_kwargs={}
        policy=Policy()
        configure_policy(policy,4)
        policy._sample_actions();policy._sample_actions()
        self.assertEqual(seen,[0,1,2,3,0,1,2,3])
        self.assertIsNone(get_current_dit_step())

if __name__=='__main__':
    unittest.main()
