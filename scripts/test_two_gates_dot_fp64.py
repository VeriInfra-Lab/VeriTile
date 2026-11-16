"""FP64 precision, exact source-reference audits and fixed-budget evidence guards."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from scripts import two_gates_dot as fp32
from scripts import two_gates_dot_fp64 as fp64
from scripts import two_gates_statistics as stats
from scripts import export_dot_fp64_report as export


class FP64DotTests(unittest.TestCase):
    def test_precision_is_independent_of_fp32_and_only_full_budget_is_formal(self):
        p=fp64.plan();q=fp32.plan()
        self.assertEqual(p['settings'],q['settings'])
        self.assertEqual(p['seeds'],q['seeds'])
        self.assertEqual(p['profile']['distribution'],q['profile']['distribution'])
        self.assertEqual(p['profile']['input'],'fp64')
        self.assertEqual(q['profile']['input'],'fp32')
        self.assertFalse(p['diagnostic'])
        self.assertTrue(fp64.plan(samples=2**18)['diagnostic'])
        self.assertTrue(fp64.plan(smoke=True)['diagnostic'])

    def test_exact_rational_reference_checks_low_components_and_fp64_ulps(self):
        a=np.zeros((3,32,32),np.float64);b=a.copy();c=a.copy()
        c[0].fill(2.);c[1].fill(-2.);c[2].fill(np.nextafter(np.float64(0),np.float64(1)))
        ulp=np.empty_like(c);ulp[:2].fill(2.**-51);ulp[2].fill(2.**-1074)
        lo=np.zeros_like(c)
        obs,error=fp64.cpu_observations(a,b,c,c,c,c,lo,ulp)
        self.assertEqual(error,0.)
        self.assertTrue((obs[:,:3]==0).all())
        bad=lo.copy();bad[0,0,0]=2.**-52
        with self.assertRaisesRegex(ValueError,'oracle'):
            fp64.cpu_observations(a,b,c,c,c,c,bad,ulp)
        bad=ulp.copy();bad[0,0,0]=2.**-22
        with self.assertRaisesRegex(ValueError,'ULP'):
            fp64.cpu_observations(a,b,c,c,c,c,lo,bad)

    def test_fp32_arrays_cannot_be_relabelled_fp64(self):
        a=np.zeros((1,32,32),np.float64)
        with self.assertRaisesRegex(ValueError,'dtype'):
            fp64.cpu_observations(a.astype(np.float32),a,a,a,a,a,a,a)

    def test_fp64_special_diagnostics_preserve_signed_zero_and_tiny_finite_values(self):
        x=np.array([0.,-0.,np.inf,-np.inf,np.nan,2.**-1000],np.float64)
        self.assertEqual(len(set(fp64.classifications(x))),6)
        p=np.zeros((1,32,32),np.float64);q=-p
        self.assertEqual(np.count_nonzero(fp64.classifications(p)!=fp64.classifications(q)),1024)

    def test_source_reference_has_exact_products_and_not_just_a_rounded_dot(self):
        a=np.zeros((1,32,32),np.float64);b=a.copy();c=a.copy()
        a[:,:,0]=1.+2.**-27;b[:,0,:]=1.-2.**-27
        hi=np.ones_like(a);lo=np.full_like(a,-2.**-54);ulp=np.full_like(a,2.**-52)
        observed,error=fp64.cpu_observations(a,b,c,hi,hi,hi,lo,ulp)
        self.assertEqual(error,0.)
        self.assertEqual(observed[0,1],2.**-54)
        with self.assertRaisesRegex(ValueError,'oracle'):
            fp64.cpu_observations(a,b,c,hi,hi,hi,np.zeros_like(a),ulp)

    def fixture(self,root):
        p=fp64.plan(smoke=True)
        manifest={'format':fp64.FORMAT,**p,'backend':{},'sources':fp64.sources()}
        for name in manifest['sources']:
            dest=root/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((fp64.ROOT/name).read_bytes())
        rows=[];audits=[]
        for j,n in enumerate(fp32.part_counts(manifest)):
            x=np.zeros((n,32,32),np.float64);ulp=np.full_like(x,2.**-1074)
            obs=np.zeros((n,4));obs[:,3]=1.
            name=f'audit-{j:06d}.npz'
            np.savez_compressed(root/name,observations=obs,indices=np.arange(n),
                                a=x,b=x,acc=x,p=x,q=x,hi=x,lo=x,ulp=ulp)
            rows.append(fp32.summarize(obs));audits.append({'part':j,'file':name})
        rows=np.array(rows);drawn=np.array(fp32.part_counts(manifest),np.int64)
        np.savez_compressed(root/'statistics.npz',rows=rows,drawn=drawn)
        (root/'fingerprints.json').write_text(json.dumps(['0'*64]*len(rows)))
        (root/'compiled.json').write_text(json.dumps({'synthetic_fixture':'0'*64}))
        x=np.zeros((1,32,32),np.float64)
        np.savez_compressed(root/'classifications.npz',inputs=np.zeros((1,3,32,32),np.float64),reference=x,candidate=x)
        record={'state':'COMPLETE','manifest_sha256':fp32.digest(manifest),'audits':audits,
                'files':{f.name:fp32.file_hash(f) for f in root.iterdir() if f.is_file()},
                'special':{'classification_mismatches':0,'nan_payload_differences':0},
                'result':stats.evaluate_parts(rows,np.zeros(len(rows),np.int8),drawn,[1.],p['settings'],True)}
        (root/'manifest.json').write_text(json.dumps(manifest));(root/'record.json').write_text(json.dumps(record))
        return manifest,record

    def test_smoke_replays_but_cannot_enter_formal_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root);report=fp64.replay(root)
            self.assertEqual(report['result']['decision'],'SMOKE_ONLY')
            self.assertFalse(report['all_words_admission_eligible'])
            self.assertEqual(report['exact_rational_output_audits'],33*1024)

    def test_rekeyed_changed_scope_missing_audits_and_forged_gates_are_rejected(self):
        for fault in ['precision','budget','audit','file','source','decision']:
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);m,r=self.fixture(root)
                if fault=='precision':m['profile']['input']='fp32'
                elif fault=='budget':m['settings']['samples']=34
                elif fault=='audit':r['audits'].pop()
                elif fault=='file':r['files'].pop('compiled.json')
                elif fault=='source':m['sources'].pop(next(iter(m['sources'])))
                elif fault=='decision':r['result']['decision']='ACCEPT'
                r['manifest_sha256']=fp32.digest(m)
                (root/'manifest.json').write_text(json.dumps(m));(root/'record.json').write_text(json.dumps(r))
                with self.assertRaises((ValueError,KeyError)):fp64.replay(root)


class FP64ReceiptTests(unittest.TestCase):
    def receipt(self,mean=0.):
        p=fp64.plan();n=2**32
        m={'format':fp64.FORMAT,**p,'backend':{'capability':[8,9],'triton':'3.5.1'},'sources':fp64.sources()}
        row=np.array([[n,mean,1.,1.,1.,n,n,abs(mean),n,1.,1.,n]])
        result=stats.evaluate_parts(row,[0],[n],[1.],p['settings'])
        report={'complete':True,'precision':'fp64','rule':'DOT-ADD-TO-DOTACC',
                'manifest_sha256':fp32.digest(m),'source_hashes':m['sources'],'backend':m['backend'],
                'result':result,'special':{'classification_mismatches':0,'nan_payload_differences':0},
                'all_words_admission_eligible':result['decision']=='ACCEPT',
                'raw_audits_verified':len(fp32.audit_indices(m)),
                'part_statistics_verified':len(fp32.part_counts(m)),
                'exact_rational_output_audits':len(fp32.audit_indices(m))*32*1024,
                'max_audited_reference_error_ulp':0.}
        return {'kind':'dot_contraction_fp64_receipt','manifest':m,'report':report}

    def test_passing_fp64_receipt_and_rejection_both_keep_the_actual_result(self):
        self.assertTrue(export.validate(self.receipt()))
        self.assertFalse(export.validate(self.receipt(.1)))

    def test_changed_precision_smoke_budget_reference_or_forged_gates_fail_closed(self):
        for fault in ['precision','smoke','budget','oracle','se','decision','gate','audit']:
            with self.subTest(fault=fault):
                r=self.receipt(.1);m=r['manifest'];out=r['report']
                if fault=='precision':out['precision']='fp32'
                elif fault=='smoke':m['smoke']=True
                elif fault=='budget':out['result']['sampling']['peak_drawn']-=1
                elif fault=='oracle':out['max_audited_reference_error_ulp']=1.
                elif fault=='se':out['result']['bias']['standard_error']=0.
                elif fault=='decision':out['result']['decision']='ACCEPT'
                elif fault=='gate':out['result']['bias']['status']='PASS'
                elif fault=='audit':out['exact_rational_output_audits']-=1
                out['manifest_sha256']=fp32.digest(m)
                with self.assertRaises(ValueError):export.validate(r)


if __name__=='__main__':unittest.main()
