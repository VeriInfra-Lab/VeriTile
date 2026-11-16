"""Whole-tile independence, exact precision scope and incomplete-evidence gates."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from scripts import two_gates_dot as dot
from scripts import two_gates_statistics as stats
from scripts import export_dot_rules as export


class DotGateTests(unittest.TestCase):
    def test_whole_tiles_are_independent_samples_not_1024_output_lanes(self):
        obs=np.array([[1.,2.,3.,1.],[2.,3.,4.,1.],[6.,4.,5.,1.]])
        row=dot.summarize(obs)
        result=stats.evaluate_parts(row[None,:],[0],[3],[1.],dot.plan()['settings'])
        self.assertEqual(result['sampling']['valid'],3)
        self.assertAlmostEqual(result['bias']['standard_error'],obs[:,0].std(ddof=1)/np.sqrt(3))

    def test_invalid_groups_do_not_contribute_to_moments_or_peaks(self):
        obs=np.array([[1.,2.,3.,1.],[0.,0.,0.,0.],[3.,1.,2.,1.]])
        row=dot.summarize(obs)
        self.assertEqual(row[0],2);self.assertEqual(row[1],2.)
        bad=obs.copy();bad[1,0]=1.
        with self.assertRaises(ValueError): dot.summarize(bad)

    def test_ordered_oracle_and_zero_subnormal_power_boundary_ulps(self):
        a=np.zeros((3,32,32),np.float32);b=a.copy();c=a.copy()
        c[1].fill(np.nextafter(np.float32(0),np.float32(1)));c[2].fill(2.)
        oracle=c.astype(np.float64)
        ulp=np.full(a.shape,2.**-149);ulp[2].fill(2.**-22)
        observed=dot.cpu_observations(a,b,c,c,c,oracle,ulp)
        self.assertTrue((observed[:,:3]==0).all())
        ulp[0].fill(0.)
        with self.assertRaisesRegex(ValueError,'ULP'):dot.cpu_observations(a,b,c,c,c,oracle,ulp)

    def test_signed_zero_infinity_and_nan_have_distinct_classifications(self):
        x=np.array([0.,-0.,np.inf,-np.inf,np.nan,1.],np.float32)
        self.assertEqual(len(set(dot.classifications(x))),6)

    def test_real_rn_add_is_required_and_cannot_be_replaced_by_fused_mma(self):
        ptx={'reference-dot':'mma.sync .tf32.tf32.','candidate':'mma.sync .tf32.tf32.',
             'reference-add':'add.rn.f32 %r17, %r1, %r9;'}
        dot.check_lowerings(ptx)
        with self.assertRaises(ValueError):dot.check_lowerings({**ptx,'reference-add':'mma.sync .tf32.tf32.'})

    def fixture(self,root):
        p=dot.plan(smoke=True);m={'format':dot.FORMAT,**p,'backend':{},'sources':dot.sources()}
        for name in m['sources']:
            dest=root/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((dot.ROOT/name).read_bytes())
        rows=[];audits=[]
        for j,n in enumerate(dot.part_counts(m)):
            x=np.zeros((n,32,32),np.float32);o=x.astype(np.float64);u=np.full(x.shape,2.**-149)
            obs=np.zeros((n,4));obs[:,3]=1.
            name=f'audit-{j:06d}.npz'
            np.savez_compressed(root/name,observations=obs,indices=np.arange(n),a=x,b=x,acc=x,p=x,q=x,oracle=o,ulp=u)
            rows.append(dot.summarize(obs));audits.append({'part':j,'file':name})
        rows=np.array(rows);drawn=np.array(dot.part_counts(m),dtype=np.int64)
        np.savez_compressed(root/'statistics.npz',rows=rows,drawn=drawn)
        (root/'fingerprints.json').write_text(json.dumps(['0'*64]*len(rows)))
        special=np.zeros((1,32,32),np.float32)
        np.savez_compressed(root/'classifications.npz',inputs=np.zeros((1,3,32,32),np.float32),reference=special,candidate=special)
        for name,text in [('reference-dot','mma.sync.aligned.tf32.tf32.f32'),('reference-add','add.rn.f32'),
                          ('candidate','mma.sync.aligned.tf32.tf32.f32'),('observer-audit','fixture')]:
            (root/(name+'.ptx')).write_text(text)
        record={'state':'COMPLETE','manifest_sha256':dot.digest(m),'audits':audits,
                'files':{f.name:dot.file_hash(f) for f in root.iterdir() if f.is_file()},
                'special':{'classification_mismatches':0,'nan_payload_differences':0},
                'result':stats.evaluate_parts(rows,np.zeros(len(rows),np.int8),drawn,[1.],p['settings'],True)}
        (root/'manifest.json').write_text(json.dumps(m));(root/'record.json').write_text(json.dumps(record))
        return m,record

    def test_complete_smoke_is_replayed_but_never_admitted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root);r=dot.replay(root)
            self.assertEqual(r['result']['decision'],'SMOKE_ONLY')
            self.assertFalse(r['all_words_admission_eligible'])

    def test_missing_audits_lowerings_changed_scope_and_hashes_fail_closed(self):
        for fault in ['audit','precision','source','ptx','hash','budget']:
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);m,r=self.fixture(root)
                if fault=='audit':r['audits'].pop()
                elif fault=='precision':m['profile']['input']='bf16'
                elif fault=='source':m['sources'].pop(next(iter(m['sources'])))
                elif fault=='ptx':r['files'].pop('reference-add.ptx')
                elif fault=='hash':(root/'statistics.npz').write_bytes(b'changed')
                elif fault=='budget':m['settings']['samples']=34
                r['manifest_sha256']=dot.digest(m)
                (root/'manifest.json').write_text(json.dumps(m));(root/'record.json').write_text(json.dumps(r))
                with self.assertRaises((ValueError,KeyError)):dot.replay(root)


class DotExportTests(unittest.TestCase):
    def receipt(self,mean=0.,blocked=False):
        p=dot.plan();n=2**32
        m={'format':dot.FORMAT,**p,'backend':{'capability':[8,9],'triton':'3.5.1'},'sources':dot.sources()}
        row=np.array([[n,mean,1.,1.,1.,n,n,abs(mean),n,1.,1.,n]])
        result=stats.evaluate_parts(row,[0],[n],[1.],p['settings'])
        report={'complete':True,'rule':'DOT-ADD-TO-DOTACC','manifest_sha256':dot.digest(m),
                'source_hashes':m['sources'],'backend':m['backend'],'result':result,
                'special':{'classification_mismatches':int(blocked),'nan_payload_differences':0},
                'all_words_admission_eligible':result['decision']=='ACCEPT' and not blocked,
                'raw_audits_verified':len(dot.audit_indices(m)),
                'part_statistics_verified':len(dot.part_counts(m)),'lowerings':{'candidate.ptx':'0'*64}}
        return {'kind':'dot_contraction_receipt','manifest':m,'report':report,'instance_key':export.instance_key(m,report)}

    def test_rejection_is_preserved_with_empty_admission_list(self):
        r=self.receipt(mean=.1);text=export.render(r)
        self.assertFalse(export.validate(r));self.assertIn('bias := .fail',text)
        self.assertIn(':= []',text);self.assertIn('theorem policy_rejects',text)
        self.assertNotIn('theorem admit',text)

    def test_passing_data_still_requires_external_validation_for_exact_source(self):
        text=export.render(self.receipt())
        self.assertIn('DotContraction.entry contract evidence',text)
        self.assertIn('Spec.EvidenceValidated candidate.rule candidate.evidence',text)
        self.assertNotIn('axiom ',text);self.assertNotIn('sorry',text)

    def test_special_value_mismatch_cannot_enter_the_all_word_admitted_set(self):
        r=self.receipt(blocked=True)
        self.assertFalse(export.validate(r));self.assertNotIn('theorem admit',export.render(r))

    def test_rekeying_changed_precision_smoke_and_forged_decisions_does_not_admit(self):
        for fault in ['precision','smoke','decision','se','gate','budget','audit','key']:
            with self.subTest(fault=fault):
                r=self.receipt(mean=.1);m=r['manifest'];out=r['report']
                if fault=='precision':m['profile']['input']='bf16'
                elif fault=='smoke':m['smoke']=True
                elif fault=='decision':out['result']['decision']='ACCEPT';out['all_words_admission_eligible']=True
                elif fault=='se':out['result']['bias']['standard_error']=0.
                elif fault=='gate':out['result']['bias']['status']='PASS'
                elif fault=='budget':out['result']['sampling']['peak_drawn']-=1
                elif fault=='audit':out['raw_audits_verified']-=1
                out['manifest_sha256']=dot.digest(m);r['instance_key']=export.instance_key(m,out)
                if fault=='key':r['instance_key']='0'*64
                with self.assertRaises(ValueError):export.validate(r)


if __name__=='__main__':unittest.main()
