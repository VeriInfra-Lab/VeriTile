"""Pin exact task-local pure definitions without the old program/host imports."""
from pathlib import Path
import argparse,hashlib,json,re
root=Path(__file__).resolve().parents[4]
old=root/'bench/tritonbench_g/chunked_cumsum_fwd/Proofs'
new=Path(__file__).resolve().parent/'Proofs'
def norm(s):
    s=re.sub(r'/\-.*?\-/','',s,flags=re.S); s=re.sub(r'--[^\n]*','',s)
    return re.sub(r'\s+',' ',s).strip()
def pin(p):return {'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
records=[]
def check(label,p,q,a,b,replace=()):
    for x,y in replace:a=a.replace(x,y)
    assert norm(a)==norm(b),label
    records.append({'label':label,'old':pin(p),'new':pin(q),
        'normalized_fragment_sha256':hashlib.sha256(norm(b).encode()).hexdigest(),
        'normalization':list(replace)})
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=root/'.veritile-cache/kernel-evidence/chunked_cumsum_fwd/pure-reuse.json'); args=ap.parse_args()
    p,q=old/'Math.lean',new/'KernelMath.lean'
    a=p.read_text().split('structure Layout',1)[1].split('\nend\n',1)[0]
    b=q.read_text().split('structure Layout',1)[1].split('\nend\n',1)[0]
    check('all independent mathematical formulas',p,q,a,b)
    p,q=old/'WholeHost.lean',new/'KernelGeometry.lean'
    a=p.read_text().split('def ceilDiv',1)[1].split('\ndef tileState',1)[0]
    a=re.sub(r'theorem noCollision.*?(?=theorem pair_bound)','',a,flags=re.S)
    b=q.read_text().split('def ceilDiv',1)[1].split('/-- The ordinary same-call',1)[0]
    check('pure geometry, coverage and uniqueness',p,q,a,b,[('Host','Geometry')])
    p,q=old/'RawHostABI.lean',new/'KernelABI.lean'
    a=p.read_text().split('inductive Scalar',1)[1].split('/-- Materialized',1)[0]
    b=q.read_text().split('inductive Scalar',1)[1].split('#guard_msgs',1)[0]
    check('binary64-to-FP32 ABI and independent bounds',p,q,a,b)
    p,q=old/'WholeBackendTile.lean',new/'KernelExecution.lean'
    a=p.read_text().split('inductive SumTree',1)[1].split('/-- Only local',1)[0]
    b=q.read_text().split('inductive SumTree',1)[1].split('/-- Only reached',1)[0]
    check('scalar ordered scan trees and prefix certificate',p,q,a,b)
    p,q=old/'RawObserved.lean',new/'KernelScalar.lean'
    a=p.read_text().split('inductive Term',1)[1].split('def graphScan',1)[0]
    b=q.read_text().split('inductive Term',1)[1].split('def graphScan',1)[0]
    check('scalar terms, local model, interpretation and explicit contraction',p,q,a,b)
    p,q=new/'Contract.lean',new/'KernelContract.lean'
    a=p.read_text().split('theorem tensorCheck_iff',1)[1].split('\ndef noNegative',1)[0]
    b=q.read_text().split('theorem tensorCheck_iff',1)[1].split('#axiomsClean',1)[0]
    check('unchanged metadata checker equivalences',p,q,a,b)
    for i in [0,1,3,4]:
        p,q=old/f'WholeNativePlan{i}.lean',new/f'KernelPlan{i}.lean'
        a=p.read_text().split('def node0',1)[1].split('#guard_msgs',1)[0]
        b=q.read_text().split('def node0',1)[1].split('#guard_msgs',1)[0]
        check(f'certified ordered prefix row {i}',p,q,a,b)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'status':'passed','script':pin(Path(__file__)),
        'purpose':'Factor pure definitions without importing the historical program/host proof closure. No program AST duplicated.',
        'records':records},indent=2)+'\n')
    print(len(records),'exact pure-fragment correspondences passed')
if __name__=='__main__':main()
