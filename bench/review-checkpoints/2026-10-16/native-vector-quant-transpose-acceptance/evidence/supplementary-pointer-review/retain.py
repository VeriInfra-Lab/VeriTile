from pathlib import Path
import json,shutil,sys
sys.dont_write_bytecode=True
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
T=Path('/tmp/veritile-standardization-20261015')
P=T/'vector-pointer-totalization-review1-wip'
B=R/'bench/review-checkpoints/2026-10-16/native-vector-quant-transpose-acceptance'
sys.path.insert(0,str(R/'bench'))
from adapter_common import hash_file,load,write_json
expected='aa62e23414c2cafe4bc69906f304256e3bdeadc36b9bebe1cb3ebaf97b864d60'
assert hash_file(P/'receipt.json')==expected
r=load(P/'receipt.json')
for inputs in r['inputs'].values():
 for p,h in inputs.items():assert hash_file(Path(p))==h,p
for p,h in r['artifacts'].items():assert hash_file(P/p)==h,p
D=B/'evidence/supplementary-pointer-review'
assert not D.exists();D.mkdir()
for p,h in {**r['artifacts'],'receipt.json':expected}.items():
 dest=D/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/p,dest)
 assert hash_file(dest)==h
shutil.copyfile(Path(__file__),D/'retain.py')
write_json(D/'root-retention.json',{'reviewer':'root','receipt_sha256':expected,'conclusion':'Read the focused argument and actual CPU metadata witnesses. An unused total pointer field does not assert Result.ok or grant permissions. Its interpretation on absent/failed extraction is arbitrary unconsumed completion; successful extraction still binds the actual pointer. This supplies no new numerical or native correctness claim.','source_or_authority_modified':False,'independent_cpu_witnesses':5,'root_reexecuted_witnesses':False})
print(json.dumps({'retained_receipt_sha256':expected,'artifacts':len(r['artifacts']),'destination':str(D)},indent=2))
