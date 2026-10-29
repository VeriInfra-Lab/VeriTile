from pathlib import Path
import hashlib,json,shutil
T=Path('/tmp/veritile-standardization-20261015')
P=T/'next-source-integration-scripts-wip'
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
B=R/'bench/review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance'
assert (B/'evidence/preparation-state.json').is_file()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P/'candidate-receipt.json')=='f83daffb290b0443e82961dde8e679cf723b6604c3c89a6fca6c23e76e128e4a'
v=json.loads((P/'candidate-receipt.json').read_text())
review=T/'next-source-integration-root-script-review.json'
assert sha(review)=='11049f10a8cae46011b1714391e4d183cd872e3f89d72a2f195cc864d4996dde'
state=json.loads((B/'evidence/preparation-state.json').read_text())
assert state['script_sha256']==v['script_sha256']
files=dict(v['artifacts']);files['candidate-receipt.json']=sha(P/'candidate-receipt.json')
for rel,h in files.items():
 assert sha(P/rel)==h,rel
 dest=B/'evidence/integration-scripts'/rel
 if dest.exists():assert sha(dest)==h,rel
 else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/rel,dest)
 assert sha(dest)==h
dest=B/'evidence/integration-scripts/root-script-review.json'
assert not dest.exists();shutil.copyfile(review,dest)
shutil.copyfile(Path(__file__),B/'evidence/retain-script-review.py')
print(json.dumps({'script_package_artifacts':len(files),'root_script_review_sha256':sha(dest)}))
