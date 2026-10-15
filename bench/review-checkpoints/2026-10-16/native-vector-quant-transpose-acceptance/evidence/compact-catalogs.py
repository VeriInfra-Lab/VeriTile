from pathlib import Path
import json,sys,shutil
sys.dont_write_bytecode=True
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015')
B=R/'bench/review-checkpoints/2026-10-16/native-vector-quant-transpose-acceptance';G=T/'native-vector-quant-current-global'
sys.path.insert(0,str(R/'bench'));from adapter_common import load,hash_file,canonical,sha,write_json
assert (G/'receipt.json').exists(), 'Do not change serialized inputs during the global audit'
assert not (B/'evidence/json-formatting.json').exists()
records=[]
for name in ['current-source-catalog.json','catalog-supplement.json']:
 p=B/name;old=p.read_bytes();data=load(p)
 assert old==(json.dumps(data,indent=2,ensure_ascii=False)+'\n').encode()
 canonical_hash=sha(canonical(data));old_hash=sha(old)
 new=(json.dumps(data,separators=(',',':'),ensure_ascii=False)+'\n').encode()
 assert len(new)<100*1024*1024
 p.write_bytes(new)
 assert load(p)==data and sha(canonical(load(p)))==canonical_hash
 records.append({'path':p.relative_to(R).as_posix(),'before_sha256':old_hash,'after_sha256':hash_file(p),'before_bytes':len(old),'after_bytes':len(new),'canonical_sha256':canonical_hash,'parsed_content_equal':True,'previous_bytes_reproducible':'json.dumps(parsed, indent=2, ensure_ascii=False) + newline'})
assert load(G/'receipt.json')['catalog_sha256']==sha(canonical(load(B/'current-source-catalog.json')))
write_json(B/'evidence/json-formatting.json',{'schema':1,'reviewer':'root','reason':'Generated coverage snapshots exceeded100MiB per file. Only JSON whitespace changed; exact parsed objects, canonical review hashes, source proofs and frozen evidence remain unchanged. Formatting performed after global audit completed.','records':records})
shutil.copyfile(Path(__file__),B/'evidence/compact-catalogs.py')
print(json.dumps(records,indent=2))
