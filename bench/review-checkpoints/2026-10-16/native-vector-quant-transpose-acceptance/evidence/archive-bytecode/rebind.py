from pathlib import Path
import copy,sys,shutil
sys.dont_write_bytecode=True
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015')
B=R/'bench/review-checkpoints/2026-10-16/native-vector-quant-transpose-acceptance';D=B/'evidence/archive-bytecode'
sys.path.insert(0,str(R/'bench'));from adapter_common import load,hash_file,write_json
old=load(D/'numerical-queue-before.json');new=load(R/'bench/tritonbench_g/deferred_numerical_experiments.json')
key='repo:'+str((B/'evidence/producer/quant-transpose/evidence/pre-quant-baseline/__pycache__/reviewed_numerical_extensions.cpython-312.pyc').relative_to(R))
expected=copy.deepcopy(old);h=expected['exact_numerical_inputs'].pop(key);expected['exact_numerical_inputs'][key+'.bin']=h
assert expected==new,'Archive relocation changed more than the one exact input path'
ledgerpath=R/'bench/original_program_scope.json';ledger=load(ledgerpath);before=hash_file(ledgerpath)
assert ledger['deferred_queue']['sha256']==hash_file(D/'numerical-queue-before.json')
ledger['deferred_queue']['sha256']=hash_file(R/'bench/tritonbench_g/deferred_numerical_experiments.json');write_json(ledgerpath,ledger)
shutil.copyfile(Path(__file__),D/'rebind.py')
write_json(D/'rebind.json',{'reviewer':'root','old_input_path':key,'new_input_path':key+'.bin','exact_input_bytes_sha256':h,'queue_only_change':'One exact_numerical_inputs path renamed; all values and every other JSON field unchanged','ledger_only_change':'deferred_queue.sha256','before_ledger_sha256':before,'after_ledger_sha256':hash_file(ledgerpath),'new_queue_sha256':ledger['deferred_queue']['sha256']})
print('Preserved all semantic authority fields; rebound only archive path and ledger queue hash.')
