import json
from pathlib import Path
out=Path(__file__).resolve().parent/'notes'
def ev(task,file,line,end=None):return {'file':f'bench/tritonbench_g/{task}/{file}','line':line,'end_line':end or line}
def gap(t,n,kind,finding,why,refs):return {'id':t.upper()+'-S'+str(n),'kind':kind,'finding':finding,'why_blocks_full_original_scope':why,'can_defer_as_numeric_assumption':False,'evidence':[ev(t,*r) for r in refs]}
def write(t,preserved,mapping,dtype,branches,gaps,numerical):
 d={'task':t,'status':'structural_gaps_confirmed','whole_original_program_accepted':False,'completion_increment':0,'existing_proofs_preserved':preserved,'production_stage_mapping':[{'function':f,'coverage':c} for f,c in mapping],'dtype_review':dtype,'branch_review':branches,'structural_gaps':gaps,'numerical_items_eligible_for_explicit_deferral':numerical,'precondition_assessment':'The existing bounds, nonoverlap, positive tile size and valid-address premises are legitimate selected-domain assumptions. No unrestricted invalid-input behavior or hypothetical hardware defect is demanded. Missing wrapper composition and dtype admission are separate structural questions.'}
 p=out/(t+'.json');assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n')