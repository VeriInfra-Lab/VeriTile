"""Read-only checks; writes only the explicitly named report outside the repo."""
from common import *
p=arguments(__doc__);p.add_argument('--report',type=Path,default=HERE/'preflight.json');a=p.parse_args()
r,b,old=context(a);require(a.report.resolve().is_relative_to(HERE),'Preflight output must remain in this candidate directory')
packages=package_inputs(a);report={'kind':'read_only_candidate_preflight','observed_head':git(r,'rev-parse','HEAD').decode().strip(),'observed_dirty':bool(git(r,'status','--porcelain')),'integration_executed':False,'specs_sha256':digest(HERE/'specs.json'),'packages':[]}
for p in packages:
 rows=p['records'];largest=max(rows,key=lambda x:x[0].stat().st_size)
 report['packages'].append({'task':p['spec']['task'],'receipt_bound_archive_records':len({rel for _,rel,_ in rows}),'module_sources':len(p['sources']),'largest':{'path':str(largest[0]),'bytes':largest[0].stat().st_size},'archived_lean':sum(str(x[0]).endswith('.lean') for x in rows),'archived_pyc':sum(str(x[0]).endswith('.pyc') for x in rows),'all_pins_and_artifact_hashes_passed':True,'table_cases':[len(t['cases']) for t in p['delta']['requirements']],'callback_unions':[len(t['callback_union']) for t in p['delta']['requirements']],'declaration_port_bindings':sum(len(z['declarations']) for q in p['delta']['rules'] for z in q['scope_bindings'])})
report['queue_estimate']=estimated_queue(r,packages)
report['runtime_prerequisites']=['Clean committed predecessor and gzip helper; runtime HEAD captured only then','Explicit existing .json.gz ledger; explicitly named predecessor metadata files','Root review of added add_example catalog source view and script transformations','Actual catalog/build/test/global checks recorded by run_checks.py after install/rebind']
write(a.report,report);print(json.dumps(report,indent=2))
