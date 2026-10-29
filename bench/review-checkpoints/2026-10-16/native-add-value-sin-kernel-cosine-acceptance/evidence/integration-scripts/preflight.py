"""Read-only frozen package checks. No future base, plan or integration result is invented."""
from common import *
parser=arguments(__doc__);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
root,checkpoint,prior=context(args);report=external_output(root,args.report)
require(not report.exists(),'Use a new report path; preserve earlier attempts')
packages=package_inputs(args)
rows=[]
for package in packages:
 item=package['spec'];records=package['records'];largest=max(records,key=lambda x:x[0].stat().st_size)
 rows.append({'task':item['task'],'receipt_bound_archive_records':len({rel for _,rel,_ in records}),'module_sources':len(package['sources']),'largest_artifact_bytes':largest[0].stat().st_size,'largest_artifact_path':str(largest[0]),'total_archive_bytes':sum(path.stat().st_size for path,_,_ in records),'all_artifact_pins_verified':True,'families':len(package['delta']['rules']),'bindings':len(package['delta']['provenance']),'cases':sum(len(t['cases']) for t in package['delta']['requirements']),'table_cases':[len(t['cases']) for t in package['delta']['requirements']],'callback_unions':[len(t['callback_union']) for t in package['delta']['requirements']],'declaration_port_bindings':sum(len(b['declarations']) for r in package['delta']['rules'] for b in r['scope_bindings'])})
data={'schema':1,'status':'passed_for_frozen_input_preflight_only','spec_sha256':digest(HERE/'specs.json'),'packages':rows,'future_base_commit':None,'baseline_counts':BASELINE,'expected_counts':EXPECTED,'active_paths_require_gzip':True,'historical_deletions_authorized':False,'repository_modified':False,'source_integration_executed':False,'primary_completion_increment':0,'legacy_acceptance_increment_before_integration':0}
write(report,data);print(json.dumps(data,indent=2))
