"""Run only prepare's read-only prefix against the actual future committed base."""
from common import *
parser=arguments(__doc__);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args();root,checkpoint,prior=context(args)
target=external_output(root,args.report);require(not target.exists(),'Plan report must be fresh')
source=(HERE/'prepare.py').read_text();marker='# All prechecks above are read-only. Mutation starts here.';require(source.count(marker)==1,'Read-only boundary changed')
namespace={'__name__':'__prepare_plan__','__file__':str(HERE/'prepare.py')};saved=sys.argv
try:
 sys.argv=[str(HERE/'prepare.py'),*runtime_arguments(args)]
 exec(compile(source.split(marker)[0],str(HERE/'prepare.py'),'exec'),namespace)
finally:sys.argv=saved
state_record=namespace['state_record'];supplement=namespace['supp']
data={'schema':1,'kind':'reviewable_three_source_integration_plan','status':'passed_read_only_preparation','base_commit':state_record['base_commit'],'scripts_receipt_sha256':state_record['scripts_receipt_sha256'],'storage_validation_sha256':state_record['storage_validation_sha256'],'preparation_state':state_record,'new_modules':len(state_record['new_source_sha256']),'supplement_declarations':sum(len(t['declarations']) for t in supplement['tasks']),'supplement_roles':dict(Counter(d['role'] for t in supplement['tasks'] for d in t['declarations'])),'source_views':state_record['new_source_views'],'workflow_candidate_sha256':hashlib.sha256(namespace['newtext'].encode()).hexdigest(),'historical_deletions_authorized':False,'repository_modified':False,'legacy_acceptance_increment':0,'primary_completion_increment':0}
write(target,data);print(json.dumps({'report':str(target),'sha256':digest(target),'actual_base_commit':state_record['base_commit'],'repository_modified':False}))
