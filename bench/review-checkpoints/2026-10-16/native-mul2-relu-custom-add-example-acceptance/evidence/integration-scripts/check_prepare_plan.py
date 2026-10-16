"""Execute only the explicitly bounded read-only prefix of prepare.py."""
from common import *
source=(HERE/'prepare.py').read_text()
marker='# All prechecks above are read-only except the /tmp helper probe. Mutation starts here.'
require(source.count(marker)==1,'Prepare read-only boundary changed')
namespace={'__name__':'__prepare_plan__','__file__':str(HERE/'prepare.py')}
exec(compile(source.split(marker)[0],str(HERE/'prepare.py'),'exec'),namespace)
record=namespace['state_record'];supplement=namespace['supp']
write(HERE/'prepare-plan-report.json',{'status':'passed','observed_committed_base':record['base_commit'],'repository_mutations':0,'new_modules':len(record['new_source_sha256']),'old_ledger':record['baseline_ledger_path'],'new_ledger':record['ledger_path'],'old_ledger_exact_bytes_archive':record['baseline_ledger_archive'],'old_active_ledger_removed_during_rebind':record['deactivated_ledger_paths'],'supplement_declarations':sum(len(t['declarations']) for t in supplement['tasks']),'supplement_role_counts':dict(Counter(d['role'] for t in supplement['tasks'] for d in t['declarations'])),'add_example_source_view':record['add_example_source_view'],'gzip_storage_receipt_sha256':record['gzip_storage_receipt_sha256'],'queue_estimate':record['queue_size_preflight']})
print((HERE/'prepare-plan-report.json').read_text())
