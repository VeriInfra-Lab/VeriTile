"""Install archive-only adapter transforms and extend authority dispatch/tests."""
from common import *
from transforms import adapter_source,extension_source,exact_test_source
p=arguments(__doc__);a=p.parse_args();r,b,old,s=state(a)
require(not (b/'evidence/install-state.json').exists(),'Install phase already recorded')
packages=package_inputs(a);planned={};adapter_reviews=[]
for package in packages:
 item=package['spec'];source=package['producer']/'authority_adapter.py'
 require(not (r/item['adapter']).exists(),'New adapter already exists: '+item['adapter'])
 text=adapter_source(source.read_text());planned[item['adapter']]=text
 adapter_reviews.append({'task':item['task'],'producer_sha256':digest(source),'installed_sha256':hashlib.sha256(text.encode()).hexdigest(),'ast_preservation_check':'passed_except_single_archive_path_assignment_and_helper'})
extension=r/'bench/reviewed_numerical_extensions.py';tests=r/'bench/tests/test_exact_numerical_bindings.py'
require(extension.read_bytes()==git(r,'show',s['base_commit']+':bench/reviewed_numerical_extensions.py'),'Dispatcher changed since baseline')
require(tests.read_bytes()==git(r,'show',s['base_commit']+':bench/tests/test_exact_numerical_bindings.py'),'Tests changed since baseline')
planned['bench/reviewed_numerical_extensions.py']=extension_source(extension.read_text())
planned['bench/tests/test_exact_numerical_bindings.py']=exact_test_source(tests.read_text(),packages)
for rel,text in planned.items():ast.parse(text,filename=rel)
for rel,text in planned.items():(r/rel).write_text(text)
write(b/'evidence/install-state.json',{'schema':1,'kind':'observed_adapter_transform_checks','adapters':adapter_reviews,'installed_sha256':{rel:digest(r/rel) for rel in planned},'prior_validation_predicates_unchanged':True,'full_tests':'not_run_by_install'})
print(json.dumps({'phase':'install','adapters':len(adapter_reviews),'new_families':sum(x['families'] for x in SPEC['packages']),'new_ports':sum(x['ports'] for x in SPEC['packages']),'tests':'pending'}))
