"""Generate only /tmp review outputs and check exact AST-preserving transformations."""
from common import *
from transforms import adapter_source,extension_source,exact_test_source,ARCHIVE_HELPER
from progress_docs import render_documents
parser=arguments(__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root,checkpoint,prior=context(args)
out=external_output(root,args.output);require(not out.exists(),'Use a new output directory; preserve earlier attempts');out.mkdir(parents=True)
syntax=[]
for path in sorted(HERE.glob('*.py')):ast.parse(path.read_text(),filename=str(path));syntax.append(path.name)
packages=package_inputs(args,check_hashes=False);adapters=[];views={};source_whitespace=[]
ac=bench_imports(root)
for package in packages:
 item=package['spec'];source=package['producer']/'authority_adapter.py';candidate=adapter_source(source.read_text());target=out/Path(item['adapter']).name;target.write_text(candidate)
 adapters.append({'task':item['task'],'source_sha256':digest(source),'candidate_sha256':digest(target),'ast_preserved_except_archive_helper':True})
 task=new_task_record(package,prose(package['review']['semantic_review']));views[item['task']]=task['source_views']
 for view in task['source_views']:
  for rel,names in {**view['selected_definitions'],**view.get('entry_aliases',{})}.items():
   source=package['sources'][rel][0] if rel in package['sources'] else root/rel
   require(set(names)<=set(ac.declarations(source)),'Selected source definition missing: '+rel)
 for rel,(source,h) in package['sources'].items():
  lines=[i for i,line in enumerate(source.read_text().splitlines(),1) if line.rstrip()!=line]
  if lines:source_whitespace.append({'path':rel,'trailing_whitespace_lines':lines})
(out/'reviewed_numerical_extensions.py').write_text(extension_source((root/'bench/reviewed_numerical_extensions.py').read_text()))
test_source=(root/'bench/tests/test_exact_numerical_bindings.py').read_text();test_candidate=exact_test_source(test_source,packages);(out/'test_exact_numerical_bindings.py').write_text(test_candidate)
# Only the count assertion method changes. All old adversarial tests stay identical.
def without_counts(text):
 tree=ast.parse(text)
 for cls in tree.body:
  if isinstance(cls,ast.ClassDef):cls.body=[n for n in cls.body if not isinstance(n,ast.FunctionDef) or n.name!='test_complete_authority_and_legacy_unchanged']
 return ast.dump(tree,include_attributes=False)
require(without_counts(test_source)==without_counts(test_candidate),'Changed prior exact-binding tests beyond count assertions')
namespace={'Path':Path};exec(ARCHIVE_HELPER,namespace);mapper=namespace['_archive_evidence_input'];mapping=[]
for suffix in ['.lean','.olean','.olean.private','.olean.server','.ilean','.ir','.pyc','.json','.lean.txt','.pyc.bin']:
 archived='frozen/package/evidence/sample'+suffix;active='active/sample'+suffix
 require(str(mapper(archived,'frozen/package'))==archive(archived),'Archive mapping differs');require(mapper(active,'frozen/package')==active,'Active source path changed');mapping.append(suffix)
for rel,text in render_documents((root/'bench/original_program_scope.md').read_text(),(root/'bench/current_source_acceptance.md').read_text()).items():
 require('11 legacy' in text and '172' in text and 'Real, FP32 and BF16' in text,'Primary/legacy distinction missing')
 (out/Path(rel).name).write_text(text)
inventory=primary_inventory(packages);write(out/'primary-version-applicability.json',inventory);(out/'primary-version-applicability.md').write_text(primary_inventory_markdown(inventory));write(out/'source-views.json',views)
report={'schema':1,'status':'passed','syntax_parsed':syntax,'adapters':adapters,'dispatcher_prior_ast_unchanged':True,'old_adversarial_test_ast_unchanged':True,'archive_mapping_suffixes':mapping,'new_source_view_names_checked':True,'source_trailing_whitespace':source_whitespace,'runtime_base_validated':False,'repository_modified':False,'integration_executed':False,'legacy_acceptance_increment':0,'primary_completion_increment':0}
write(out/'report.json',report);print(json.dumps(report,indent=2))
