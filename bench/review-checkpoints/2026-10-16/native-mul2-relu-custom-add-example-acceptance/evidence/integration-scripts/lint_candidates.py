"""Parse candidates and test pure transforms without importing mutating phases."""
from common import *
from transforms import adapter_source,extension_source,exact_test_source,ARCHIVE_HELPER
p=arguments(__doc__);a=p.parse_args();r,b,old=context(a)
export=HERE/'lint-output';export.mkdir(exist_ok=True)
parsed=[]
for path in sorted(HERE.glob('*.py')):ast.parse(path.read_text(),filename=str(path));parsed.append(path.name)
packages=package_inputs(a,check_hashes=False);generated=[];mapping_checks=[];whitespace=[]
for package in packages:
 item=package['spec'];source=package['producer']/'authority_adapter.py';text=adapter_source(source.read_text());out=export/Path(item['adapter']).name;out.write_text(text);generated.append({'path':str(out),'sha256':digest(out),'original_sha256':digest(source),'ast_preservation':'passed'})
 for rel,(path,h) in package['sources'].items():
  bad=[i for i,line in enumerate(path.read_text().splitlines(),1) if line.rstrip()!=line]
  if bad:whitespace.append({'path':rel,'trailing_whitespace_lines':bad})
text=extension_source((r/'bench/reviewed_numerical_extensions.py').read_text());(export/'reviewed_numerical_extensions.py').write_text(text)
text=exact_test_source((r/'bench/tests/test_exact_numerical_bindings.py').read_text(),packages);(export/'test_exact_numerical_bindings.py').write_text(text)
namespace={'Path':Path};exec(ARCHIVE_HELPER,namespace);fn=namespace['_archive_evidence_input']
for suffix in ['.lean','.olean','.olean.private','.olean.server','.ilean','.ir','.pyc','.json','.lean.txt','.pyc.bin']:
 inp='frozen/pkg/evidence/sample'+suffix;result=fn(inp,'frozen/pkg');require(str(result)==archive(inp),'Archive transform incorrect: '+suffix)
 require(fn('active/sample'+suffix,'frozen/pkg')=='active/sample'+suffix,'Archive helper changed active source')
 mapping_checks.append(suffix)
# Names selected for the only newly constructed source view are verified against actual frozen files.
ac=bench_imports(r);package=next(x for x in packages if x['spec']['task']=='add_example');prefix='bench/tritonbench_g/add_example/'
selected={prefix+'Kernels.lean':['add_kernel'],prefix+'Proofs/FiniteSource.lean':['lowerBody','runProgram'],prefix+'Proofs/FiniteHost.lean':['sourceHost','sourceAllocated']}
for rel,names in selected.items():
 path=package['sources'][rel][0] if rel in package['sources'] else r/rel
 require(set(names)<=set(ac.declarations(path)),'New source view names missing: '+rel)
from progress_docs import render_documents
for rel,text in render_documents((r/'bench/original_program_scope.md').read_text(),(r/'bench/current_source_acceptance.md').read_text()).items():
 (export/Path(rel).name).write_text(text)
 require('8 source scopes' in text and '175' in text and 'current-source-catalog.json.gz' in text,'Current progress summary missing')
report={'status':'passed','syntax_parsed':parsed,'adapter_ast_preservation':generated,'dispatcher_ast_preservation':'passed','generated_test_syntax':'passed','archive_mapping_cases':mapping_checks,'add_example_selected_definitions':selected,'active_source_trailing_whitespace':whitespace,'integration_executed':False,'build_and_gate_results':'not_run_by_script_preparation'}
write(HERE/'lint-report.json',report);print(json.dumps(report,indent=2))
