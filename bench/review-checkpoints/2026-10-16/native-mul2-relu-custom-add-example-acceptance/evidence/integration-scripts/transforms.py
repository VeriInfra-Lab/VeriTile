"""Pure, AST-checked transformations; no file writes occur on import."""
from common import *
ARCHIVE_HELPER = '''
def _archive_evidence_input(relative, package_path):
    """Relocate only frozen evidence; the caller still checks safety and hashes."""
    rel = Path(relative)
    package = Path(package_path)
    if rel.is_relative_to(package):
        text = rel.as_posix()
        if text.endswith('.lean'):
            return Path(text + '.txt')
        if text.endswith(('.olean', '.olean.private', '.olean.server', '.ilean', '.ir', '.pyc')):
            return Path(text + '.bin')
    return relative
'''

def adapter_source(source):
 original=ast.parse(source);binds=[n for n in ast.walk(original) if isinstance(n,ast.FunctionDef) and n.name=='bind']
 require(len(binds)==1,'Adapter must contain exactly one bind function')
 bind=binds[0];arg=bind.args.args[0].arg;lines=source.splitlines(keepends=True)
 insertion=bind.body[0].lineno-1;indent=' '*bind.body[0].col_offset
 statement=f'{arg} = _archive_evidence_input({arg}, package_path)'
 lines.insert(insertion,indent+statement+'\n');new=''.join(lines).rstrip()+'\n'+ARCHIVE_HELPER
 tree=ast.parse(new);helper=tree.body.pop();require(isinstance(helper,ast.FunctionDef) and helper.name=='_archive_evidence_input','Helper placement changed')
 changed=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='bind')
 require(ast.dump(changed.body.pop(0))==ast.dump(ast.parse(statement).body[0]),'Unexpected bind transformation')
 require(ast.dump(tree)==ast.dump(original),'Adapter logic changed beyond archive mapping')
 return new

def extension_source(source):
 tree=ast.parse(source);original=deepcopy(tree)
 f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='extend_reviewed_authority')
 require(isinstance(f.body[-1],ast.Return),'Extension dispatcher must end with return')
 f.body[-1]=ast.Assign(targets=[ast.Name(id='result',ctx=ast.Store())],value=f.body[-1].value)
 for i,item in enumerate(SPEC['packages']):
  module=Path(item['adapter']).stem;alias='_extend_next_'+item['tag'].replace('-','_')
  package='bench/review-checkpoints/'+SPEC['intended_checkpoint']+'/evidence/producer/'+item['tag']
  review='bench/review-checkpoints/'+SPEC['intended_checkpoint']+'/evidence/root-validation/'+item['tag']+'/review.json'
  text=f'from {module} import {item["adapter_function"]} as {alias}\nresult = {alias}(root, result, {package!r}, {item["receipt"]!r}, {review!r}, {item["review"]!r})'
  f.body.extend(ast.parse(text).body)
 f.body.append(ast.Return(value=ast.Name(id='result',ctx=ast.Load())))
 ast.fix_missing_locations(tree);candidate=ast.unparse(tree)+'\n'
 check=ast.parse(candidate);cf=next(n for n in check.body if isinstance(n,ast.FunctionDef) and n.name=='extend_reviewed_authority')
 cf.body=cf.body[:-9];require(isinstance(cf.body[-1],ast.Assign),'Missing preserved dispatcher return')
 cf.body[-1]=ast.Return(value=cf.body[-1].value)
 require(ast.dump(check)==ast.dump(original),'Prior dispatch/validators changed')
 return candidate

def exact_test_source(source,packages):
 nports=1689+sum(sum(len(z['declarations']) for q in p['delta']['rules'] for z in q['scope_bindings']) for p in packages)
 cases=[29,20,9,340,2601,2400,1144,4498]+[len(t['cases']) for p in packages for t in p['delta']['requirements']]
 unions=[15,13,3,9,26,14,14,22]+[len(t['callback_union']) for p in packages for t in p['delta']['requirements']]
 replacements={
  'self.assertEqual(153,len(queue[\'rules\']))':f'self.assertEqual({EXPECTED["numerical_rules"]},len(queue[\'rules\']))',
  '[29,20,9,340,2601,2400,1144,4498]':str(cases),
  '[15,13,3,9,26,14,14,22]':str(unions),
  'self.assertEqual(1689,':f'self.assertEqual({nports},',
  "self.assertEqual(1642,result['exact_bindings'])":f"self.assertEqual({EXPECTED['exact_bindings']},result['exact_bindings'])",
  "self.assertEqual(11041,result['declaration_cases'])":f"self.assertEqual({EXPECTED['declaration_cases']},result['declaration_cases'])",
 }
 for before,after in replacements.items():
  require(source.count(before)==1,'Expected test assertion not unique: '+before);source=source.replace(before,after)
 # Existing all-case and tamper loops already traverse every registered table.
 # Explicit omission checks ensure the newly accepted tasks cannot borrow old deferrals.
 extra='''
 def test_next_batch_required_port_omissions_are_rejected(self):
  for table in authority['requirements'][8:]:
   case=next(c for c in table['cases'] if c['required_binding_refs'])
   row,p,c=usage(table['task'],lambda t,x:t['declaration']==table['declaration'] and x==case)
   gate.validate_used_numerics(table['task'],row,p,c,queue,root=REPO)
   for index in range(len(row['deferred_numerical_assumptions'])):
    with self.subTest(task=table['task'],omitted=index):
     changed=deepcopy(row);changed['deferred_numerical_assumptions'].pop(index)
     with self.assertRaises(ValueError):gate.validate_used_numerics(table['task'],changed,p,c,queue,root=REPO)
'''
 marker="if __name__=='__main__':";require(source.count(marker)==1,'Exact test entry marker changed')
 source=source.replace(marker,extra+'\n'+marker);ast.parse(source);return source
