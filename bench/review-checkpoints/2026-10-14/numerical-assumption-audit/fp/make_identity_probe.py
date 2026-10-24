from pathlib import Path
import json,re
out=Path('/tmp/veritile-assumptions-20261014/fp')
es=json.loads((out/'initial.json').read_text()); files=sorted({e['file'] for e in es})
s=Path('VeriTile/Meta/StatementAudit.lean').read_text()
part1=s[s.index('private def fpProofType'):s.index('private def ppSpecField')]
part2=s[s.index('private def specString?'):s.index('private def printCompactAtoms')]
a=part2.index('private def printFPAtom'); b=part2.index('private structure FPAssumptionState')
part2=part2[:a]+'''private def printFPAtom (entry : Expr) (_details : Bool := false) : MetaM Unit := do
  let rule ← Meta.mkAppM ``VeriTile.Spec.RuleEntry.rule #[entry]
  let contract ← Meta.mkAppM ``VeriTile.Spec.AtomicRule.contract #[rule]
  let id ← specString? ``VeriTile.Spec.Contract.ruleID contract
  let key ← specString? ``VeriTile.Spec.Contract.instanceKey contract
  let item := Lean.Json.mkObj [
    ("rule_id", id.map Lean.Json.str |>.getD .null),
    ("instance_key", key.map Lean.Json.str |>.getD .null)]
  logInfo m!"ATOM_JSON {item.compress}"

'''+part2[b:]
part2=part2.replace('elab "#print_fp_assumptions "','elab "#print_fp_identity "')
part2=part2.replace('FP assumptions used by {shortSpecName name}:','DECL {name}:')
pre='\n'.join('import '+f[:-5].replace('/','.') for f in files)
pre+='\nimport VeriTile.Meta.StatementAudit\nopen Lean Elab Command VeriTile.Meta\nnamespace FPInventoryProbe\n'
cmd=[]
for e in es:
 src=Path(e['file']).read_text(); pos=re.search(r'^specification '+re.escape(e['declaration'])+r'\b',src,re.M).start(); stack=[]
 for line in src[:pos].splitlines():
  m=re.match(r'^(namespace|section)(?:\s+(\S+))?\s*$',line)
  if m: stack.append((m[1],m[2])); continue
  m=re.match(r'^end(?:\s+(\S+))?\s*$',line)
  if m:
   assert stack, (e['file'],line)
   stack.pop()
 e['qualified_declaration']='.'.join(n for k,n in stack if k=='namespace')+'.'+e['declaration']
 cmd.append('#print_fp_identity '+e['qualified_declaration'])
part1=re.sub(r'\b(true|false)\b', r'Bool.\1',part1)
part2=re.sub(r'\b(true|false)\b', r'Bool.\1',part2)
(out/'IdentityProbe.lean').write_text(pre+part1+part2+'\nend FPInventoryProbe\n'+'\n'.join(cmd)+'\n')
(out/'initial.json').write_text(json.dumps(es,indent=2)+'\n')
print(len(files),'imports;',len(es),'queries')
