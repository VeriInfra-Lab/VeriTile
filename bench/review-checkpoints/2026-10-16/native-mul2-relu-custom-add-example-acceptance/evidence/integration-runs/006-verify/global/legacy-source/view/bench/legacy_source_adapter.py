#!/usr/bin/env python3
"""Replay the unchanged legacy source-comparison blocks on explicit old entries.

Global membership/public/build gates are handled by the current catalog. New
typed entries and task-local SC entries are audited through their explicit
registry and source bridge; they are never fed to a pretend shared-DSL parser.
"""
from __future__ import annotations
import argparse
import importlib.util
import shutil
import subprocess
import sys
from adapter_common import *
from current_source_catalog import validate_catalog

START = '# A source view must be extractable before its call/statement inventory can be'
END = '# ---- Machine-checkable trust audit (Phase E)'

def source_blocks(script):
    if script.count(START) != 1 or script.count(END) != 1:
        raise ValueError('legacy source audit boundaries changed; review required')
    text = script[script.index(START):script.index(END)]
    if 'source_only' in text or 'check_comparator' in text or 'lake build' in text:
        raise ValueError('unexpected non-source command in source block')
    return text

def legacy_projection(root, task):
    """Exact historical entry/local-import closure plus unchanged layout headers.

    An unrelated newly added typed helper is not the historical source selected
    by port_layout. It remains mandatory in the full catalog and build gate.
    """
    entry=task['historical_source_entry']
    known={r['path']:r['observed_sha256'] for r in task['files']}
    if entry not in known:
        raise ValueError('historical source absent from catalog')
    selected=set();active=set();prefix='bench.tritonbench_g.'+task['task']+'.'
    def visit(name):
        if name in active:raise ValueError('legacy local source import cycle: '+name)
        if name in selected:return
        if name not in known:raise ValueError('legacy local import absent from catalog: '+name)
        text=checked(root,name,known[name]).read_text()
        selected.add(name);active.add(name)
        for dep in imports(text):
            if dep.startswith(prefix):
                visit(Path(*dep.split('.')).with_suffix('.lean').as_posix())
        active.remove(name)
    visit(entry)
    if Path(entry).name=='Kernels.lean':
        for name in ('Correct.lean','FPEquiv.lean'):
            path=(Path(entry).parent/name).as_posix()
            if path not in known:raise ValueError('legacy split layout header missing: '+path)
            checked(root,path,known[path]);selected.add(path)
    return {name:known[name] for name in sorted(selected)}

def prepare(root, tools_root, catalog, output):
    validate_catalog(root,catalog)
    if output.exists():
        raise ValueError('legacy replay output must be a fresh directory')
    output.mkdir(parents=True)
    view=output/'view';view.mkdir()
    input_bindings={}
    def copy(src,name):
        relative(name)
        data=src.read_bytes();h=sha(data)
        if name in input_bindings and input_bindings[name]!=h:
            raise ValueError('conflicting replay input: '+name)
        input_bindings[name]=h
        dest=view/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    # Existing tool implementations stay byte-identical. No shell source call
    # is rewritten and no task-local concurrency source is reclassified.
    for dirname in ['bench','scripts']:
        for p in sorted((tools_root/dirname).glob('*.py')):
            copy(p,p.relative_to(tools_root).as_posix())
    script=within(tools_root,'bench/audit_tritonbench_g.sh').read_text()
    block=source_blocks(script)
    copy(tools_root/'bench/audit_tritonbench_g.sh','bench/audit_tritonbench_g.sh')
    # Close nonstandard source-profile receipt/artifact dependencies exactly.
    sys.path.insert(0,str(tools_root/'scripts'))
    spec=importlib.util.spec_from_file_location('_legacy_acceptance_for_inputs',tools_root/'bench/acceptance_tritonbench_g.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    profile_inputs=mod.source_profile_inputs(tools_root)
    for name,h in profile_inputs.items():
        copy(checked(tools_root,name,h),name)
    tasks=[t for t in catalog['tasks'] if t['historical_source_entry']]
    selected={t['task'] for t in tasks}
    projected={}
    for t in tasks:
        projected[t['task']]=legacy_projection(root,t)
        for name,h in projected[t['task']].items():
            copy(checked(root,name,h),name)
    for name,h in catalog.get('python_input_sha256',{}).items():
        if Path(name).parts[2] in selected:
            copy(checked(root,name,h),name)
    for name,h in catalog['build_input_sha256'].items():
        if not name.startswith(BASE):
            copy(checked(root,name,h),name)
    shell='#!/usr/bin/env bash\nset -uo pipefail\nPORTS_ROOT="$PWD/bench/tritonbench_g"\nfailures=0\n'+block
    shell+='\nif [ "$failures" -gt 0 ]; then exit 1; fi\nprintf "Explicit legacy source comparisons passed\\n"\n'
    (view/'run-source-blocks.sh').write_text(shell)
    input_bindings['run-source-blocks.sh']=sha(shell.encode())
    return {'schema':1,'kind':'historical_selected_source_replay',
        'catalog_sha256':sha(canonical(catalog)),
        'tasks':sorted(selected),'source_entries':{t['task']:t['historical_source_entry'] for t in tasks},
        'projected_task_files':projected,
        'excluded_nonlegacy_task_files':[f['path'] for t in tasks for f in t['files'] if f['path'] not in projected[t['task']]],
        'source_script_sha256':hash_file(tools_root/'bench/audit_tritonbench_g.sh'),
        'unchanged_source_blocks_sha256':sha(block.encode()),
        'python_gate_blocks':block.count("if python3 - "),
        'profile_dependency_count':len(profile_inputs),'input_sha256':input_bindings,
        'explicit_new_entries_not_run_through_legacy_dsl':[t['task'] for t in catalog['tasks'] if not t['historical_source_entry']],
        'boundary':'Replay covers historical selected Python/DSL views only. New original/candidate source admissions use their exact registry/bridge; task-local SC is not ordinary shared DSL. No proof/native/comparator claim.'}

def run(root,tools_root,catalog,output):
    receipt=prepare(root,tools_root,catalog,output)
    log=output/'source.log'
    with log.open('w') as stream:
        process=subprocess.run(['bash','run-source-blocks.sh'],cwd=output/'view',stdout=stream,stderr=subprocess.STDOUT)
    changed=[]
    for name,h in receipt['input_sha256'].items():
        try:checked(output/'view',name,h)
        except ValueError:changed.append(name)
    receipt.update(exit_code=process.returncode,status='passed' if process.returncode==0 and not changed else 'failed',
        changed_inputs=changed,log_sha256=hash_file(log))
    write_json(output/'receipt.json',receipt)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--tools-root',type=Path,required=True)
    p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=run(a.root,a.tools_root,load(a.catalog),a.output)
    print(__import__('json').dumps({k:r[k] for k in ['status','exit_code','tasks','python_gate_blocks']},indent=2))
    return int(r['status']!='passed')

if __name__=='__main__':sys.exit(main())
