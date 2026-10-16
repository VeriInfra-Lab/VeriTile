#!/usr/bin/env python3
"""Validate an explicit next-value registry and render a separate 184-task ledger.

Never mutate the original 41 registry or accept a native experiment. Public
statement discovery verifies an explicit review; it never chooses a model role.
Current value contracts and historical numerical-law debt are separate axes.
"""
from __future__ import annotations
import argparse,hashlib,json,re,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'bench'))
from spec_sheet import split_decls,split_statement
from value_source_entries import registered_value_sources
BASE='bench/tritonbench_g/'
MODELS={'backend_value_contract','bit_interpreter_contract','recurrence_identity','memory_frame_contract','protocol_contract'}
RELATIONS={'full_declared_stage','strict_specialization','not_an_execution_contract'}
VARIANTS={'original_source_translation','observed_compiled_graph','compile_repaired_candidate_graph',
          'compile_repaired_candidate','source_and_compiled_graph_specialization','standalone_sc_model'}
DTYPE_RELATIONS={'selected_explicit_profiles_only','complete_enumerated_wrapper_dtype_domain','new_source_declared_profiles'}

def digest(data):return hashlib.sha256(data).hexdigest()

def stage_summary(task,eligible=None,source_variant=None):
    """A specialization/witness/identity cannot fill a whole stage."""
    wanted=set(task['required_value_stages']);covered=set();bits=set()
    for d in task['declarations']:
        if d['role']!='parametric_value_contract' or d['scope_relation']!='full_declared_stage':continue
        if eligible is not None and d['declaration'] not in eligible:continue
        # A separate candidate/compiled graph cannot prove original-source values.
        if source_variant is None and d['source_variant']!=task['primary_source_variant']:continue
        if source_variant is not None and d['source_variant']!=source_variant:continue
        covered.update(d['covers_stages'])
        if d['model_kind']=='bit_interpreter_contract':bits.update(d['covers_stages'])
    return {'required_stages':sorted(wanted),'covered_value_stages':sorted(covered),
        'closed_software_stages':sorted(bits),'uncovered_stages':sorted(wanted-covered),
        'all_declared_stages_have_value_contract':wanted<=covered,
        'all_declared_stages_have_closed_software_contract':wanted<=bits}

def receipt_source_bindings(obj):
    """Read source hashes explicitly present in a pinned receipt, not its names."""
    found={}
    def visit(x):
        if isinstance(x,dict):
            for key,value in x.items():
                if isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) and (key.endswith('.lean') or key.endswith('.py')):
                    found.setdefault(key,set()).add(value)
            path=x.get('source') or x.get('file') or x.get('path')
            if isinstance(path,str):
                for field in ['source_sha256','after_sha256','sha256']:
                    value=x.get(field)
                    if isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value):found.setdefault(path,set()).add(value)
            for value in x.values():visit(value)
        elif isinstance(x,list):
            for value in x:visit(value)
    visit(obj);return found

def collect(root=ROOT):
    inputs={};cache={}
    def read(name,expected=None):
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Missing or escaping input: '+name)
        data=path.read_bytes();h=digest(data)
        if expected is not None and h!=expected:raise ValueError('Stale input: '+name)
        inputs[name]=h;return data
    def load(name,expected=None):return json.loads(read(name,expected))
    registry=load(BASE+'fp_value_extensions.json')
    if registry.get('schema')!=1 or registry.get('native_gpu_execution') is not False:raise ValueError('Invalid extension schema/native status')
    frozen=registry['preserved_original_registry'];original=load(frozen['path'],frozen['sha256'])
    snapshots={n:load(r['path'],r['sha256']) for n,r in registry['baseline'].items()}
    if original!=snapshots['fp_value_repairs.json'] or len(original['tasks'])!=41:
        raise ValueError('Original 41 registry changed')
    baseline=snapshots['numerical_obligations.json'];gaps=snapshots['fp_proof_gaps.json']
    historical_inputs=dict(snapshots['numerical_obligations_review.json']['input_sha256'])
    for oldport in snapshots['coverage_review.json']['ports']:
        historical_inputs[oldport['python_file']]=oldport['python_sha256']
    if len(baseline['tasks'])!=184 or len(baseline['declarations'])!=747:raise ValueError('Unexpected frozen baseline')
    oldtasks={t['task']:t for t in baseline['tasks']}
    review=load(registry['accepted_scope_review']['path'],registry['accepted_scope_review']['sha256'])
    accepted={t['task']:t for t in review['tasks']}
    if set(accepted)!={t['task'] for t in gaps['cast_store_only_tasks']}:raise ValueError('Accepted 23-task scope drift')
    index=load(registry['delivery_index']['path'],registry['delivery_index']['sha256'])
    indexed={p['package']:p for p in index['packages']}
    parent=registry['packages_root']
    packages={};package_sources={}
    for key,row in indexed.items():
        receipt=load(parent+'/'+row['receipt'],row['receipt_sha256'])
        rr=load(parent+'/'+row['root_review'],row['root_review_sha256'])
        if rr.get('status')!=row['review_status']:raise ValueError('Root review status mismatch: '+key)
        if rr.get('receipt_sha256',row['receipt_sha256'])!=row['receipt_sha256']:raise ValueError('Root review receipt mismatch: '+key)
        packages[key]=row;package_sources[key]=receipt_source_bindings(receipt)
    plan=load(registry['native_plan']['path'],registry['native_plan']['sha256'])
    if plan['native_execution'] is not False:raise ValueError('Unexpected new native execution claim')
    entries=registered_value_sources(root)
    entry_registry=load(BASE+'fp_value_source_entries.json')
    for e in entry_registry['entries']:
        for name,h in e['input_sha256'].items():read(name,h)
    def declarations(file):
        if file not in cache:
            text=read(file).decode(); cache[file]=(text,{d['name']:d for d in split_decls(text)})
        return cache[file]
    # Old declarations are immutable textual claims; changed proof bodies and
    # source graphs do not convert their historical model labels into new ones.
    for old in baseline['declarations']:
        _,ds=declarations(old['file']); d=ds.get(old['declaration'])
        if d is None or digest(split_statement(d['text'])[0].encode())!=old['statement_sha256']:
            raise ValueError('Historical public statement changed: '+old['file']+'::'+old['declaration'])
    rows=[];tasks=[];seen=set();unmet=[]
    for task in registry['tasks']:
        name=task['task']
        if name in seen or name not in oldtasks:raise ValueError('Invalid extension task: '+name)
        seen.add(name)
        if task.get('primary_source_variant') not in {'original_source_translation','compile_repaired_candidate'}:
            raise ValueError('Invalid primary source graph: '+name)
        if task.get('dtype_scope_relation') not in DTYPE_RELATIONS:
            raise ValueError('Invalid dtype coverage relation: '+name)
        if task['native_binding_status']!='not_established':raise ValueError('Native claim without separate review: '+name)
        required=set(task['required_value_stages'])
        if not required or len(required)!=len(task['required_value_stages']):raise ValueError('Invalid required stage set: '+name)
        if name in accepted and task['accepted_scope']!=accepted[name]['actual_public_scope']:
            raise ValueError('Old accepted scope rewritten: '+name)
        bound_packages=[];pending=[]
        for binding in task['packages']:
            key=binding['package'];expected=packages.get(key)
            if expected is None: pending.append('review_chain:'+key);continue
            if (binding['receipt_sha256']!=expected['receipt_sha256'] or binding['root_review_sha256']!=expected['root_review_sha256']):raise ValueError('Stale task/package binding: '+key)
            if 'pending_domain' in (expected['review_status'] or ''):pending.append('domain_compatibility:'+key)
            bound_packages.append(key)
        support_bindings={}
        for path,h in task['source_sha256'].items():
            read(path,h)
            evidence=[p for p in bound_packages+['combined-frontend-consumers','maximum-all-consumers','int32-consumer-audit'] if h in package_sources.get(p,{}).get(path,set())]
            if historical_inputs.get(path)==h:evidence.append('frozen_baseline')
            if not evidence:pending.append('unbound_task_support:'+path)
            support_bindings[path]=evidence
        files=task.get('public_files',[task['public_file']])
        if not files or len(set(files))!=len(files) or task['public_file'] not in files:
            raise ValueError('Invalid explicit public file set: '+name)
        final_sources={}
        for actual_file in files:
            if not actual_file.startswith(BASE+name+'/') or not actual_file.endswith('.lean'):raise ValueError('Public source outside task: '+name)
            text,actual=declarations(actual_file)
            specifications={n for n,d in actual.items() if d['kind']=='specification'}
            if specifications!={d['declaration'] for d in task['declarations'] if d['file']==actual_file}:raise ValueError('Public declaration set differs from review: '+name)
            # Every public file must have its own receipt-bound source bytes.
            final_hash=digest(text.encode());sources=bound_packages+['combined-frontend-consumers','maximum-all-consumers']
            source_evidence=[p for p in sources if final_hash in package_sources.get(p,{}).get(actual_file,set())]
            if not source_evidence:pending.append('unbound_final_public_source')
            final_sources[actual_file]=(actual,final_hash,source_evidence)
        for d in task['declarations']:
            if d['file'] not in final_sources or d['model_kind'] not in MODELS or d['scope_relation'] not in RELATIONS or d['source_variant'] not in VARIANTS:
                raise ValueError('Invalid declaration model/scope/source review: '+name)
            actual,final_hash,source_evidence=final_sources[d['file']]
            if not set(d['covers_stages'])<=required:raise ValueError('Undeclared stage: '+d['declaration'])
            if d['model_kind']=='bit_interpreter_contract' and d.get('unclosed_numeric_callbacks'):
                raise ValueError('Unclosed callback promoted to software closure: '+d['declaration'])
            if d['extra_numeric_law_ids']:raise ValueError('New explicit law requires numerical-law review: '+d['declaration'])
            if d['model_kind'] in {'recurrence_identity','memory_frame_contract','protocol_contract'}:
                if d['role']!=d['model_kind'] or d['scope_relation']!='not_an_execution_contract' or d['covers_stages']:
                    raise ValueError('Identity/frame/protocol promoted to value contract')
            elif d['role']!='parametric_value_contract':raise ValueError('Fixed witness cannot enter parametric registry')
            statement=split_statement(actual[d['declaration']]['text'])[0]
            if digest(statement.encode())!=d['statement_sha256']:raise ValueError('New public statement changed: '+d['declaration'])
            if d['native_binding_status']!='not_established':raise ValueError('Declaration claims native binding')
            rows.append({**d,'task':name,'statement':statement,'final_source_sha256':final_hash,
                         'source_receipt_packages':source_evidence,'dtype_profile':task['dtype_profile'],
                         'scope':task['accepted_scope'],'extra_numeric_status':'no_extra_numeric_law_not_native_accepted'})
        summary=stage_summary(task)
        if summary['uncovered_stages']:pending.append('uncovered_accepted_stages')
        source_entry=entries.get(name)
        if source_entry:
            expected_graph='compile_repaired_candidate' if source_entry['source_variant']=='compile_repaired_candidate' else 'original_source_translation'
            if task['primary_source_variant']!=expected_graph:raise ValueError('Primary graph differs from admitted source: '+name)
        if task['baseline_group']=='missing_source' and source_entry is None:pending.append('explicit_source_admission_missing')
        if pending:current='value_contract_review_pending'
        elif source_entry and source_entry['source_variant']=='compile_repaired_candidate':
            current='candidate_only_software_contract' if summary['all_declared_stages_have_closed_software_contract'] else 'candidate_only_backend_contract'
        elif summary['all_declared_stages_have_closed_software_contract']:current='full_declared_profile_software_value_contract'
        else:current='full_declared_profile_backend_value_contract'
        tasks.append({k:v for k,v in task.items() if k not in {'source_sha256','evidence_detail','declarations'}} | {
            **summary,'support_source_bindings':support_bindings,'graph_stage_coverage':{g:stage_summary(task,source_variant=g) for g in sorted({d['source_variant'] for d in task['declarations']})},'current_value_status':current,'review_pending':pending,'source_entry':source_entry,
            'frontend_kind':source_entry.get('frontend_kind','shared_triton_dsl') if source_entry else 'shared_triton_dsl',
            'memory_semantics':source_entry.get('memory_semantics','single_program_total_typed_memory') if source_entry else 'single_program_total_typed_memory',
            'all_python_dtypes_or_host_stages_covered':False,'unified_acceptance':'pending',
            'historical_fp_adequacy_status':oldtasks[name]['fp_adequacy_status'],
            'historical_numeric_status':oldtasks[name]['numerical_status'],
            'historical_extra_numeric_ids':oldtasks[name]['extra_numeric_ids'],
            'fixed_witnesses_counted_as_contracts':0,
            'declaration_model_counts':dict(Counter(d['model_kind'] for d in task['declarations'])),
            'additional_specializations':[d['declaration'] for d in task['declarations'] if d['scope_relation']=='strict_specialization']})
        unmet.extend({'task':name,'condition':x} for x in pending)
    if {t['task'] for t in registry['tasks'] if t['baseline_group']=='cast_store_only'}!=set(accepted):raise ValueError('23-task extension set is incomplete')
    wanted_public={(d['file'],d['declaration']) for d in baseline['declarations']} | {(d['file'],d['declaration']) for d in rows}
    actual_public=set()
    for public_path in sorted((root/BASE).glob('*/*.lean')):
        file=public_path.relative_to(root).as_posix()
        _,parsed=declarations(file)
        actual_public.update((file,n) for n,d in parsed.items() if d['kind']=='specification')
    if actual_public!=wanted_public:
        raise ValueError('Unreviewed/missing current public specifications: '+repr(sorted(actual_public^wanted_public)))
    extension={t['task']:t for t in tasks}
    known_revisions={r['task']:r for r in registry.get('known_pending_semantic_revisions',[])}
    for revision in known_revisions.values():
        if revision['task'] not in oldtasks:raise ValueError('Unknown semantic revision task')
        if revision['status']=='current_reference_semantics_revalidation_pending':continue
        if revision['status']!='backend_equivalence_reference_revalidated':raise ValueError('Invalid semantic revision status')
        binding=revision.get('package',{});key=binding.get('package');expected=packages.get(key)
        if (expected is None or binding.get('receipt_sha256')!=expected['receipt_sha256'] or
            binding.get('root_review_sha256')!=expected['root_review_sha256'] or
            expected['review_status']!='signed32_reference_semantics_revalidated_full_original_scope_pending_integration'):
            raise ValueError('Unreviewed semantic revision: '+revision['task'])
        if not revision.get('source_sha256'):raise ValueError('Semantic revision sources omitted')
        for path,h in revision['source_sha256'].items():
            read(path,h)
            if h not in package_sources[key].get(path,set()):raise ValueError('Unbound semantic revision source: '+path)
    for revision in registry.get('original41_migration_evidence',[]):
        if revision['task'] not in {t['task'] for t in original['tasks']}:raise ValueError('Migration outside original41')
        if not set(revision['packages'])<=set(packages):raise ValueError('Unbound original41 migration')
    current=[]
    for old in baseline['tasks']:
        new=extension.get(old['task'])
        current.append({'task':old['task'],'baseline_fp_adequacy_status':old['fp_adequacy_status'],
            'historical_numeric_status':old['numerical_status'],'historical_extra_numeric_ids':old['extra_numeric_ids'],
            'current_value_status':known_revisions[old['task']]['status'] if old['task'] in known_revisions else new['current_value_status'] if new else old['fp_adequacy_status'],
            'known_reference_revision':known_revisions.get(old['task']),
            'value_extension':new,'native_functional_correctness':'not_established_by_this_review',
            'integration_accepted':False})
    for name in ['bench/fp_value_extensions.py','bench/value_source_entries.py','scripts/spec_sheet.py','bench/port_layout.py']:
        read(name)
    return {'schema':1,'status':'candidate_184_task_recount_pending_unified_acceptance',
      'summary':{'tasks':len(current),'baseline_public_records_preserved':len(baseline['declarations']),
        'original_41_registry_unchanged':True,'new_specification_records':len(rows),
        'new_parametric_value_contracts':sum(d['role']=='parametric_value_contract' for d in rows),
        'recurrence_identities_not_value_contracts':sum(d['role']=='recurrence_identity' for d in rows),
        'memory_frames_not_value_contracts':sum(d['role']=='memory_frame_contract' for d in rows),
        'protocol_contracts_not_value_contracts':sum(d['role']=='protocol_contract' for d in rows),
        'standalone_model_value_contracts':sum(d['role']=='parametric_value_contract' and d['source_variant']=='standalone_sc_model' for d in rows),
        'source_frontend_classes':dict(Counter(t['frontend_kind'] for t in tasks)),
        'fixed_witnesses_promoted':0,'extended_tasks':len(tasks),
        'extension_dtype_relations':dict(Counter(t['dtype_scope_relation'] for t in tasks)),
        'software_profile_dtype_relations':dict(Counter(t['dtype_scope_relation'] for t in tasks if t['current_value_status']=='full_declared_profile_software_value_contract')),
        'original23_dtype_relations':dict(Counter(t['dtype_scope_relation'] for t in tasks if t['baseline_group']=='cast_store_only')),
        'new_declaration_models':dict(Counter(d['model_kind'] for d in rows)),
        'current_task_statuses':dict(Counter(t['current_value_status'] for t in current)),
        'historical_numeric_statuses':dict(Counter(t['historical_numeric_status'] for t in current)),
        'native_experiment_jobs':len(plan['jobs']),'native_execution':False,
        'original_pending_exact_bindings':sum(t['status']=='pending_exact_binding' for t in baseline['experiment_tasks']),
        'review_pending_items':len(unmet)},
      'original41_migration_evidence':registry.get('original41_migration_evidence',[]),
      'historical_obligations':baseline['obligations'],'historical_experiment_tasks':baseline['experiment_tasks'],
      'historical_refuted_model_tasks':gaps['numeric_model_revision_tasks'],
      'unresolved_original_sources':[t['task'] for t in current if t['current_value_status']=='missing_source'],
      'candidate_only_sources':[t['task'] for t in current if t['current_value_status'].startswith('candidate_only')],
      'new_declarations':rows,'tasks':current,'unmet_conditions':unmet,
      'interpretation':['A full declared profile means every listed previously accepted value stage in the explicitly named physical storage/primitive profile. It does not mean all Python caller dtypes, full host execution, compiler equivalence or GPU acceptance.',
        'New contracts do not relabel the historical 747 public records or remove old numerical-law obligations. Conv2d new value semantics and its historical refuted round=id premise are separate.',
        'Source translations and observed compiled/candidate graphs are independent contracts. No universal dot+add=dotAcc/FMA equality is inferred.',
        'Finite native experiments are pending profile checks, not universal semantic proofs. Normal inputs and accepted configuration/stage conditions are not new numeric-law experiment debt.'],
      'integration_blockers':['Full combined source build and exact supplemental roots have not been accepted by this metadata review.', 'Canonical coverage/numerical inventory and source audit must consume the explicit original/candidate source registry and retain task-local SC boundaries; legacy FPCorrect discovery alone remains insufficient.']+[
          r['task']+': reference-semantic revision awaits review' for r in known_revisions.values() if r['status']=='current_reference_semantics_revalidation_pending'],
      'input_sha256':dict(sorted(inputs.items()))}

def render(report):
    s=report['summary'];out=['# 浮点值合同扩展：184项候选统计','',
      f"冻结旧记录保持：{s['baseline_public_records_preserved']} 条公开记录、原41项登记。新增 {s['new_specification_records']} 条 specification：{s['new_parametric_value_contracts']} 条参数化值合同（含独立模型 {s['standalone_model_value_contracts']} 条）、{s['recurrence_identities_not_value_contracts']} 条递推恒等式、{s['memory_frames_not_value_contracts']} 条单独 frame 合同、{s['protocol_contracts_not_value_contracts']} 条协议合同。固定见证不进入合同数量。",'',
      '本表等待统一构建和正式登记验收。所有硬件对应仍待完成；完整范围指明确 dtype/原语配置下的原接受阶段，不代表所有 Python 调用 dtype 或完整 host 程序。','',
      f"原5项精确绑定待办保留（实际计数 {s['original_pending_exact_bindings']}）；本轮独立原核/候选硬件任务 {s['native_experiment_jobs']} 项。旧3项全域 round=id 仍在历史义务轴；Conv2d 新值合同不使用该前提，也不改写旧声明。",'',
      '| 任务 | 新值模型（完整声明配置） | 范围/存储边界 | 旧dtype域关系 | 源状态 |','|---|---|---|---|---|']
    for row in report['tasks']:
        t=row['value_extension']
        if not t:continue
        out.append('| '+row['task']+' | '+row['current_value_status']+' | '+(t['accepted_scope']+' '+t['dtype_profile']).replace('|','\\|')+' | '+t['dtype_scope_relation']+' | '+t['source_status']+' |')
    out+=['','## 原接受范围与 dtype 范围','',
      f"原23项中，全部已接受控制/阶段在新明确配置内保留；完整原wrapper枚举dtype域为 {s['original23_dtype_relations'].get('complete_enumerated_wrapper_dtype_domain',0)} 项（TMA与triton_matmul），仅新增明确dtype profile为 {s['original23_dtype_relations'].get('selected_explicit_profiles_only',0)} 项。不是23项所有dtype都闭合。",'',
      f"完整软件profile按dtype范围分：{s['software_profile_dtype_relations']}。源前端另分：{s['source_frontend_classes']}。task-local SC 的并发模型不代表共享 DSL 或 GPU 内存语义。修复候选单列。",'',
      '旧公共header逐文本保留不等于其引用定义语义未变。参考语义修复按单独的 receipt、复核和所有修改源码哈希核对；未完成的复核仍列为 pending。原41最大值迁移仅补充证据。','',
      '## 逐项复核结果','',f"未满足项：{s['review_pending_items']}。原始源码仍缺：{', '.join(report['unresolved_original_sources'])}。仅候选合同：{', '.join(report['candidate_only_sources'])}。",'',
      'Q/K 有理数缩放、Linear ReLU/identity、row-major布局及active-defined转换是显式附加特例，不能补齐全域阶段。Flash Phi 的 maximum recurrence 是恒等式，不是执行合同。', '',
      'SGMV 原 Python singleton-tuple 仍编译失败，已接受 body 的源级值合同与去逗号候选编译图分开。Linear fast_gelu 原源码的 constexpr 失败亦保留。', '',
      '原41项的 maximum 迁移与全库核心变化只可作为补充源码/重编证据，不回写旧登记模型角色。新字段不接受 FPCorrect 文件存在作为 source admission。','',
      '```sh','python3 bench/fp_value_extensions.py --check','```','']
    for item in report['unmet_conditions']:out.append('- '+item['task']+': '+item['condition'])
    return '\n'.join(out)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');args=p.parse_args()
    r=collect(); outputs={BASE+'fp_value_current_status.json':json.dumps(r,ensure_ascii=False,indent=2)+'\n',BASE+'fp_value_current_status.md':render(r)}
    for name,text in outputs.items():
        path=ROOT/name
        if args.check:
            if not path.exists() or path.read_text()!=text:raise SystemExit('Stale extension report: '+name)
        else:path.write_text(text)
    print(json.dumps(r['summary'],ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
