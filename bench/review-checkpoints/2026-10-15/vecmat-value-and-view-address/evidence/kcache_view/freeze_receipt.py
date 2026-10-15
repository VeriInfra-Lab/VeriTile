import datetime, hashlib, json, pathlib, shutil, subprocess
root=pathlib.Path('/tmp/veritile-standardization-20261015/kcache-view-wip')
repo=pathlib.Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
prior=pathlib.Path('/tmp/veritile-standardization-20261015/kcache-host-wip/evidence/host-grid-receipt.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(prior)=='7f3e09619c57f347efa9825f871056b4279b7f599001b019282f647a958c7c78'
old=json.loads(prior.read_text())
records=json.loads((root/'evidence/verification-results.json').read_text())
assert len(records)==3 and all(r['exit_code']==0 for r in records)
for r in records:
    assert r['source_sha256_before']==r['source_sha256_after']==sha(root/r['module'])
    assert r['log_sha256']==sha(root/r['log'])
    assert 'DISALLOWED' not in (root/r['log']).read_text()
for path,digest in (old['new_sources']|old['existing_dependencies']).items():
    assert sha(repo/path)==digest,(path,'frozen dependency changed')
assert sha(repo/old['original_source']['path'])==old['original_source']['sha256']
# Preserve failed/diagnostic drafts outside the integration candidate tree.
scratch=root.parent/'kcache-view-scratch-20261015'
scratch.mkdir(exist_ok=True)
for relative in ['general-draft.lean','view-shape-compile.log','view-compile.log','view-step.log','view-cases.log',
    'bench/tritonbench_g/kcache_copy_triton/Proofs/HostViewDraft.lean',
    'bench/tritonbench_g/kcache_copy_triton/Proofs/HostViewDraft128.lean',
    'bench/tritonbench_g/kcache_copy_triton/Proofs/HostViewStep.lean']:
    source=root/relative
    if source.exists():
        dest=scratch/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(source),str(dest))
links={str(p.relative_to(root)):{'target':str(p.resolve()),'sha256':sha(p)}
    for p in root.rglob('*.olean') if p.is_symlink()}
deps={path:sha(repo/path) for path in old['new_sources']|old['existing_dependencies']}
for path in links:
    source=path.removesuffix('.olean')+'.lean'
    if (repo/source).exists():deps[source]=sha(repo/source)
receipt={
 'schema':1,
 'task':'kcache_copy_triton',
 'status':'compiled_logical_view_address_preservation_pending_independent_review',
 'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'whole_program_acceptance':False,
 'newly_closed_subgap':'Parameterized logical rank4-to-rank3 successful-view address preservation',
 'source_model':'The existing HostPrepare.inferViewStrides / reshape definitions are unchanged.',
 'original_source':old['original_source'],
 'frozen_prior_receipt':{'path':str(prior),'sha256':sha(prior)},
 'new_sources':{r['module']:r['source_sha256_after'] for r in records},
 'dependency_sources':deps,
 'lean_verification':records,
 'verification_results_sha256':sha(root/'evidence/verification-results.json'),
 'verification_script_sha256':sha(root/'evidence/verify_view.py'),
 'case_generator_sha256':sha(root/'generate_view_cases.py'),
 'freeze_script_sha256':sha(pathlib.Path(__file__)),
 'axiom_audit':'All three modules execute #auditModuleAxioms; all exported results execute #axiomsClean and use only the standard base. Actual compiler exit codes were recorded by subprocess.run.',
 'proof_scope':[
   'Successful inferViewStrides returns exactly three strides, including zero-size shapes.',
   'Every positive dimension is split into singleton or arbitrary >1. All 16 symbolic partitions are proved; no upper bounds or finite parameter probes are used.',
   'Coefficients of all non-singleton coordinates are derived from the actual chunk/consume algorithm; singleton coordinates are zero.',
   'For every Fin a, Fin b, Fin h, Fin d coordinate, the original four-stride offset equals the flattened three-stride offset. The flattened leading index is proved < a*b.',
   'All Nat strides, including zero/broadcast strides and all admitted noncontiguous-view patterns, are covered without injectivity, contiguity or arbitrary address-mapping premises.',
   'The theorem derives actual returned tensor shape/strides/region from reshape success with copied=false, then gives read equality for an arbitrary load function and common storage offset.',
   'Zero dimensions are admitted universally; their element-address statements are vacuous because a corresponding Fin domain is empty. Successful empty-prefix reshape metadata is still covered.',
 ],
 'remaining_nondeferrable_gaps':[
   'Independent review and integration/catalog registration; this receipt adds no whole-program acceptance.',
   'Native Torch reshape/allocation correspondence remains separate from correctness of this explicit Lean algorithm.',
   'Physical tensor/region alias mapping and native storage-offset/byte-address representation remain open. Translation by an arbitrary common logical offset does not provide that bridge.',
   'Full original caller/dtype/error domain, negative n, signed/native-width metadata and resource legality remain open; no new native fp64/raw dtype claim is made.',
   'Original ignored nonunit cache-tail stride, ignored lengths stride and unchecked legacy cache shape remain source behavior, not corrected semantics.',
   'Concurrent scheduling, race freedom and schedule-independent CUDA refinement remain open.',
 ],
 'numerical_deferrals':[],
 'repository_mutation':False,
 'dependency_olean_scaffolding':links,
 'exclude_from_integration':'All .olean files/symlinks and all external scratch drafts are build/diagnostic scaffolding. Candidate Lean source inventory is exactly new_sources.',
 'scratch_archive':str(scratch),
}
output=root/'evidence/view-address-receipt.json'
output.write_text(json.dumps(receipt,indent=2)+'\n')
for relative in receipt['new_sources']:
    (root/relative).chmod(0o444)
output.chmod(0o444)
print(json.dumps({'receipt':str(output),'sha256':sha(output),'sources':receipt['new_sources']},indent=2))
