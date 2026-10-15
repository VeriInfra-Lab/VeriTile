#!/usr/bin/env python3
"""Read-only source review; all generated artifacts remain in this directory."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
OUT = Path(__file__).resolve().parent
CAT = 'bench/review-checkpoints/2026-10-16/quantization-and-mixed-softmax/current-source-catalog.json'
LEDGER = 'bench/original_program_scope.json'
QREVIEW = 'bench/review-checkpoints/2026-10-16/quantization-and-mixed-softmax/root-review.json'
BASE = 'bench/tritonbench_g/'
OLDRELU = Path('/tmp/veritile-standardization-20261015/relu-original-wip/evidence/dtype-audit')

def digest(data):
    return hashlib.sha256(data).hexdigest()

def write(name, value):
    p = OUT / name
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return {'path': str(p), 'sha256': digest(p.read_bytes())}

inputs = {}
cache = {}
def read(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    data = p.read_bytes()
    key = str(p)
    if key in inputs:
        assert inputs[key] == digest(data), 'source changed during review: ' + key
    inputs[key] = digest(data)
    cache[key] = data
    return data.decode()

evidence = {}
def ev(ident, path, start, end, finding):
    text = read(path)
    lines = text.splitlines(keepends=True)
    assert 1 <= start <= end <= len(lines), (path, start, end, len(lines))
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    excerpt = ''.join(lines[start-1:end])
    evidence[ident] = {'path': str(p), 'file_sha256': inputs[str(p)],
        'line_start': start, 'line_end': end, 'excerpt_sha256': digest(excerpt.encode()),
        'finding': finding, 'excerpt': excerpt}

ev('checker_inventory', 'bench/check_original_program_scope.py', 129, 172,
   'Every function is inventoried; a reviewed non-JIT test helper may be a harness. No dtype token does not establish an unconstrained domain. Top-level imports/calls, Assert, BoolOp, masks and tl.where are not separate automatic branch IDs.')
ev('checker_evidence', 'bench/check_original_program_scope.py', 189, 207,
   'Whole execution evidence must be original-source full-stage execution. There is no general compiler/native refinement predicate; standalone protocol/rejection tables cannot alone cover an execution stage.')
ev('checker_gate', 'bench/check_original_program_scope.py', 333, 417,
   'Exact schema-2 coverage and reviewed public evidence are required. Nonempty structural_gaps fails regardless of why prose was placed there. The checker does not itself derive whether those prose gaps are in scope.')
ev('checker_numerics', 'bench/check_original_program_scope.py', 265, 330,
   'Numerical declarations/cases/callbacks must map to exact queue registrations; no omitted source stage can be hidden in a numerical law.')
ev('exact_numerical_gate', 'bench/exact_scope_gate.py', 1, 35,
   'Pinned numerical tables constrain exact registered cases. Broadening configurations requires a reviewed new binding, not only a changed label.')
ev('global_boundary', 'bench/current_source_audit.py', 337, 370,
   'Global source/metadata/comparator gates explicitly leave native_functional_correctness not established. No universal runtime proof is a separate gate.')
ev('relu_source', BASE+'relu_triton_kernel/relu_triton_kernel.py', 5, 36,
   'All pids perform load and comparison; only pid zero stores. Wrapper fixes output fp32, block1024, numel and ceil grid.')
ev('relu_harness', BASE+'relu_triton_kernel/relu_triton_kernel.py', 45, 70,
   'test_relu is not JIT and is not called by production functions. Its top-level invocation does not make it a required production stage under the checker harness rule.')
ev('relu_typed_source', BASE+'relu_triton_kernel/Proofs/TypedValue.lean', 18, 117,
   'Twelve successful source materializations include integer/bool and fp16/bf16/fp32, with the original store gate. fp64 is absent; fp8 is not a successful supported route in the old audit.')
ev('relu_public_and_bug', BASE+'relu_triton_kernel/TypedGridCorrect.lean', 14, 46,
   'Full grid theorem and explicit caller_tail_untouched already exist. Faithful characterization cannot be advertised as full-tensor mathematical ReLU.')
ev('relu_host', BASE+'relu_triton_kernel/Proofs/HostWrapper.lean', 62, 90,
   'Finite host primitive sequence already models output dtype/device, numel, dead num_blocks calculation, fixed block and ceil-grid launch.')
ev('relu_host_typed', BASE+'relu_triton_kernel/Proofs/HostWrapper.lean', 165, 212,
   'The host has a 12-profile typed launch and constructive relu_logical_correct helper; it is not one of the public specifications in the current catalog.')
ev('relu_source_grid', BASE+'relu_triton_kernel/SourceGridCorrect.lean', 1, 38,
   'Checked byte/source grid covers nine integer/bool encodings. It is stronger physical-model evidence, not a mandatory replacement for already valid floating logical source execution.')
ev('vector_source', BASE+'vector_addition/vector_addition.py', 5, 27,
   'Wrapper allocates x-typed output, asserts CUDA, takes output.numel, fixes block1024 and dispatches full ceil grid. There are no contiguity, equal-shape, equal-device-index or same-dtype checks.')
ev('vector_harness', BASE+'vector_addition/vector_addition.py', 34, 70,
   'Non-JIT test_add is a harness under the actual source-inventory rule; seed/RNG/test-dictionary/module import semantics are not required production stages.')
ev('vector_host', BASE+'vector_addition/Proofs/HostWrapper.lean', 81, 169,
   'Host control/error sequence is modeled, but logicalLaunch intentionally ignores dtype/layout and invokes the old real-valued grid. A typed wrapper composition remains real work.')
ev('vector_admission', BASE+'vector_addition/Proofs/DTypeAdmission.lean', 31, 147,
   'Finite canonical17x17 partition is189 dtype-gate successes,64 signature rejections,36 addition rejections. Of217 semantic promotions,28 are rejected earlier; they do not require successful kernels.')
ev('vector_admission_public', BASE+'vector_addition/DTypeAdmissionCorrect.lean', 1, 41,
   'Public admission declarations explicitly remain finite protocol models, with no original execution claim. Keep them as supporting evidence and compose them with a source-bound host outcome contract.')
ev('vector_fp64', BASE+'vector_addition/FP64GridCorrect.lean', 1, 44,
   'All29 fp64 dtype routes have logical grid evidence; only20 floating-output routes are unconditional in values. Nine integer/bool-output routes require OutputDefined.')
ev('vector_defined_domain', BASE+'vector_addition/Proofs/FP64Memory.lean', 47, 98,
   'Fresh output separation is a host-primitive property to bind. OutputDefined is an extra value-domain condition and cannot silently replace source behavior outside that domain.')
ev('vector_order', BASE+'vector_addition/Proofs/TypedSchedule.lean', 1, 34,
   'Logical ownership and schedule results already exist; source-specific read/write conflict freedom under fresh output is finite work, universal GPU scheduler refinement is not.')
ev('matrix_source', BASE+'matrix_transpose/matrix_transpose.py', 5, 36,
   'Wrapper hardcodes both storages to fp16, fresh randn/zeros, strides and one program. tl.arange requires an admission domain; arbitrary Nat logical dimensions do not claim all native dimensions compile.')
ev('matrix_host', BASE+'matrix_transpose/HostCorrect.lean', 1, 49,
   'Existing host model supplies source shapes, exact contiguous strides, arbitrary sampled fp16 words, zero output and one original kernel.')
ev('matrix_public', BASE+'matrix_transpose/HostCorrect.lean', 144, 196,
   'Public full host transpose/frame and finite index bound already exist; trusted Torch primitives can discharge freshness/metadata without proving the allocator.')
ev('quant_global_source', BASE+'quantize_global/quantize_global.py', 7, 39,
   'Real remaining work: abs/max/unsqueeze/reciprocal wiring, x shape to int8 output, assertions, both autotune block sizes, ceil grid and tuple return.')
ev('quant_transpose_source', BASE+'quant_transpose_kernel/quant_transpose_kernel.py', 30, 79,
   'Real remaining work: grouped pid, last partial group, masks and actual output stride ordering, abs/max/reciprocal producer and host assertions/return.')
ev('quant_review_global', QREVIEW, 212, 259,
   'The new review mixes the valid host/grid/dtype obligations with broad Native physical memory/runtime/launch refinement. Split source wiring from ordinary trusted toolchain semantics.')
ev('quant_review_transpose', QREVIEW, 263, 311,
   'Same broad boundary plus a genuine missing grouped-pid mapping. The new llrint/narrowing execution remains valid selected fp32 progress.')
ev('relu_old_profile_counts', str(OLDRELU/'receipt.json'), 62, 73,
   'Frozen original-source audit records17 canonical inputs,13 generated frontend programs and no backend/GPU executions, at N1025 B1024.')
ev('relu_fp64_old_route', str(OLDRELU/'frontend-profiles.json'), 283, 306,
   'Original fp64 source compares ordered >=0 in f64, selects in f64 and truncates to f32 for the actual store.')
ev('relu_fp8_old_rejections', str(OLDRELU/'frontend-profiles.json'), 307, 350,
   'Two ordinary fp8 types reject at comparison, two fnuz fp8 types reject at architecture signature admission. These are rejection cases, not missing successful arithmetic paths.')

catalog = json.loads(read(CAT))
ledger = json.loads(read(LEDGER))
quant_review = json.loads(read(QREVIEW))
selected = ['relu_triton_kernel', 'vector_addition', 'matrix_transpose', 'quantize_global', 'quant_transpose_kernel']
rows = {r['task']: r for r in ledger['tasks'] if r['task'] in selected}
tasks = {r['task']: r for r in catalog['tasks'] if r['task'] in selected}
snapshot = {'catalog': CAT, 'ledger': LEDGER, 'ledger_rows': rows,
            'catalog_task_rows': tasks}
write('snapshot-selected-rows.json', snapshot)

inventories = {}
for name in selected:
    p = BASE+name+'/'+name+'.py'
    tree = ast.parse(read(p))
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    calls = {n.func.id for f in functions if not f.name.startswith('test_') for n in ast.walk(f)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    inventories[name] = []
    for f in functions:
        decorators = [ast.unparse(d) for d in f.decorator_list]
        inventories[name].append({'function': f.name, 'line': f.lineno,
            'may_be_test_harness': f.name.startswith('test_') and f.name not in calls and not any('jit' in d for d in decorators),
            'assertion_lines': [n.lineno for n in ast.walk(f) if isinstance(n, ast.Assert)],
            'automatic_branch_lines': [n.lineno for n in ast.walk(f) if isinstance(n,(ast.If,ast.IfExp,ast.For,ast.AsyncFor,ast.While,ast.Match))]})

old = json.loads(read(OLDRELU/'receipt.json'))
profiles = json.loads(read(OLDRELU/'frontend-profiles.json'))
assert old['source_sha256'] == inputs[str(ROOT/(BASE+'relu_triton_kernel/relu_triton_kernel.py'))]
assert len(profiles) == 17
admitted = [r['input'] for r in profiles if r['status'] == 'frontend_generated']
rejected = [r['input'] for r in profiles if r['status'] == 'frontend_error']
assert len(admitted) == 13 and len(rejected) == 4
for r in profiles:
    if 'ttir' in r:
        assert digest(read(OLDRELU/r['ttir']).encode()) == r['ttir_sha256']
assert digest(read(OLDRELU/'frontend-profiles.json').encode()) == old['artifacts']['frontend-profiles.json']

def cls(task, index, action, keep, move, evidence_ids):
    return {'task': task, 'original_gap_index': index,
        'original_finding': rows[task]['structural_gaps'][index]['finding'],
        'action': action, 'retained_finite_obligation': keep,
        'trusted_boundary_or_non_obligation': move, 'evidence': evidence_ids}

classification = [
cls('relu_triton_kernel',0,'split',
    'Bind source element offsets/masks to tensor views and a stated admitted integer-address domain. Account for actual source-visible out-of-bounds/overflow rather than assuming safe arbitrary shapes.',
    'Little-endian hardware implementation, allocator provenance implementation, MMU/access-fault/trap/poison ordering and compiler pointer simulation are ordinary target semantics boundaries; no universal proof is demanded by this standard.', ['relu_source','relu_source_grid','checker_gate']),
cls('relu_triton_kernel',1,'split_and_credit_existing_host',
    'Retain a reviewed exact source-to-model correspondence and public composition of wrapper dtype/device/numel/block/grid/return with all successful typed routes. Existing HostWrapper.relu_logical_correct covers12 routes and should receive credit.',
    'Independent verified Python extractor, LLVM/backend correctness, implementation proof of empty_like and import-time harness/module closure are not additional production-stage requirements.', ['relu_source','relu_host','relu_host_typed','checker_inventory']),
cls('relu_triton_kernel',2,'split',
    'Check finite source read/write footprints and independence under a fresh disjoint output; all active high-pid loads and pre-gate computations remain required. Preserve genuine source races if they exist.',
    'Universal GPU instruction interleavings, synchronizer/runtime/fault implementation and hypothetical numerical callback side effects are not separate blockers. Numerical callbacks are pure functions in the accepted semantics.', ['relu_source','relu_source_grid','global_boundary']),
cls('relu_triton_kernel',3,'replace_with_supported_domain_inventory',
    'Add actual fp64 source execution, integrate13 successful profiles, classify4 observed fp8 rejection routes at their real first stages, and bind the finite wrapper domain. Preserve source-specific layout/address assumptions.',
    'Do not demand successful rejected fp8 kernels, duplicate byte-level proofs for the already executed fp16/bf16/fp32 logical paths, or every arbitrary Python object/complex/quantized/packed type merely because no dtype assert is written. Any claimed accepted extra route requires positive source/API admission evidence.', ['relu_typed_source','relu_old_profile_counts','relu_fp64_old_route','relu_fp8_old_rejections']),
cls('vector_addition',0,'split',
    'Review the29 typed source routes against exact original load/add/store/promotion and host dispatch. Different interpreters are allowed when their source/domain correspondence is explicit.',
    'A machine-verified Python-to-TTIR-to-LLVM-to-GPU compiler proof and mandatory migration into shared FP.Structural are not required.', ['vector_source','vector_fp64','checker_evidence']),
cls('vector_addition',1,'narrow_to_source_independence',
    'Use fresh output and disjoint lane windows to justify the same result for source load/compute/store ordering. Retain source-level races for APIs that permit writable aliases.',
    'Do not prove the whole GPU scheduler or all native instruction traces. Here input-input aliases are harmless reads; empty_like supplies a distinct live output under the trusted API contract.', ['vector_source','vector_order','vector_defined_domain']),
cls('vector_addition',2,'split',
    'Bind tensor views/storage offset/extent and finite pid/offset arithmetic to the successful-domain theorem. A missing logical input cell is not silently a valid source input.',
    'Do not require per-task implementation proofs of endianness, CUDA provenance or physical byte memory when typed load/store semantics already define widths.', ['vector_source','vector_defined_domain']),
cls('vector_addition',3,'split',
    'Compose empty_like dtype/shape/freshness, CUDA assertion, output.numel, block1024/grid and return with typed execution. Treat undefined numeric conversions as explicit operation outcomes or faithfully modeled policies; do not claim native rollback from custom errors.',
    'Generic allocation/runtime failure machinery, exact poison/trap/partial-write visibility and module/harness composition are not required absent a source-specific observable claim.', ['vector_host','vector_source','vector_defined_domain','checker_inventory']),
cls('vector_addition',4,'correct_stale_success_requirement',
    'Represent first-stage dtype rejection in the finite wrapper outcome contract:64 signature rejects plus36 addition rejects among289 canonical pairs.',
    'The28 promoted fnuz pairs are in the64 signature rejects. Their lack of a successful logical grid is correct and not a missing success branch.', ['vector_admission','vector_admission_public']),
cls('vector_addition',5,'narrow',
    'Substantive exact source-to-model review with hashes and all operations preserved.',
    'No general Python/AST semantics theorem or verified extractor required.', ['checker_evidence','vector_source']),
cls('vector_addition',6,'split',
    'Keep allocation result metadata/freshness and actual CUDA assertion outcome in the finite wrapper.',
    'Drop original module/harness effects from production completion after explicit test_harness classification.', ['vector_source','vector_host','vector_harness','checker_inventory']),
cls('vector_addition',7,'move_to_trusted_boundary',
    'Pin the chosen frontend/target admission semantics and preserve any source-declared launch option.',
    'Fresh ASTSource versus JIT/cache implementation equivalence is not source-program work; source-independent cache bugs and environment overrides are not newly invented branches.', ['vector_admission','checker_gate']),
cls('vector_addition',8,'finite_domain_review',
    'Use an explicit canonical dtype/API domain and known rejection classes. Check actual source-declared or positively admitted aliases without assuming all Python values are supported.',
    'No unbounded requirement for arbitrary user objects, custom tensor subclasses, monkey patches, hidden compiler overrides or dtype families unsupported by the pinned ABI.', ['checker_inventory','vector_admission']),
cls('vector_addition',9,'move_to_trusted_boundary',
    'Preserve rejection stage and source assertion/error propagation where relevant to the public outcome.',
    'Exact native exception class wrapping, diagnostic spelling and traceback construction are ordinary runtime behavior, not required kernel semantics.', ['vector_host','vector_admission_public']),
cls('vector_addition',10,'move_to_trusted_boundary',
    'Pin declared source configuration and preserve genuine integer overflow/unsafe-memory behavior in the source contract.',
    'General native lowering/runtime/debug/sanitizer correctness is not a source branch.', ['global_boundary','checker_gate']),
cls('vector_addition',11,'move_to_trusted_boundary',
    'Typed source loads/stores, masks, extent and ownership remain in scope.',
    'A universal physical-memory/GPU execution refinement proof is not separately required.', ['vector_source','global_boundary']),
cls('vector_addition',12,'replace_with_dispatch_composition',
    'Compose the160 shared-interpreter routes and29 fp64 routes by exact dtype dispatch, using a common external typed result contract or explicit relation between their memory carriers.',
    'There is no requirement for one monolithic189-profile interpreter. Cardinality alone is not coverage, but heterogeneity alone is not a defect.', ['vector_fp64','vector_host','checker_evidence']),
cls('vector_addition',13,'retain_only_execution_link',
    'Existing admission model is supporting evidence; construct a source-bound host outcome contract whose rejected branch does not launch and whose admitted branch invokes the actual grid theorem.',
    'Do not demand successful execution for28 rejected fnuz profiles or a correctness proof for all Python/JIT exception machinery.', ['vector_admission_public','vector_admission','checker_evidence']),
cls('vector_addition',14,'replace_overbroad_policy',
    'All real source operations/stages/admitted dtype routes/branch outcomes and exact numerical bindings must be covered. Preserve known mismatches, unsafe value assumptions and incomplete host wiring.',
    'The blanket host/module/harness/native storage/address/compiler/error/scheduling statement conflates finite source obligations with the trusted semantic boundary; replace it with the explicit finite policy below.', ['checker_inventory','checker_gate','global_boundary'])]

finite_standard = [
 {'id':'A1','requirement':'Freeze exact original source and production entrypoints; inventory every production kernel/helper/wrapper, literal dtype branch, implicit promotion, output cast, mask, loop and declared launch configuration. Retain test helpers in inventory but mark them test_harness only after the checker production-call/JIT review. Top-level test invocations do not require proving Python import or RNG internals.', 'evidence':['checker_inventory','checker_gate']},
 {'id':'A2','requirement':'State a finite supported API/target dtype domain grounded in source declarations and actual canonical signature admission. For each case, classify success or first rejection. A missing Python guard does not quantify over every possible Python object. A rejected route is covered by faithful rejection, never by inventing a successful kernel; a known accepted route cannot be omitted.', 'evidence':['checker_inventory','vector_admission','relu_old_profile_counts']},
 {'id':'A3','requirement':'For successful cases, execute every source stage and operation and prove the complete source-visible returned memory/metadata/frame, including loops, tails, masks, casts and source store gates. Source-to-model correspondence is substantively reviewed and pinned; a standalone numerical formula or source-independent protocol is insufficient. Different faithful typed interpreters are permitted.', 'evidence':['checker_evidence','checker_gate']},
 {'id':'A4','requirement':'Host composition is a finite sequence of specified Python/Torch/launch primitives: allocation metadata/freshness, exact size/stride calculations, assertions, selected configs, calls and return wiring. Primitive implementation correctness is trusted. Primitive numerical operations such as abs/max/reciprocal may be abstract or deferred numerically, but producer/consumer wiring must be proved. Do not replace launch success or final values with an unconstrained assumed result.', 'evidence':['relu_host_typed','vector_host','quant_global_source']},
 {'id':'A5','requirement':'Make source-relevant memory/layout and integer-address admission explicit. Bind actual view bases/strides/offsets; prove or represent finite-width behavior, lane/program ownership, source races and actual masks. Under fresh disjoint output and disjoint access footprints, a finite independence argument suffices; proving CUDA allocators, MMUs, schedulers, caches and compilers is outside this standard. New safety restrictions must be named and justified, not silently presented as coverage of a larger source domain.', 'evidence':['vector_source','vector_defined_domain','relu_source_grid','matrix_source']},
 {'id':'A6','requirement':'Every unverified numerical law or callback, including concrete software arithmetic-to-native correspondence, must be registered by exact statement/case/dtype/operation/configuration and future validation. Such conformance may be deferred. Missing fp64 operations, incorrect promotion, undefined-value-domain restrictions, omitted source stages or known false conclusions cannot be numerically deferred. An OutputDefined premise needs a documented defined-behavior domain or a faithful complementary outcome, not a promise that all inputs satisfy it.', 'evidence':['checker_numerics','exact_numerical_gate','vector_defined_domain']},
 {'id':'A7','requirement':'Bind current public whole-stage execution evidence through scope_schema2, exact original_scope IDs/cases and complete dtype-domain metadata, then obtain an independent substantive root review. Keep required build/axiom/source/comparator checks separate. No blanket compiler/hardware correctness gate may be added merely because native_functional_correctness remains not_established.', 'evidence':['checker_gate','global_boundary']},
 {'id':'A8','requirement':'Keep as-written behavior and the intended mathematical operator distinct. A source bug is not repaired by broadening metadata or numerical assumptions. Full source characterization may be recorded truthfully, but a refuted intended functional claim cannot count as verified. ReLU pid-zero-only tail behavior remains explicit even after fp64 completion.', 'evidence':['relu_source','relu_public_and_bug']}
]

quant = {
 'scope':'Current quantization-and-mixed-softmax root-review additions only; frozen producer and integrated proofs remain unchanged.',
 'replace_generic_remaining_obligation':'Move Native physical memory/runtime/launch refinement into a named trusted-boundary record. Keep precise source allocation metadata, pointer/stride/address domain and launch/grid composition as finite structural obligations.',
 'quantize_global_keep':['Source abs/max/unsqueeze/reciprocal producer relation to loaded inverse scale and returned absmax.', 'Exact output shape/int8 allocation, CUDA assertion, n_elements and returned tuple.', 'Both declared BLOCK_SIZE1024/2048 paths and complete ceil-grid coverage.', 'An evidence-grounded admitted input/scale dtype inventory and rejection categories; fp32 alone is selected scope.'],
 'quant_transpose_keep':['All preceding producer/allocation/dtype obligations adapted to 2D.', 'Actual grouped one-dimensional pid mapping including last partial group.', 'Input/output stride argument order, masks, host assertions and full output coverage.'],
 'numerical_deferrable':['llrintf exceptional-input and rounding policy for the explicit fp32-to-BitVec64 port.', 'Floating multiplications, reduction/reciprocal and any rounding/fusion/denormal correspondence, per source operation and config.', 'Signed-int8 modulo256 store conversion is structurally implemented; native scalar cast correspondence may be an explicit operation binding, without demanding a whole memory-system proof.'],
 'not_deferrable':['Missing absmax producer-to-kernel relation.', 'Missing grouped scheduling/grid composition.', 'Missing actually admitted dtype routes.', 'Reclassifying the supplied-coordinate transpose tile as a complete host program.'],
 'evidence':['quant_review_global','quant_review_transpose','quant_global_source','quant_transpose_source']
}

candidates = [
 {'rank':1,'task':'matrix_transpose','reason':'Smallest whole-program closure candidate: hardcoded fp16 wrapper and existing public whole-host exact-bit transpose/frame; no arithmetic callback is required to transpose arbitrary sampled words.',
  'actions':['Review a finite successful launch domain for arange/tile/resource/index admissibility and explicitly classify known zero/non-power-of-two rejection, without claiming arbitrary Nat compiles.', 'Use trusted randn/zeros primitive specs to bind fresh regions, shape, strides and returned tensor; no RNG distribution or allocator implementation proof.', 'Publish/bind the existing HostCorrect result to exact wrapper/kernel scope IDs and fp16 complete wrapper domain; classify test harness.', 'Preserve rejected caller/configuration boundaries and run normal current-source/build checks on any actual change.'],
  'evidence':['matrix_source','matrix_host','matrix_public']},
 {'rank':2,'task':'relu_triton_kernel','reason':'Small finite dtype execution deficit:13 old frontend-admitted profiles,12 represented. Parent-selected next implementation is the missing fp64 source route. Full mathematical ReLU remains false for N>1024, irrespective of scope closure.',
  'actions':['Add a genuine64-bit input carrier and source-ordered ordered-ge-zero/select-f64/store-narrow-f32 route. Do not encode fp64 as mathematical real or fp32.', 'Execute original address/mask/load/result statements for all pids and preserve pid==0 store gate, full grid and arbitrary initial output tail.', 'Use the existing12-profile typed and nine-profile stronger word/byte proofs as evidence; do not require duplicating byte machinery for all floating profiles.', 'Bind host dispatch13 successes plus4 known fp8 rejections using the finite original source/API domain.', 'Register new exact numerical callback ports and configurations. Current old ReLU queue binding at N1025/access-grid is not a blanket source-grid/all-N binding.', 'Keep the known source bug/counterexample visible; do not turn source behavior coverage into an unqualified whole-tensor ReLU theorem.'],
  'evidence':['relu_old_profile_counts','relu_typed_source','relu_fp64_old_route','relu_fp8_old_rejections','relu_public_and_bug']},
 {'rank':3,'task':'vector_addition','reason':'Successful dtype-route inventory already totals189 under the pinned canonical17x17 domain, with100 finite dtype rejections. Main remaining work is finite typed host dispatch, source-domain/address binding, value-outcome treatment and numerical registration, rather than missing28 successful fnuz kernels.',
  'actions':['Compose81 integer/bool,25 small-float,18 mixed-fp32,36 mixed-small and29 fp64 routes with exact promotion/store and x-typed output.', 'Replace legacy real host logicalLaunch in the accepted public wrapper contract with dtype dispatch to existing execution proofs; bind empty_like, CUDA assertion, numel, block1024 and return.', 'Compose first64 signature and36 addition rejection outcomes under the trusted frontend primitive semantics. Keep protocol tables supporting evidence until the host execution statement exists.', 'State input byte/view/extent and finite index domain; use output freshness and disjoint access ownership instead of universal GPU scheduling proof.', 'Resolve OutputDefined outside-domain coverage and register every used numerical family, including software arithmetic conformance. Do not count9 fp64 integer-output routes as value-unrestricted.', 'Publish exact original_scope cases and complete dtype-domain evidence, with independent substantive review.'],
  'evidence':['vector_source','vector_host','vector_admission','vector_admission_public','vector_fp64','vector_defined_domain']}
]

ledger_revision = {
 'apply_only_after_root_review':True,
 'preserve':['All frozen checkpoints, producer receipts, existing proof sources, initial_structural_gaps and reviewed_partial_extensions.', 'Exact old gap strings in this supplement, with original task/index and source hashes.', 'Current pending statuses and zero acceptance increment until actual finite obligations are discharged.'],
 'new_current_fields_recommended':['A versioned acceptance_boundary_review referencing this calibration.', 'trusted_semantics with explicit Python/Torch/Triton/load-store/allocation contracts, distinct from deferred_numerical_assumptions.', 'current finite structural_gaps with an operation/stage/source path and a concrete closure action.', 'known_source_counterexamples and supported/rejected domain cases so known wrong behavior is never made green by editing prose.'],
 'mechanics':['No changes to checker needed merely to remove overbroad generic gap prose after substantive review; structural_gaps is a reviewed list, not a generated native-obligation set.', 'Do not bypass full_execution_evidence or relabel DTypeAdmission protocol tables as execution. Compose source-bound outcomes first.', 'Do not change current-source hashes/status to accepted without complete schema2 stage/dtype/branch evidence, exact numerical bindings and review.', 'Where the current metadata schema cannot express a needed source rejection outcome cleanly, add a narrowly scoped reviewed execution/outcome contract rather than weaken existing evidence checks.']
}

report = {'schema':1,'kind':'acceptance_standard_calibration','reviewer':'/root/numerical_adapter_finish',
 'created_at_utc':datetime.now(timezone.utc).isoformat(),
 'authority':'Parent-assigned calibration of user wording: 覆盖原程序全部声明的阶段、dtype 和分支；缺少验证的 numerical assumptions 可明确登记后延期。',
 'status':'proposed_for_root_review_no_acceptance','repository_edits':0,'git_mutations':0,
 'new_lean_runs':0,'new_frontend_runs':0,'gpu_runs':0,'acceptance_increment':0,
 'summary':'Current scope-gap prose overextends the standard to generic compiler/runtime/hardware and harness correctness. Preserve finite source-specific obligations and exact numerical deferrals; do not relax evidence, hide source bugs or count rejected dtypes as successful kernels.',
 'checked_current_catalog':CAT,
 'checker_requires_generic_native_refinement':False,
 'checker_required_boundary_explanation':'The gate checks the reviewed structural_gaps list and exact scope/numerical evidence. Generic native/compiler obligations arise in ledger/review prose, not a separate checker predicate. Source-to-model semantic review remains mandatory.',
 'finite_acceptance_definition':finite_standard,
 'function_inventory_crosscheck':inventories,
 'relu_old_evidence':{'original_source_hash_matches_current':True,'canonical':17,'frontend_generated':13,'frontend_rejected':4,'successful_inputs':admitted,'rejected_inputs':rejected,'represented_successful_profiles':12,'missing_successful_profile':'float64','old_run_boundary':'N1025,B1024,CUDAsm90,Triton3.7.1,frontend only; no new execution and no all-config empirical inference.'},
 'gap_classification':classification,
 'quantization_calibration':quant,
 'nearest_closure_candidates':candidates,
 'suggested_ledger_revision':ledger_revision,
 'independence_limit':'Read-only independent semantic calibration. No proof compilation, no frontend/native rerun, no repository integration, no acceptance. Frozen source observations remain scoped evidence and are not hardware correctness proofs.'}
write('calibration.json',report)
write('evidence.json',evidence)

markdown = '''当前验收缺口中的通用工具链要求应收敛，但真实源码缺口仍然保留。

用户标准是覆盖原程序所有声明阶段、dtype 和分支；明确登记的 numerical assumptions 可以延期。`check_original_program_scope.py` 并没有另设通用 Python、Torch、编译器、GPU 正确性门槛。它检查 exact scope metadata、公开执行契约、数值登记和 root 的实质审查。`current_source_audit.py:369` 仍明确写着 native_functional_correctness=not_established。

19 条 ReLU/vector 当前 structural_gaps 已在 `calibration.json` 逐条保留原文并拆分：真实 source-to-model、typed host/grid、地址域、dtype 路线和 source-visible 错误继续阻塞；通用 JIT/cache、LLVM/backend、allocator/MMU、debug/sanitizer、native exception 字符串和 import-time harness 不再混作额外原程序缺口。完整证据与逐行哈希在 `evidence.json`。

有限验收定义：固定原源码和 production entrypoints；有限、基于实际来源的 dtype/configuration 域；全部成功路线的执行与输出/frame；被拒绝路线的真实首个拒绝阶段；可信 Python/Torch/launch primitives 下的有限 host 参数与结果连接；实际内存/整数地址/访问冲突语义；逐 statement/case/port 登记未验证数值对应；最后绑定 schema2 与独立实质审查。测试 helper 经现有 checker 的 JIT/调用关系检查后标注 test_harness，不证明整个 import/RNG 系统。

ReLU 旧证据已重新核对：原源码哈希相符；17 个常规输入中13个前端生成、4个 fp8 拒绝。现有12个成功 typed source profile 已覆盖整数/bool 与 fp16/bf16/fp32，真正缺 fp64。不能要求4个被拒绝 fp8 成功执行，也不应因缺浮点 byte-grid 扩展而否认已有 logical execution。fp64 下一步需保留64-bit load、ordered comparison、f64 select、f32 store narrowing，以及所有pid的预门控计算和仅pid0写回。

Vector addition 的189个成功 dtype 路线不需要统一成单一解释器，但必须用一个有限、源码对应的 host dispatch 连接。289个常规 pair 的实际 dtype admission 划分为189成功、64签名拒绝、36加法拒绝；所谓28个“promotion成功但缺grid”的fnuz组合就在64个签名拒绝内。现有 protocol tables 不可直接冒充执行契约；旧 host logicalLaunch 仍是 .real 路线，需要typed连接。9个fp64整数/bool输出路线的 OutputDefined 也不可被隐去。

量化新 root-review 的“Native physical memory/runtime/launch refinement”同样应拆分：通用运行时实现移到信任边界；abs/max/unsqueeze/reciprocal 到 scale 的连接、实际分配/断言/返回、完整grid、transpose grouped pid和其他实际admitted dtype仍是有限结构缺口。llrintf与浮点/窄化对应可精确登记延期；已实现的整数寄存器与int8 store不能被无穷硬件要求重新变成未实现。

最接近闭合的候选是 matrix_transpose、relu_triton_kernel、vector_addition；具体动作见 JSON。matrix_transpose 固定fp16且已有全host按位契约，主要是明确合法arange配置和源primitive连接。按父agent当前优先级，下一项具体实现应是ReLU剩余fp64。ReLU在N>1024时留下尾部的已知源码错误必须保留：全源码行为刻画不等于全tensor数学ReLU正确。

本包建议保留历史冻结记录，新增边界校准补充，再将当前 structural_gaps 改成带源位置和有限闭合动作的真实缺口。没有仓库修改、Lean/前端/GPU新执行或验收增量。父agent审查此包后再决定后续实现。
'''
(OUT/'README.md').write_text(markdown)
for key, expected in inputs.items():
    assert digest(Path(key).read_bytes()) == expected, 'input changed before freeze: '+key
write('input-sha256.json',inputs)
manifest = {p.name:digest(p.read_bytes()) for p in OUT.iterdir() if p.is_file() and p.name not in {'receipt.json'}}
receipt = {'schema':1,'kind':'acceptance_calibration_receipt','status':'frozen_for_root_review',
 'artifact_sha256':manifest,'input_files_checked':len(inputs),'evidence_ranges_checked':len(evidence),
 'original_gap_rows_classified':len(classification),'quantization_tasks_calibrated':2,
 'candidate_tasks':len(candidates),'repository_edits':0,'new_lean_runs':0,
 'new_frontend_runs':0,'gpu_runs':0,'acceptance_increment':0}
ref=write('receipt.json',receipt)
print(json.dumps({'receipt':ref,'counts':{k:receipt[k] for k in ['input_files_checked','evidence_ranges_checked','original_gap_rows_classified','quantization_tasks_calibrated','candidate_tasks']}},indent=2))
