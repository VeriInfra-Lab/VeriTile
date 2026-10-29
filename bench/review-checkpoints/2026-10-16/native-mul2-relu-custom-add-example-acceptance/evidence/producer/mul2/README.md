# triton_mul2 finite source acceptance candidate, revision 2

This additive package proposes repairs for all three blockers in the independently reviewed revision 1. It is frozen for independent root review, contributes **zero acceptance increments**, and makes no native functional-correctness claim. Only root may integrate the module sources and exact numerical authority into the repository. The producer has edited only this temporary package.

The historical package is preserved under `prior-v1/`: all 436 receipt-bound artifacts and eight original module source snapshots. Its receipt SHA256 is `d68f20d8057c543afb8fb90d4c359f38e4c8a7bda8d583b9bad710588e633ac7`. The original independent review remains blocked and unchanged; the ten relevant artifacts under `peer-review/` are checked against review receipt `c20026fa7c10d68c74cdf58998bcb027c4b338fc39a76972b12981d273864cda`. `evidence/revision-lineage.json` binds every historical path.

## Repairs

- **MUL2-ACCESS-OWNERSHIP:** `AccessContext` separates tensor owner metadata from actual read/write permissions. Every permission query uses the current launch device and the accessed byte. A peer allocation or mapped host buffer is admissible when accessible; local ownership alone grants no permission.
- **MUL2-NATIVE-POINTER-TRANSLATION:** `Runtime.pointers` returns actual extracted ABI pointers, translated allocation bases and a shared device-to-backing address map. Loads use actual device addresses through that map. Stores retain both actual device addresses and original backing addresses, and a new compiled theorem proves their resolution for every emitted byte. The result, allocation frame, original tensor metadata and in-place identity remain expressed in the original backing memory. Pointer setup supplies no values or final memory. Null raw pointers bypass unused mapping obligations, matching the exact installed driver.
- **MUL2-I32-MINIMUM-B1-GRID-CASE:** all 26 missing grid-parser rejection cases are added: 13 admitted dtypes times two wrappers at `numel=2^31, BLOCK_SIZE=1`. The compiled theorem `host_grid_threshold` proves the exact threshold `n <= (2^31-1)*B` for every positive block. The table audit partitions all four size classes at that threshold for all 21 valid powers of two. Larger valid blocks at the signed minimum remain all-masked and require no numerical ports.

## Review entry points

1. `after/bench/tritonbench_g/triton_mul2/Proofs/FiniteMemory.lean`, `FiniteProgram.lean`, and `FiniteHost.lean` define context permissions, actual pointer/address translation and the host order.
2. `FinitePointerBoundary.lean` proves actual store/backing resolution and the exact grid threshold. `FiniteRefinement.lean` adds the resolution property to the independent launch result. `FiniteCorrect.lean` exposes the strengthened public contract.
3. `evidence/pointer-replay/` compiles the exact installed pointer extraction and integer extraction functions with an explicitly stubbed CUDA attribute query. `evidence/pointer-replay-context.json` explains the unchanged historical wording in that script's captured output.
4. `evidence/pointer-execution-probe.json` contains 394 executions of the unchanged wrapper ASTs using the actual pointer extractor/parser and synthetic byte/access spaces. It covers local, accessible peer, distinct pinned-host aliases, outplace pointer-setup failure for pageable output, permission denial, complete tile loading, in-place identity, result/frame, empty unmapped backing and grid-error precedence. It is source-plumbing evidence, not a live CUDA accessibility experiment.
5. `evidence/grid-case-audit.json` records 84 complete size/block partitions, 228 native parser boundary checks, 2,704 schema path checks and 352 first-failure-stage checks.
6. `evidence/case-mapping-v1-v2.json` maps every old case and each of the 64 old numerical ports to the strengthened statement/configuration. The table now has 4,674 cases. No scalar operation or numerical domain changed.
7. `authority_adapter.py` is only a proposed extension. It requires independently supplied frozen receipt and substantive root-review hashes, revision 2, current source hashes, frozen source hashes, historical inputs and every artifact. `evidence/authority-check.json` exercises this in an explicitly synthetic fixture, rejects 64 omitted ports and 15 adapter/queue mutations, and preserves whole-queue equality. It is not approval.

## Validation and limits

All nine modules compile, with 36 named standard-axiom and seven independent-reference checks. All 14 Lean attempts (including three failures), source snapshots and logs remain under `evidence/attempts/`; script failures and their repairs remain under `evidence/script-attempts/`. The before/after snapshots verify 174 unchanged project source/object pairs and 36 direct external objects. The finalizer independently enumerates direct external imports to detect omitted object dependencies.

The original native evidence is inherited unchanged: 78 profiles (30 compiled, 42 rejected, six frontend-only), plus the original 172 IR comparisons. No GPU launch or live accessibility test occurred. Four floating multiply families remain deferred through exactly 64 case-specific ports; integers, Boolean encoding, masks, pointer arithmetic, translation, permissions, aliasing, metadata, allocation, error order and grid rejection are structural and cannot be discharged as numerical deferrals. The ordinary Torch/runtime/compiler/MMU boundary remains explicit.

Read the frozen `evidence/receipt.json` and run `verify_package.py` to check all bound sources and artifacts. Do not rerun generating scripts inside a frozen package; copy it first. Receipt and review pins must be supplied by independent root review before integration.
