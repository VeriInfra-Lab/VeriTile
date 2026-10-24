# logsumexp_fwd: explicit reduction graph and reachable internal-stage gaps

This private supplement changes no repository source, public specification or task classification. It keeps the accepted complete per-program **fp32** block-LSE scope: arbitrary D, positive B, both HAS_SCALE branches, arbitrary Real scale, both program ids, arbitrary initial memory/register/undef, partial tails, all-masked extra programs and input/output alias. The Python wrapper's final `z.logsumexp(-1)` and dtype conversion remain outside that scope. The three old FP declarations and original source are archived under `original-scope/` with hashes and original declaration anchors.

## Actual new Lean results

Three new modules, fifteen named standard-axiom checks:

- `Proofs/WordGraph.lean`: replaces whole max/sum callbacks with concrete binary32 maximum-number policy and two independently specified valid reduction trees. Add/sub/mul have their actual per-operation software rounding. `complete_row` and `complete_row_readback_frame` prove successful complete source execution to an independent initial-memory formula and exact one-cell update/full frame. No output equality is a premise and no old input restriction is added.
- `Proofs/ExpGapReachable.lean`: a legal D=17/B=32 row has one ordinary finite negative input, sixteen +0 inputs, and fifteen -Inf padding lanes. Both flag branches, using exact scale=1 in the fixture only, reach the pinned Bloom exponent-collision inputs. The observed fused scale/sub prefix also reaches them on these two exact fixtures. The generic contract is **not** restricted to scale=1.
- `Proofs/ScaleFusionGap.lean`: ordinary finite x=0x3f800001 and scale=0x3f7ffffe produce a rounded scaled maximum of 1. The source's separate multiply/subtract gives +0; the observed compiled FMA prefix gives 0xa8800000 = -2^-46. This is a reachable internal-prefix counterexample, not a final LSE-output or GPU counterexample.

The strict Real.exp rounding cells and multiply collision are **reused**, not recounted as new numerical proofs, from [the pinned Bloom parent](parent-binding.json). Its receipt is copied and pinned. Directly correctly-rounded exp cannot universally equal an arbitrary SFU applied after the observed rounded log2(e) multiply: the two inputs collapse before the SFU but require different rounded exp words. This rules out a proposed universal **internal exp-stage** binding. It does not establish unequal final stored outputs.

## Remaining boundaries

This is still a **backend-value contract**, not full software/native closure. `exp32`, `log32`, and arbitrary `Real → fp32` scalar conversion remain explicit unresolved parameters. The refinement makes the remaining scalar conversion visible; it does **not** solve or rename it. The old `.real` cell tag is the explicit-fp32 profile's erased carrier, not physical fp64 storage evidence. Runtime NaN/Inf scale has no finite Real representative; these runner cases characterize the actual kernel beyond that scalar representation, without claiming old Real-scale coverage of nonfinite values.

Native pointer/storage representation, unbounded Nat metadata versus actual integer/address representability, scalar ABI/rounding, max NaN/zero policy, shuffle/reduction tree and instruction semantics are still separate bindings. Both selected software trees are explicit and valid; they are not asserted to be the native schedule. All padded lanes remain in the graph, including scale=0 or negative scale and all-masked cases. No finite-row, positive-scale or nonzero-sum premise was introduced.

## Original compiler evidence

[Source audit](source-audit.json) checks exact original JIT signature/body, original heuristic and all six autotune num_warps members. The probe bypasses only decorators to select exact compiler profiles; it does not edit the JIT body. Forty-eight fresh-cache offline profiles compiled: SM80/90 × warps 1/2/4/8/16/32 × both HAS_SCALE branches × fusion off/on, fp32 D17/B32. This is one legal D/B specialization, not an exhaustive dimension audit. The explicit num_stages=3 is a captured default, not an original host keyword.

[Dataflow summary](dataflow-summary.json) and [full recorded paths](native-dataflow.json) check the actual source SSA through masked load → optional scale → maxnum reduce → subtract → exp → sum → log → final add/store. PTX contains 36 separate-sub prefixes and 12 scaled-FMA prefixes. All exponents use the rounded 0x3fb8aa3b multiply then `ex2.approx.f32`. The log is an expanded libdevice logf polynomial with **`fma.rn.ftz.f32`**, special-value selection and control flow; it is not a simple `lg2.approx` primitive and is not implemented/proved by this package.

The inspector rejects altered operands, stale/predicated/multiple-destination writes and control boundaries in its checked scalar spans. Its maximum register graph records shuffle dependencies, not shuffle lane/synchronization semantics. It is **not** a general PTX/CFG verifier or proof of the guarded log algorithm. Complete source/compiler artifacts, metadata, actual ptxas/libdevice/Python and full Triton package hashes are retained before/after.

## Runnable experiment handoff

[experiment-contract.json](experiment-contract.json) lists exact E-exp/scale/prefix/reduce/log/whole relations, profile matrix, known false equalities and missing expected evaluators. The runner executes the **whole original accepted Triton kernel body**, capturing complete input/output allocations and frame at nonzero pointer origins. A separate exponent diagnostic captures internal scale/ex2 raw words. Ten cases cover the two exp fixtures/both flags, the fusion fixture, zero/negative/Inf/NaN scales and an explicitly marked all-masked extra program. The original wrapper's host second stage is not silently included.

There is **no whole-output expected bit evaluator**. Output status is CHARACTERIZATION, never full numerical PASS or “matches any model.” The two universal equalities already refuted by proofs are not listed as assumptions that experiments could establish. Driver/GPU environment binding and value observations have separate statuses.

Current evidence: 26 CPU tests, 48 original offline compiler profiles, and 6 offline runner compiles (2 diagnostic plus 4 original flag/architecture combinations) passed. `native-attempt-final/report.json` is **NOT_RUN**, exit 2, zero launches: CUDA unavailable. CPU tests and offline compilation are not GPU evidence.

From this directory, using fresh output paths:

```sh
python -B verify.py
python -B replay.py --output /tmp/root-logsumexp-fresh
/opt/conda/bin/python -B capture.py compile --output /tmp/new-lse-offline --arch 80 --arch 90 --num-warps 4 --fusion 0
/opt/conda/bin/python -B capture.py run --output /tmp/new-lse-capture --device cuda:0 --num-warps 4 --fusion 0
/opt/conda/bin/python -B capture.py run --output /tmp/new-lse-fused --device cuda:0 --num-warps 4 --fusion 1
```

`replay.py` verifies the package and ordered direct-object dependencies before compiling only the three new modules in a new overlay; it does not overwrite the parent Bloom or integration objects. `final-build.json`, `theorems.json` and `reused-import-closure.json` record source/object/log/import pins. No new `specification` registration is proposed.
