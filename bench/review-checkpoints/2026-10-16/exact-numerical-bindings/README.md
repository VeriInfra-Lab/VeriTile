# Exact numerical bindings: production adapter

This checkpoint proposes repository-root-aware exact numerical deferrals. It adds 18 scalar rule families and 86 source-bound ports (133 declaration/port bindings) over 58 cases in three public declarations. Every new numerical obligation remains explicitly unvalidated. This work adds no native execution and accepts no additional original programs.

The 11 legacy rules remain byte-for-byte equivalent as data; the current collector extends the queue to 29 rules. All 35 descriptive follow-up entries remain preserved and are not registered wholesale. Original-program acceptance still requires complete stages, admitted dtypes, branches, source/graph correspondence, memory/address/layout behavior, host behavior, and error behavior. A numerical queue cannot discharge those structural obligations.

## Exact supported scope

The fp64 vector-addition entries cover 29 admitted dtype pairs at the frozen CUDA sm90 frontend configuration; the second public declaration covers its 20 floating-output cases. ReLU covers nine integer/bool source dtypes only at N=1025, block_size=1024, sm90. Signed negative and nonnegative branches retain separate scalar predicates; unsigned/bool zero uses the source-selected fromNat route. Other frontend configurations and ReLU N values remain unregistered.

The authority re-reads live repository Lean declarations, callback sources, original Python, frozen receipts, TTIR, and exact configuration evidence. It rejects changed or missing pinned sources, public statements, receipts, cases, or scalar ports. Each scalar port needs its own explicit deferral reference. The production scope checker passes its requested root into this authority.

## Evidence and validation

The retained run passed 238 Python tests: 210 existing tests in five unchanged files and 28 added exact-binding tests. It also passed queue generation/check, the exact collector, and production checker imports/help. The structural gate AST is unchanged except for passing the explicit root into numerical validation; the original strict scope tests remain unchanged.

The actual collector CLI was loaded from one copied checkout and run with --root pointing to a second checkout. Python audit hooks prohibited data reads from the first checkout, the live repository, and the old prototype/evidence roots. The positive run passed. Removing the relocated original Python source then failed with the expected missing-input error; no fallback was permitted. Python source files used to load the collector and their corresponding bytecode probe paths were the only first-checkout exceptions. The failed initial harness attempt that also blocked an import-cache probe is retained under evidence/validation/failed-attempts.

The completed-test source guard, exact commands, logs, output hashes, generated authority/queue, source inventory, and validation report are retained under evidence/. No tests were rerun during recovery or packaging. All 292 repository inputs/helpers/tests selected by the retained harness still match commit 74cb67ae99cd5cf77096dc74868d8adcb243b46a. The intervening original-scope review commit changed none of those bytes.

All 60 exact inputs resolve within the selected repository root after integration. Of these, 59 were already present. The one supplemental source-routes receipt is copied byte-for-byte at evidence/fp64-source-routes/receipt.json, SHA-256 5254bfa22f320ac782fa2beb212ad5092f26f3bc5222b740b5540166346b88f9. The core and grid receipts independently reference that hash. The original frozen prototype and original metadata baseline are unchanged.

The historical validation harness is retained for audit and records its temporary workspace paths. Normal production checks below use only repository files. The prototype receipt and root review are lineage evidence; they do not stand in for review of this production adapter.

## Integration

The integration manifest lists seven production files and the checkpoint tree. The package receipt pins the exact source and evidence bytes. From the proposed package, run the following preflight, replacing the two absolute paths with the package and checkout locations:

```sh
python3 -B bench/review-checkpoints/2026-10-16/exact-numerical-bindings/integrate.py --package /absolute/package --root /absolute/checkout
```

Preflight makes no changes. After root review, use the same command with --apply to copy precisely the manifest files. This script does not invoke git. It checks all destination preimages before copying and accepts later commits only when every bound input/helper/test remains unchanged. Existing test files, original metadata baseline, Lean proofs, and scope statuses are outside the copy manifest.

Then run from the repository root:

```sh
python3 -B bench/deferred_numerical_experiments.py --root . --check
python3 -B bench/exact_numerical_bindings.py --root .
python3 -B bench/check_original_program_scope.py --help
```

The prior 238-test result can be reused for these exact unchanged bytes. Root review and the normal overall scope audit remain integration steps.
