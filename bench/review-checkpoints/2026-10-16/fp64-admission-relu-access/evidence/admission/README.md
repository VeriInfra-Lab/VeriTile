# Finite sm90 dtype admission model

Two Lean proof modules and four public protocol-model specifications classify all
289 ordered pairs of 17 canonical scalar signatures. They prove the selected-gate
partition: 189 pass, 64 reject at signature serialization, 36 reject at addition
promotion. Standalone promotion has 217 successes and 72 failures. Of the 64
signature rejections, 28 have successful standalone promotion and 36 overlap its
failures; those 36 do not reach addition on the audited source route.

This adds four model contracts, zero execution contracts, zero value dtype
profiles and zero accepted original programs. All 100 rejected pairs, including
the 28 promotion-success fnuz pairs, remain separate from value execution.

Actual final compilation exits are 0/0: 17 named axiom checks, two module audits,
zero warnings and zero noncircular-command checks. Earlier real failures and
source snapshots are retained under evidence/attempt-01 and attempt-02. A separate
successful executable export compares all 289 model rows to prior frozen probes.

The source audit freezes 28 exact Python units and performs 24 checks, including
builder.options assignment, argument serialization before body traversal, the
sm90 supported set and tensor promotion branches. Prior probe bytes are preserved.
No original Python module, frontend compiler, native backend or GPU was executed
for this batch. The four public statements have protocol_contract metadata and
covers_stages=[] because they do not establish original-program execution.

Python/AST-to-model semantic refinement is still a structural gap. Host allocation,
CUDA assertions, JIT/cache/launch paths, module/harness effects, other dtype families,
architecture or option overrides, exception wrapping, lowering/debug/runtime
behavior, physical memory and GPU execution remain open. These gaps are not
numerical assumptions. This package adds no numerical assumptions.
