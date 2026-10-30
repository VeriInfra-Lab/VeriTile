# Source binding

Original Python SHA256:
`de4ee0e9235d0ead69d3fa94f00b341444e869344ca5c3f216a22ce8e44bc8b0`.
The original is unchanged. The candidate retains the historical single-grid
i64 index repair and equivalent difference-form window predicate. The checked
wrapper replaces historical payload resolution/private-output/copyback with
direct caller views and metadata/storage guards.

| Python/selected native operation | Checked graph/proof mapping |
| --- | --- |
| Packed starts/lengths and GQA division | Checked.Geometry, NativeIndex.Correspondence, raw signed MetadataBinding |
| Masked Q, K and V loads | Recurrence.query/key/value; Execution.inputRead_active and input stability |
| `zeros - inf` initializer | Exact compiler-folded 0xff800000 word; offline PTX check |
| `qk=zeros; qk+=dot(q,k)` | pending dot consumed with zero initializer; first WGMMA has clear=false-accumulate selector |
| Host reciprocal-square-root scale | Host.scale exact FP32 dyadics, local literal-to-FP32 interpretation |
| Causal mask then window mask, both -1e9 | Recurrence.scores preserves both source selects and finite values |
| Max and equality guard at -1e9 | Complete physical ordered max tree plus delayed raw IEEE equality/select |
| Natural exp | Rounded multiply by raw log2(e) word 0x3fb8aa3b followed by exp2 instruction |
| Sum and online normalizer | Complete physical sum tree, source alpha/beta order and selected FMA contraction |
| Zero-normalizer guard to 1e-9 | Delayed raw equality/select at each physical consumer |
| P scaling, accumulator scaling, `p.to(v.dtype)` | Ordered divide/multiply/product nodes and same-format cast |
| `acc+=dot(p,v)` | Live scaled-accumulator WGMMA chain, including live zero words |
| Masked output store | Execution.Coord coverage, source_pointer, dynamic stores, readback, precise byte frame |

The Real public root executes the complete existing AST body through
`stepStmts`, proves successful output readback against `Logical.expected`, and
composes projected stores in current physical memory. This preserves arbitrary
read/read aliases through ordinary input-role reification.

The FP public root deliberately has a selected scalar/store graph boundary.
`SourceGraph.Binding` is an external correspondence for the actual selected
specialization. It supplies the selected fragments for the exact parsed
`Program.kernel` prelude, loop body/control and epilogue, with local graph maps
for raw input leaves, initialization, one iteration, loop count, final cast and
store operands. `SourceGraph.value` and `WholeFP.selectedDispatch` execute these
fields; their proofs consume the local maps and `Recurrence.loop_recurrence`
to reach the independent initial-word recurrence. Neither the phantom source
indices nor `source_split` infer compiler correspondence. The eight saved
profiles do not certify it for every admitted call. `NativeInstructions.evaluate` expands all
dots into fixed k8 instructions, retaining the clear/live mode from source
syntax rather than testing an accumulator's numerical value. Historical helper
theorems that only established AST execution success are not used as an
independent FP output target. The source-to-graph relation and exact compiler
specialization are structural interpretation boundaries, with no desired
output or execution-equals-spec assumption.

The source audit in `check_contract.py` checks the two and only two floating
equality sites, dot/scale/exp/normalizer/update/cast statements, and the literal
initialization. `check_lowering.py` verifies WGMMA widths, exp2 family, native
stores, folded negative infinity, clear QK and live PV selectors for eight
current specializations. Full reduction-owner/compiler routing conformance
remains conditional on the selected structural profile; no offline check is
reported as GPU numerical evidence.

The public `WholeFP.checked_call` composes the local little-endian `WordCodec`
with readback to conclude every output word and all four of its bytes, as well
as the exact outside-store byte frame.

Evidence logs: `/tmp/veritile-context-mistral-check.log`,
`/tmp/veritile-context-mistral-lowering-final.log`,
`/tmp/veritile-context-mistral-lowering/report.json`, and
`/tmp/veritile-context-final-build.log`,
`/tmp/veritile-context-bytes.log`, `/tmp/veritile-context-source-graph.log`, and
`/tmp/veritile-context-source-public.log`.
