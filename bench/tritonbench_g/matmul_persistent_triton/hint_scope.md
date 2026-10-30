# Persistent matmul coordinate-hint checkpoint

The integrated five-module package proves mathematical facts and counterexamples
for the coordinate vectors carrying the original source hints. Independent
local Lean replay passed all five modules and `HintAudit`, reproducing their
artifact hashes and all 43 axiom outputs (two axiom-free) with no diagnostics.
It does not change any public theorem domain or establish compiler, device-memory,
numerical or complete-port correctness.

## Installed implementation and its interpretation

The installed package is Triton 3.7.1. `Proofs/hint_implementation_evidence.json` records
full-file SHA256 hashes, exact source line excerpts, the installed compiler
binary hash, binary addresses and DWARF source lines.

* `language/core.py:2936–2967` validates constexpr integer hint arguments and
  delegates. Its short `multiple_of` docstring is insufficient to describe
  the combined tensor analysis contract.
* `language/semantic.py:1815–1825` checks the number of dimensions, writes
  `tt.divisibility` or `tt.contiguity` on the input's IR handle and returns the
  same tensor. It inserts no arithmetic change or runtime check.
* `runtime/interpreter.py:1162–1181` also only records these attributes;
  `TensorHandle.set_attr` at lines 56–57 assigns a dictionary entry. Successful
  interpreter execution would not validate the promised values.
* The installed `libtriton.so`'s `AxisInfoAnalysis::visitOperation` reads the
  literal strings `tt.contiguity`, `tt.divisibility`, and `tt.constancy` and
  calls `AxisInfo::initDimVectorFromHint` to replace the corresponding inferred
  vectors. The exact override calls map to `AxisInfo.cpp:1142,1144,1146` in
  the binary's DWARF. The setter maps to lines 1196–1203.
* The binary also contains the AxisInfo alignment/contiguity consumers. Their
  inspected DWARF/disassembly corresponds to `getAlignment` lines 1308–1337
  and `getContiguity` lines 1275–1293: the relevant dimension's divisibility
  and contiguity constrain alignment, with byte scaling for pointers, and
  per-thread layout contiguity adds a further restriction.

The wheel does not install `AxisInfo.h/.cpp`; the `/project/...` build paths
recorded by DWARF are absent. The included **release-tag comparison sources**
were fetched from Triton's `v3.7.1` tag and hashed. Their relevant implementation
lines and operations agree with the inspected installed binary. DWARF v4
contains no source checksum, so this is not a reproducible-build proof or a
cryptographic source-to-binary identity claim.

The release header, lines 38–85, states the intended AxisInfo meaning:
`contiguity[d]=C` partitions an axis into consecutive groups of C integer
values; `divisibility[d]=D` concerns the **first element of each such group**.
For the source's one-dimensional vector of length B, carrying C=B and D=B,
the mathematical obligation is

```
v[0] mod B = 0
v[lane] = v[0] + lane    for 0 <= lane < B
```

It is not the claim that every lane is divisible by B. The Lean theorem
`HintMath.every_lane_multiple_is_wrong` proves the distinction on the full
tile `[128,129,...,255]`. `TilePromises` is an explicit mathematical predicate
for this interpretation, not a formalization or axiom of compiler soundness.
Power-of-two and other frontend requirements for the two recorded profiles
remain in the unchanged `NativeConfig` results.

Both frozen original TTIR profiles attach the hint pairs to the row/column
`arith.select` results after tail clamping. A replayable offline pass probe
using the installed compiler confirms that both pairs remain after
convert-to-TritonGPU and coalescing. It neither supplies runtime values nor
executes on a GPU; it is implementation evidence, not a correctness proof.

## Checked source/control facts

Original revision `603e28a`, source SHA256
`4537e22452bcb9df7844521498b61232772d7ae44e9aa3220c38c2b341f2152c`:
lines 49–52 initialize coordinate caches; 56–64 update the actual persistent
loop and grouped schedule; 66–73 compute, clamp and hint coordinates only at
`ki == 0`; 74–79 use them with K-only load masks. The host wrapper at 95–115
does not assert that M/N are block multiples.

The new files have the namespace prefix
`VeriTile.Bench.TritonBenchG.MatmulPersistentTriton`:

| Module | Checked result |
|---|---|
| `HintMath` | Explicit contiguous/aligned predicates; full-tile sufficient conditions; every nonempty partial tile refutes full contiguity; clamping preserves first-lane alignment. |
| `HintReached` | Reached signed-i32 coordinate/clamp values equal the derived grouped schedule; full-tile promises; K-phase reuse; reached partial-tile refutations; K=0 has no reached hint. |
| `HintProfiles` | Both recorded-profile witnesses satisfy all reached coordinate promises, including a one-lane last K block; explicit M/N-tail witnesses and counterexamples. |
| `HintCached` | Source offset-vector cache starts with arange, updates only at actual ki=0 and is otherwise retained; its reached values equal the independently derived coordinate expressions. |
| `HintCoverage` | Every valid output tile reaches an actual ki=0 hint site; every nonzero M/N remainder yields a legal PID and reached noncontiguous cached vector; exact all-reached criterion. |

The strongest theorem is `HintCoverage.all_reached_promises_iff`. Under the
unchanged explicit `NativeDomain.Admissible p` arithmetic inequalities and
`0 < count p`:

```
(every legal PID and reached iteration satisfies both coordinate promises)
    iff BM divides M and BN divides N.
```

The proof derives schedules, legal PIDs, actual native-projection control and
cached offsets. No successful execution, safety property, coordinate invariant
or prepared state is assumed. In the converse direction a nonzero remainder
constructs an actual `ki == 0` iteration with a violating vector. The iff is
about these separate coordinate predicates and does not narrow the existing
mathematical public contract. If K=0, no hint site executes; no divisibility
condition is asserted to be necessary in that case.

## Two recorded compile profiles and K tails

Both profiles retain BM=128, BN=256, GM=8, SMS=132 and eight warps. fp16 uses
BK=64/stages=3; E4M3 uses BK=128/stages=4. Native runtime dimensions and strides
are dynamic signed i32 in the existing compilation evidence.

For the original arithmetic witness M=2176, N=2304, K=129, every legal PID
and every reached iteration satisfies the coordinate promises. This includes
PID0's second tile and the shortened final grouped schedule. Both profiles'
last K block has lane 0 enabled and lane 1 disabled, proved using the actual
signed native-projection K mask. K tails do not alter the M/N vectors.

A counterexample in **each same compile profile** uses M=129, N=257, K=129,
A strides 129/1 and B/C strides 257/1. All prior `Admissible` inequalities
are proved. The real host grid is 4; legal PID3 reaches tile (1,1), and its
actual first `ki == 0` branch caches

```
row:    [128, 0, 0, ...]    length 128
column: [256, 0, 0, ...]    length 256
```

Lane 1 refutes the full-length contiguity promise. The starts remain aligned.
The counterexample persists throughout that PID's K phases, including the
partial last phase. The symbolic coverage theorem proves the analogous
failure for **any nonzero M or N remainder**, within the stated integer domain
and with a nonempty K loop, not only this concrete example.

## Validation and boundaries

All five new modules pass fresh local Lean compilation. All 43 named theorems
have explicit axiom output: 41 use only the standard axioms `propext`,
`Classical.choice`, `Quot.sound`; two use none. No `sorry`, new axiom or
`native_decide` is introduced. The independent replay receipt at
`/tmp/veritile-persistent-hints-independent-review/lean-replay-unrestricted/REPLAY_RECEIPT.json`
records the successful module/artifact/log hashes. The safe external replay
driver is `/tmp/veritile-persistent-hints-review-replay.py`; its output directory
must be fresh. The original in-package packaging driver must not be rerun over
the frozen logs. Core artifacts are frozen at exactly the inputs used for
the first successful replay, to isolate this package from subsequent shared
integration. This is local Lean replay, not official comparator replay.

The source/control model remains handwritten and source-mapped. This package
does not prove a full-source/TTIR interpreter refinement, the soundness of
AxisInfo or any compiler transformation, physical pointer/allocation alignment,
native numeric results, GPU execution, concurrency or complete-port acceptance.
The violated coordinate promise is a proven source-level issue under the
documented interpretation; no particular wrong compiled result or device fault
is inferred without additional backend evidence.

All changes in this task are new `/tmp` files. No shared Proofs, public entry,
registry, Lake membership, Git commit or previous checkpoint is modified.
