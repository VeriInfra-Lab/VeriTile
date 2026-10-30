# chunk_retention kernel proof packet

The selected executable is `Whole001/chunk_retention_repaired.py`. `KernelOnly/Kernels.lean`, `Contract.lean`, `Correct.lean`, and `FPEquiv.lean` provide the current four-kernel proof surface in `VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelOnly`. The task-root entries import this surface and retain the historical `Checked` and `Whole001` APIs.

`source_correct` uses `ProgramIO.Implements`. The main FP entry is now `source_candidate_equivalence`, an actual original/candidate program pair; the retained optional `source_fp_equiv` uses `AddressedProgramIO.Implements`. `KernelOnly/IO.lean` exposes the actual source invocation, original program and grid coordinates, eight finite input channels, seven possible output channels, exact strides and stage masks, and no private scratch. `KernelOnly/Spec.lean` gives independent Real retention/VJP formulas and ordered native scalar FP32 formulas on these finite inputs. The scope remains four standalone kernels at arbitrary legal coordinates. Saved histories are inputs to the stages that consume them; the interface does not assert a combined host pipeline.

Actual source access safety, successful execution, every active typed output, and the exact exterior memory frame are conclusions of the shared judgments. `Proofs/IOSafety.lean` derives all four kernels' actual load/store bounds from the shared IO windows, following the source entry, loops and final stores. `Proofs/IOResults.lean` connects the existing complete executions to finite input arrays and independent output formulas. `Proofs/PublicCorrectness.lean` composes those results for the two short public specifications. No handwritten kernel AST or new numerical assumption is needed.

The covered stores are forward history H, optional HT, output O, reverse history dH, DQ, DK, and partial DV. Saved H and dH are ordinary inputs to the standalone O and QKV kernels. The Real targets use the independent retention recurrence and scalar gradient sums. The native targets retain ordered scalar FMA chains, casts, multiplication order, and scale placement. QKV derives operand provenance from ordinary input words and proves the local materialization boundary before the transposed dK result is added.

`correct` and `fp_equiv` close these results over actual `VeriTile.Torch.FP32.Tensor` objects. `KernelTensor.initial` constructs every region input from one caller backing read at the observed Tensor `dataPtr + 4*offset`. Standard `TensorLaws`, actual shapes, resolved contiguous metadata, and legal program coordinates derive logical coordinates and active bounds, including arbitrary storage offsets and partial-DV's leading key-block dimension. `input_read`, `input_matrix_read`, and `native_input_read` identify these reads with genuine Tensor logical values.

All outputs use one shared `KernelTensor.project` post-backing read. Ordinary output separation proves that overlapping descriptions of a physical store select the same region cell. The Tensor endpoint concludes all logical output reads, the exact exterior frame, preservation of every used input's full backing storage (including offset prefixes/tails), and the region frame simultaneously. Output/input backing separation is an ordinary kernel buffer guard; read-only inputs may alias each other. Binding, launch, and separation contracts contain no numerical result or successful-run premise.

The current native endpoint admits all FP32 input words and uses the existing `Numeric` IEEE observation profile: fixed-width local integer/add/sub/mul/FMA/exp2/log2/host-scale laws, with NaNs observed canonically by that profile. Literal laws cover only 0, 1, -5, and 64. There is no matrix, block, reduction, or complete-output oracle in those laws. Native provenance is derived for the finite addresses actually read under source masks; unrelated memory has no tagging premise. `KernelStaging` records the audited local dot-consumption/materialization profile; compiler and hardware conformance remain external interpretation boundaries. This packet does not claim raw NaN-payload equality or native-to-Real equality. The historical raw-byte API remains separately available.

The Python kernel bodies, precision, casts, `allow_tf32=False`, contiguous decorators, wrapper checks, allocation choices, and return behavior are unchanged by this proof completion. Existing evidence in `.veritile-cache/task-deliveries/chunk_retention/` records 52 actual-binder SM80/SM90 compilations, the quotation audit, and wrapper/source controls. `check_source.py` pins the original source and selected repair kernel ASTs. New Lean receipts and logs cover each selected-source result, numerical bridge, Tensor closure, and standard public entry.

The proof scope is kernel execution, numerical targets, actual Tensor reads, and frames. Historical host/autograd/allocator/byteheap developments are retained for compatibility and are not the acceptance boundary of this packet. No allocator, compiler, resource, or host-lifecycle project was added. The common launcher's grid-product overflow issue remains deferred in `bench/TODO.md`; the per-axis guards remain unchanged.

`KernelOnly/Correct.lean` and `KernelOnly/FPEquiv.lean` expose one main specification each, without raw state, execution or per-cell postconditions in their public statements. Their additional actual Tensor applications remain `KernelOnly.TensorBridge.correct` and `KernelOnly.TensorBridge.fp_equiv` in optional `KernelOnly/TensorBridge.lean`, with the same complete statements, shared post-read and frames. They reuse the original `KernelPacket` execution foundation also used by the new public IO proofs. Supporting proof modules do not import public reader entries. The format pass runs targeted Lean checks; it adds no GPU run or independent numerical replay.

The actual pair compares the complete four-stage dispatch against the existing
`forwardOScaleFirst` optimization: only the forward-output stage commutes the
final FP32 multiplication operands; the other three complete stages use their
unchanged original kernels. `Whole001/Kernels.lean` contains its full ordinary
`triton` quotation next to the original. This gives the scalar scale operand the
first position without moving scale through the attention sum or changing any
dot/FMA chain; no measured speedup is claimed. `ScaleFirstSource`,
`ScaleFirstSafety` and `ScaleFirstObservation` prove complete loop execution,
actual access safety, every observed output cell and the raw exterior frame.
`ScaleFirstPair` lifts these to the unchanged addressed input interface for every
legal original PID and both state flags. Its narrow syntax view contains actual
invocations and uses the original runner, input binding and masks. The strict
all-word `ScalarOpsLaws.mul_comm` certificate is the only numerical rewrite;
`Numeric.ScalarLaws` and `LiteralLaws` retain their original model boundary.
There are no experimental rules or deferred numerical experiments. The other
old dot-left candidates are not used or claimed: changing dot/seed placement
would need a separate faithful connection to the selected FMA lowering.
