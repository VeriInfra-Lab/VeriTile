import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.CheckedResident
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ResidentPublic
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.OutcomeFrames
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.PhysicalRoles
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.IntegerBinding

namespace VeriTile.Bench.Repairs.CrossEntropyOps.HostFPEquiv

/-!
Whole-call FP32 returned-byte equivalence.

Revision003 adds `complete_resident`, the source-to-resident entry. It derives
returned loss/zloss/gradient bytes, exact return identity and tuple, captured
saved-state continuity and caller-byte frames from the existing source proofs
plus one ordinary external compiler/frontend/runtime correctness contract.
It has no NativeBindings, ReturnPath or whole native write-exclusion premise.
The fixed source-level tape retains operation provenance and fusion policy.
`Engine.labels` is explicitly trusted ordinary frontend semantics, including
all embedded and inherited sites; no compiler implementation proof is claimed.
The historical `complete` and outcome helpers below remain available unchanged.
 The exact full source invocation
selects its own reduction/communication trees and per-occurrence fusion policy;
width alone never selects a graph. Loss, logging z-loss and gradient are all
covered, including their actual return descriptors and negative-view storage.

`complete` links actual source slots and saved context to one installed decoder
image, ordered register operands, exact host ledger, request identities and
collective routes, last stores, call completion and returned bytes. It also
proves the source/host expression result and exact eight-entry backward ABI.
`forward_all_outcomes` / `backward_all_outcomes` transport finite actual native
write traces on either API success or failure; their address-only exclusion
certificates are structural runtime evidence, not numerical assumptions.

Precision: FP32 storage/arithmetic, int64 labels and widened address metadata.
Domain: all 2^32 input word patterns per operand; no finite-input restriction.
Unsplit `(z*L)*L` and split host `z*square(L)` remain different operation graphs.
The only numerical premises are fixed-arity `Obligations`, local `Peepholes`
and `ConstantPorts` from the frozen CE1 library. No FMA = separate mul/add law
or whole reduction/kernel/output law is used. Ordinary compiler/decoder,
allocator, Torch and collective runtime contracts remain explicit. No GPU
execution or native numerical conformance is claimed by the producer builds.
-/
export Resident (checked_complete_resident complete_resident complete_resident_values resident_marker_frames
  CompilerCorrect Engine CapturedRegions ReturnedValue CallerFrames PeerCalls PeerMessages)
export NativeGraph (complete NativeBindings NativeReturnedBytes SourceReturned
  transmitted_sources forward_all_outcomes backward_all_outcomes)
export ConfiguredPlan (reference_equiv barrier_stops_contraction)
export InvocationGraphs (selected_graph_equiv alignment_graphs_differ fused_operand_order)
export Physical (forward_arguments backward_arguments forward_logit_pointer forward_scalar_roles
  forward_target_pointer backward_logit_pointer backward_upstream_pointer backward_saved_pointers backward_store_pointer)
export Native (ignored_comparison relative_label_exact guarded_relative_label rank_class_metadata annotated_class_failure)
export Host (python_return)

#axiomsClean complete_resident
#axiomsClean complete
#axiomsClean transmitted_sources
#axiomsClean forward_all_outcomes
#axiomsClean backward_all_outcomes
#axiomsClean forward_arguments
#axiomsClean backward_arguments
#axiomsClean guarded_relative_label
#axiomsClean python_return
#specNonCircular CrossEntropy1.Floating.Obligations avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#specNonCircular CrossEntropy1.Floating.Peepholes avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.CrossEntropyOps.HostFPEquiv
