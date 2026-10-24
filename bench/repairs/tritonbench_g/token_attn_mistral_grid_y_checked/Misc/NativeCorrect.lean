import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Misc.NativeIOResults

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked
open VeriTile Triton FP.Structural TokenAttnMistral
noncomputable section

/-- Native observation of the complete original source: FP32 loads/stores,
source accumulation precision, every padded operation, all legal reduction
trees and adjacent fusion choices. Primitive add/mul/FMA semantics retain the
existing model boundary; no floating rewrite law is assumed. Arbitrary symbolic
inputs also admit every original tagged-word heap through PhysicalIO.binding. -/
theorem kernel_native (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat)
    (separate : Grid.Separate p) (unique : Grid.Unique p work) :
    (IO.native R p work counts).ObservesOn (IO.observe I S p work)
      (Ports.Domain p work counts) (Reference.native I R S p work counts) :=
  PublicIO.native I R S p work counts separate unique

#specNonCircular Reference.native avoiding [kernel,TokenAttnMistral.kernel,FP.Structural.exec,Grid.run,outputValues]
#specNonCircular Ports.Domain avoiding [kernel,TokenAttnMistral.kernel,FP.Structural.exec,Grid.run,outputValues]
#stmtSurfaceSubset kernel_native ⊆ [Parameters,Grid.PID,Grid.Separate,Grid.Unique,Native.Instructions,
  Native.Reductions,Native.Schedule,Native.Term,Native.Word,Native.instInhabitedTerm,
  IO.native,IO.observe,Ports.Domain,Reference.native,ProgramIO.ObservesOn]
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked
