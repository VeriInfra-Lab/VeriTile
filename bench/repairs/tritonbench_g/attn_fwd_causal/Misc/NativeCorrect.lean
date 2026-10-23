/- Retained single-source native/software correctness. Original declaration
namespaces and full statements are preserved as ordinary auxiliary theorems. -/
import bench.repairs.tritonbench_g.attn_fwd_causal.NativeIO
import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.NativePublicIO

namespace VeriTile.Bench.Repairs.AttnFwdCausal.FPEquiv
open VeriTile Triton FP.Structural Source Whole
noncomputable section

/-- The complete selected source, finite FP32 word inputs and a pure native
output observer at each feature's physical reduction owner. -/
abbrev kernelIO := NativeIO.program

/-- Source-to-independent-native observation over every legal complete grid.
TF32 QK, FP16 P/V and PV, FP32 carry, FMA order, both key folds and the FP16
output are retained. Instructions are the existing primitive model; no new
optimization, transformation law or primitive-conformance experiment is assumed. -/
theorem kernel_native (instructions : Native.Instructions) (p : Parameters)
    (apart : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    (kernelIO instructions p work).Implements (NativeIO.formula instructions p work) :=
  NativePublicIO.implements instructions p apart work legal

#axiomsClean kernel_native
#specNonCircular Reference.native avoiding [Whole.answer,Source.accumulated,Source.update,runPrograms,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdCausal.FPEquiv
