import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Misc.NativeCorrect
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Correct
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.FPEquiv
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.KernelEndpoints

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.TensorBridge
open VeriTile.Triton FP.Structural Checked KernelTensor
noncomputable section

theorem fused_recurrent_hgrn_correct (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (t : Arguments c U S V runtime)
    (BDF BDB : Nat) (tileF : tileCheck BDF=Bool.true) (tileB : tileCheck BDB=Bool.true)
    (memory : Role → Nat → ℝ) (other : State ℝ) :
    ∃ f b, Kernel.Claim realOps c U S V runtime t BDF BDB memory other f b ∧
      Kernel.Derivative c U S V runtime t memory b :=
  Kernel.real_complete c U S V runtime accepted t BDF BDB tileF tileB memory other

theorem fused_recurrent_hgrn_fp32 (p : Native.Instructions) (laws : Native.LocalConformance p)
    (schedule : Native.Schedule) (c : Shape) (U S V : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (t : Arguments c U S V runtime)
    (BDF BDB : Nat) (tileF : tileCheck BDF=Bool.true) (tileB : tileCheck BDB=Bool.true)
    (words : Role → Nat → BitVec 32) (other : State Native.Term) :
    ∃ f b, Kernel.Claim Native.sourceOps c U S V runtime t BDF BDB (Kernel.terms words) other f b ∧
      Kernel.NativeValues p schedule c U S V runtime t words b ∧
      Kernel.NativeTensorResult p schedule c U S V runtime t b :=
  Kernel.native_complete p laws schedule c U S V runtime accepted t BDF BDB tileF tileB words other

#axiomsClean fused_recurrent_hgrn_correct
#axiomsClean fused_recurrent_hgrn_fp32
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.TensorBridge
