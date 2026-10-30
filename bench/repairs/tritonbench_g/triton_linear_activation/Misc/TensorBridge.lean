import bench.repairs.tritonbench_g.triton_linear_activation.Correct
import bench.repairs.tritonbench_g.triton_linear_activation.Misc.NativeCorrect
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelPointer
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelEntries
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelBits

/-! Optional actual-Tensor pointer applications of the kernel preparation laws.
The exported declarations are ordinary supporting theorem aliases. -/
namespace VeriTile.Bench.Repairs.TritonLinearActivation.TensorBridge
open VeriTile.Triton FP.Structural KernelSource KernelValues KernelContract KernelTensor KernelEntries
noncomputable section

/-- Kernel-only ideal Real execution on actual prepared Tensor bytes. The
input-only projection does not assert floating execution equals ideal Real
arithmetic. All source stores and the complete tagged-cell frame are derived. -/
theorem correct (r : Request) (ts : Fin 4 → Checked.Observed) (flag : Bool)
    (admit : Checked.Domain r ts flag) (c : Fin 98) (regions : Fin 5 → RegionName)
    (prepared : Prepared r regions) (bytes : Nat → BitVec 8) (s : State ℝ)
    (binding : StorageBound r regions prepared (KernelBits.realAt r.mode bytes) s)
    (pid : s.pids 0<r.shape.programs c) :
    ∃ final, exec (engine KernelRealArithmetic.algebra KernelRealArithmetic.dot)
      (source r.mode (parameters r c regions)) s=some final ∧
      TensorResult r c regions prepared s final
        (denseOutput r regions prepared (KernelBits.realAt r.mode bytes))
        (densePre r regions prepared (KernelBits.realAt r.mode bytes)) ∧
      Formed r c regions s :=
  KernelEntries.correct r ts flag admit c regions prepared (KernelBits.realAt r.mode bytes) s binding pid


/-- Successful source/compiler-model execution and all actual Tensor output
words. Input words come from their exact four-byte or two-byte storage codec.
The compiler policy and named fixed ISA/libdevice ports are explicit machine
semantics, never whole-dot or completed-output equality hypotheses. -/
theorem fp_equiv (profile : LocalProfile) (policy : KernelNativeModel.CompilerPolicy)
    (r : Request) (ts : Fin 4 → Checked.Observed) (flag : Bool)
    (admit : Checked.Domain r ts flag) (c : Fin 98) (regions : Fin 5 → RegionName)
    (prepared : Prepared r regions) (bytes : Nat → BitVec 8) (s : State Word)
    (binding : StorageBound r regions prepared (KernelBits.wordAt r.mode bytes) s)
    (pid : s.pids 0<r.shape.programs c) :
    ∃ final, KernelNativeModel.execute profile policy c r.mode (parameters r c regions) s=some final ∧
      TensorResult r c regions prepared s final
        (KernelNativeTensor.resultWord profile policy r c regions prepared (KernelBits.wordAt r.mode bytes))
        (KernelNativeTensor.savedWord profile policy r c regions prepared (KernelBits.wordAt r.mode bytes)) ∧
      Formed r c regions s :=
  KernelEntries.fp_equiv profile policy r ts flag admit c regions prepared (KernelBits.wordAt r.mode bytes) s binding pid


export KernelGeometry (all_outputs_covered)
#axiomsClean correct
#axiomsClean fp_equiv
export KernelPointer (actual_pointer actual_address)
#axiomsClean actual_pointer
#axiomsClean actual_address
end
end VeriTile.Bench.Repairs.TritonLinearActivation.TensorBridge
