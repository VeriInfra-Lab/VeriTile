import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.KernelNative
import bench.tritonbench_g.rms_norm_triton.Proofs.WholeIndex
namespace VeriTile.Bench.Repairs.RmsNormTriton
open VeriTile.Bench.TritonBenchG.RmsNormTriton
noncomputable section
/-- Insert the actual post-materialization pointer checks before ordinary
device/launch setup. Copy and allocation behavior is unchanged. -/
def checkedRuntime (x : TensorInfo) (n : List Nat)
    (rt : Whole.Runtime α input weight n entry live)
    (api : ∀ a : Whole.Prepared α input weight entry live,PreparedAPI a n) :
    Whole.Runtime α input weight n entry live :=
  {rt with setup := fun a =>
    if checkPointers x n (api a).pointers=.ok () then rt.setup a else .error "unsupported native pointers"}

def runtimeAdmission (x w : TensorInfo) (n : NormalizedInfo) (eps : EpsilonInfo)
    (c : InputContract x w n eps) (binding : Binding x w input weight live)
    (rt : Whole.Runtime α input weight n.dimensions entry live)
    (api : ∀ a : Whole.Prepared α input weight entry live,PreparedAPI a n.dimensions) :
    Whole.RuntimeAdmission (checkedRuntime x n.dimensions rt api) := by
  intro xc wc output schedule hx hw ha hs
  have checked : checkPointers x n.dimensions (api ⟨xc,wc,output⟩).pointers=.ok () := by
    by_contra no
    simp [checkedRuntime,no] at hs
  exact PreparedAPI.finite x w n eps c binding _ (api _) ((checkPointers_iff _ _ _).mp checked)

#axiomsClean runtimeAdmission
end
end VeriTile.Bench.Repairs.RmsNormTriton
