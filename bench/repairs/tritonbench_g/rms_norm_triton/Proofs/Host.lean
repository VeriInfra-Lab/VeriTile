import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.Native

namespace VeriTile.Bench.Repairs.RmsNormTriton.Host
open VeriTile.Triton FP.Structural VeriTile.Bench.TritonBenchG.RmsNormTriton
noncomputable section

/-- Device setup receives the owner of the actual contiguous input. The source
lexical device bracket restores the caller device on either setup outcome. -/
def contextRuntime (rt : Whole.Runtime α input weight n entry live)
    (setup : Nat → Whole.Prepared α input weight entry live → Except String (Whole.Schedule input n)) :
    Whole.Runtime α input weight n entry live :=
  {rt with setup := fun a => setup a.xc.tensor.device a}

structure Returned (α : Type) where
  outcome : Whole.Outcome α
  currentDevice : Nat

def checked [Inhabited α] (x w : TensorInfo) (n : NormalizedInfo) (eps : EpsilonInfo)
    (mdl : Algebra α) (rt : Whole.Runtime α input weight n.dimensions entry live)
    (setup : Nat → Whole.Prepared α input weight entry live → Except String (Whole.Schedule input n.dimensions))
    (api : ∀ a : Whole.Prepared α input weight entry live,PreparedAPI a n.dimensions)
    (callerDevice : Nat) : Except InputError (Returned α) :=
  match checkTensorInput x w n eps with
  | .error e => .error e
  | .ok _ => .ok ⟨Whole.executeHost mdl eps.value
      (checkedRuntime x n.dimensions (contextRuntime rt setup) api),callerDevice⟩

theorem owner (b : Binding x w input weight live)
    (a : Whole.Prepared α input weight entry live) : a.xc.tensor.device=x.device.index :=
  a.xc.device.trans b.input.device

theorem rejected [Inhabited α] (x w : TensorInfo) (n : NormalizedInfo) (eps : EpsilonInfo)
    (bad : ¬InputContract x w n eps) (mdl : Algebra α)
    (rt : Whole.Runtime α input weight n.dimensions entry live)
    (setup : Nat → Whole.Prepared α input weight entry live → Except String (Whole.Schedule input n.dimensions))
    (api : ∀ a : Whole.Prepared α input weight entry live,PreparedAPI a n.dimensions) (callerDevice : Nat) :
    ∃ e,checked x w n eps mdl rt setup api callerDevice=.error e := by
  cases result : checkTensorInput x w n eps with
  | error e => exact ⟨e,by simp [checked,result]⟩
  | ok u => cases u; exact False.elim (bad ((checkTensorInput_iff x w n eps).mp result))

end
end VeriTile.Bench.Repairs.RmsNormTriton.Host
