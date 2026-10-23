import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Proofs.IOCorrect
import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Proofs.IONative

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
noncomputable section
set_option maxHeartbeats 1000000

/-- Every previous physical input induces these logical channels, including
input aliases and native output-stride collisions. No domain restriction is
introduced by changing the public presentation to finite tensor arrays. -/
def memoryInputs (x : Layout) (m : Memory A) : Inputs A x :=
  fun port => match port with
  | ⟨0,_⟩ => fun i => logical m .q x.q i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val
  | ⟨1,_⟩ => fun i => logical m .k x.k i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val
  | ⟨2,_⟩ => fun i => logical m .v x.v i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val
  | ⟨n+3,h⟩ => False.elim (by simp [inputs] at h)

def realState (x : Layout) (m : Memory ℝ) : State KernelReal.R :=
  stateAt (parameters x) (0,0) (liftMemory (Real.lift m))
def nativeState (x : Layout) (m : Memory Word) : State KernelNative.Expr :=
  stateAt (parameters x) (0,0) (KernelNative.initialMemory m)

theorem real_domain (x : Layout) (m : Memory ℝ) :
    Logical.RealInputs x (memoryInputs x (Real.lift m)) := by
  intro port i
  fin_cases port <;> exact ⟨_,rfl⟩

theorem real_binding (x : Layout) (scale : ℝ) (m : Memory ℝ) :
    (realIO x scale).InputBinding (memoryInputs x (Real.lift m)) (realState x m) := by
  intro port i _
  fin_cases port <;> rfl

theorem native_binding (I : KernelNative.Instructions) (x : Layout) (scale : Checked.ScalarABI)
    (m : Memory Word) :
    (nativeIO I x scale).InputBinding (memoryInputs x m) (nativeState x m) := by
  intro port i _
  fin_cases port <;> rfl

theorem native_output_target (I : KernelNative.Instructions) (x : Layout) (scale : Checked.ScalarABI)
    (m : Memory Word) (i : TileIndex [x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim]) :
    Logical.nativeFormula I x scale (memoryInputs x m) (0 : Fin 2) i=
      KernelNative.target I x scale m "O"
        (FlashAttn.address x.o i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val) := by
  have value := grid_output_tensor KernelNative.algebra KernelNative.dot Logical.wordLeaf x (KernelNative.scalar scale)
    (Logical.nativeObservation I x) (memoryInputs x m) (nativeState x m) (native_binding I x scale m) i
  dsimp only [nativeState,stateAt] at value
  unfold KernelNative.target KernelNative.observe KernelNative.reference
  rw [value]
  rfl

theorem native_log_target (I : KernelNative.Instructions) (x : Layout) (scale : Checked.ScalarABI)
    (m : Memory Word) (i : TileIndex [x.shape.batches,x.shape.heads,x.shape.length]) :
    Logical.nativeFormula I x scale (memoryInputs x m) (1 : Fin 2) i=
      KernelNative.target I x scale m "L" (FlashAttn.lAddress x.shape i.1.val i.2.1.val i.2.2.1.val) := by
  have value := grid_log_tensor KernelNative.algebra KernelNative.dot Logical.wordLeaf x (KernelNative.scalar scale)
    (Logical.nativeObservation I x) (memoryInputs x m) (nativeState x m) (native_binding I x scale m) i
  dsimp only [nativeState,stateAt] at value
  unfold KernelNative.target KernelNative.observe KernelNative.reference
  rw [value]
  rfl

theorem real_output_target (x : Layout) (hs : x.shape.Valid) (ho : OutputInjective x) (scale : ℝ)
    (m : Memory ℝ) (i : TileIndex [x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim]) :
    Logical.mathematical x scale (memoryInputs x (Real.lift m)) (0 : Fin 2) i=
      some (Spec.output x m scale i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val) := by
  have value := grid_output_injective KernelReal.algebra KernelReal.dot id x (some scale) (fun _ _ => id)
    (memoryInputs x (Real.lift m)) (realState x m) (real_binding x scale m) hs ho i
  rw [real_output_formula x scale _ _ (real_binding x scale m) hs ho (real_domain x m) i] at value
  dsimp only [realState,stateAt] at value
  have old := KernelReal.grid_output x hs ho m scale i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val
    ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt,i.2.2.2.1.isLt⟩
  rw [value] at old
  exact congrArg (fun c : Cell KernelReal.R => c.read .real) old

theorem real_log_target (x : Layout) (hs : x.shape.Valid) (scale : ℝ) (m : Memory ℝ)
    (i : TileIndex [x.shape.batches,x.shape.heads,x.shape.length]) :
    Logical.mathematical x scale (memoryInputs x (Real.lift m)) (1 : Fin 2) i=
      some (Spec.lse x m scale i.1.val i.2.1.val i.2.2.1.val) := by
  have value := grid_log_tensor KernelReal.algebra KernelReal.dot id x (some scale) (fun _ _ => id)
    (memoryInputs x (Real.lift m)) (realState x m) (real_binding x scale m) i
  rw [real_log_formula x scale _ _ (real_binding x scale m) hs (real_domain x m) i] at value
  dsimp only [realState,stateAt] at value
  have old := KernelReal.grid_log x hs m scale i.1.val i.2.1.val i.2.2.1.val
    ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt⟩
  rw [value] at old
  exact congrArg (fun c : Cell KernelReal.R => c.read .real) old

#axiomsClean real_domain
#axiomsClean native_binding
#axiomsClean native_output_target
#axiomsClean native_log_target
#axiomsClean real_output_target
#axiomsClean real_log_target
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
