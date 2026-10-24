import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelRealSaved
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBound
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelCoverage
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelRealBound
open VeriTile.Triton FP.Structural KernelProgram KernelTensor KernelBound
open VeriTile.Bench.Repairs.LayernormFwdTriton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 12000000

theorem forward_written (s : Shape) (f : Flags) (scale : ℝ) (p : Program .forward s) (u : State ℝ)
    (r : RegionName) (off : Nat) (value : ℝ) (expected : KernelReal.ForwardExpected s f scale p.head p.value u r off value) :
    KernelGeometry.Writes .forward s f p r off := by
  cases expected with
  | output t j active => exact Or.inl ⟨Or.inr rfl,t.val,j,t.isLt,active,rfl⟩
  | residual t j active => exact Or.inl ⟨Or.inl rfl,t.val,j,t.isLt,active,rfl⟩
  | terminal enabled i j active =>
    exact Or.inr ⟨enabled,rfl,⟨i.val,lt_of_lt_of_le i.isLt (Geometry.nextPower_ge s.K)⟩,j,i.isLt,active,rfl⟩
theorem backward_written (s : Shape) (f : Flags) (scale : ℝ) (p : Program .backward s)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (r : RegionName) (off : Nat) (value : ℝ) (expected : KernelReal.BackwardExpected s f scale p.head p.value a g terminal r off value) :
    KernelGeometry.Writes .backward s f p r off := by
  cases expected with
  | query t i active => exact Or.inl ⟨Or.inr rfl,t.val,i,t.isLt,active,rfl⟩
  | key t i active => exact Or.inl ⟨Or.inl rfl,t.val,i,t.isLt,active,rfl⟩
  | value t j active => exact Or.inr (Or.inl ⟨rfl,t.val,j,t.isLt,active,rfl⟩)
  | vectorBeta enabled t j active =>
    exact Or.inr (Or.inr (Or.inl ⟨rfl,by simp only [enabled,ite_true]; exact ⟨t.val,j,t.isLt,active,rfl⟩⟩))
  | scalarBeta enabled t =>
    exact Or.inr (Or.inr (Or.inl ⟨rfl,by simp only [enabled,Bool.false_eq_true,ite_false]; exact ⟨t.val,t.isLt,rfl⟩⟩))
  | seed enabled i j key value => exact Or.inr (Or.inr (Or.inr ⟨enabled,rfl,i,j,key,value,rfl⟩))

def ColumnVJP (s : Shape) (scale : ℝ) (a : Fin s.BBV → Math.Input s.BK)
    (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK) : Prop :=
  ∀ (j : Fin s.BBV) (d : Math.Input s.BK),HasDerivAt
    (fun e => Math.loss scale (Math.perturb (a j) d e) (fun t => g t j) (terminal j) s.T)
    (Math.dot (Math.carry scale (a j) (fun t => g t j) (terminal j) s.T 0) d.h0+
      ∑ t∈Finset.range s.T,Math.inputPair (KernelReal.gradient s scale a g terminal t j) d t) 0
theorem column_vjp (s : Shape) (scale : ℝ) (a : Fin s.BBV → Math.Input s.BK)
    (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK) : ColumnVJP s scale a g terminal :=
  fun j d => Math.loss_derivative scale (a j) d (fun t => g t j) (terminal j) s.T

theorem forward_tensor {Store Byte : Type} (api : Memory.API ℝ Store Byte) (initial : Store)
    (s : Shape) (f : Flags) (scale : ℝ) (b : Buffers .forward s f) (p : Program .forward s) (u : State ℝ)
    (hv : u.pids 0=p.value) (hk : u.pids 1=0) (hh : u.pids 2=p.head) :
    ∃ v,KernelForward.Facts KernelReal.algebra s f scale p.head p.value
      (physical b (fun _ _ => api.read initial) u) v ∧
      Shared.TensorImage api (placement b p) (fun _ _ cell => cell.read .real) v initial ∧
      InputFrame api b p (fun _ _ cell => cell.read .real) v initial ∧
      (∀ r off value,KernelReal.ForwardExpected s f scale p.head p.value
        (logical b (fun _ _ => api.read initial) u) r off value →
        api.read (Shared.commit api (placement b p) (fun _ _ cell => cell.read .real) v initial)
          ((b.tensor r).storage.address ((b.tensor r).physicalIndex off))=value) := by
  obtain ⟨v,facts,values⟩ := KernelReal.forward_complete s f scale p.head p.value
    (physical b (fun _ _ => api.read initial) u) hv hk hh
  rw [physical_eq_logical] at values
  refine ⟨v,facts,Shared.complete api (placement b p) _ v initial,input_frame api b p _ v initial,?_⟩
  intro r off value expected
  exact (VeriTile.Bench.Repairs.FastLayernorm.KernelStorage.commit_output api (placement b p)
    (fun _ _ cell => cell.read .real) v initial r off (forward_written s f scale p _ r off value expected)).trans
      (values r off value expected)

theorem backward_tensor {Store Byte : Type} (api : Memory.API ℝ Store Byte) (initial : Store)
    (s : Shape) (f : Flags) (scale : ℝ) (b : Buffers .backward s f) (p : Program .backward s) (u : State ℝ)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (saved : KernelReal.Saved s (KernelBackwardReference.input KernelReal.algebra s f p.head p.value
      (logical b (fun _ _ => api.read initial) u)) a g terminal)
    (hv : u.pids 0=p.value) (hk : u.pids 1=0) (hh : u.pids 2=p.head) :
    ColumnVJP s scale a g terminal ∧ ∃ v,
      KernelBackwardFacts.Facts KernelReal.algebra s f scale p.head p.value
        (physical b (fun _ _ => api.read initial) u) v ∧
      Shared.TensorImage api (placement b p) (fun _ _ cell => cell.read .real) v initial ∧
      InputFrame api b p (fun _ _ cell => cell.read .real) v initial ∧
      (∀ r off value,KernelReal.BackwardExpected s f scale p.head p.value a g terminal r off value →
        api.read (Shared.commit api (placement b p) (fun _ _ cell => cell.read .real) v initial)
          ((b.tensor r).storage.address ((b.tensor r).physicalIndex off))=value) := by
  rw [←physical_eq_logical] at saved
  obtain ⟨v,facts,values⟩ := KernelReal.backward_complete s f scale p.head p.value
    (physical b (fun _ _ => api.read initial) u) a g terminal saved hv hk hh
  refine ⟨column_vjp s scale a g terminal,v,facts,Shared.complete api (placement b p) _ v initial,
    input_frame api b p _ v initial,?_⟩
  intro r off value expected
  exact (VeriTile.Bench.Repairs.FastLayernorm.KernelStorage.commit_output api (placement b p)
    (fun _ _ cell => cell.read .real) v initial r off (backward_written s f scale p a g terminal r off value expected)).trans
      (values r off value expected)

#axiomsClean column_vjp
#axiomsClean forward_tensor
#axiomsClean backward_tensor
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelRealBound
