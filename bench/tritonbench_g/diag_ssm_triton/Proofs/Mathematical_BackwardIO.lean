import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_BackwardCorrect
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ZeroIO
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ReferenceCongruence
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.BackwardIO
open VeriTile VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000

abbrev Layout := BackwardLayout

def column (p : Layout) (u : BlockState) (j : Fin p.B) : Nat := u.pids 0*p.B+j.val
def active (p : Layout) (u : BlockState) (j : Fin p.B) : Prop := column p u j<p.batch*p.dim
def offset (p : Layout) (u : BlockState) (t : Nat) (j : Fin p.B) : Nat := t*(p.batch*p.dim)+column p u j

/-- Sufficient region separation for the reused positive-length stream proof. -/
structure Separation (p : Layout) : Prop where
  initial : p.gx≠p.s
  output : p.gx≠p.y
  gradient : p.gx≠p.gy
  transition : p.gx≠p.l
  initialGradient : p.gs≠p.gx
  transitionGradient : p.gl≠p.gx
  terminal : p.gl≠p.gs

structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : 0<length → ∀ j, active p u j → column p u j < bounds p.s
  transition : ∀ j, active p u j → column p u j%p.dim < bounds p.l
  previous : ∀ t, t+1<length → ∀ j, active p u j → offset p u t j < bounds p.y
  gradOutput : ∀ t, t<length → ∀ j, active p u j → offset p u t j < bounds p.gy
  gradInput : ∀ t, t<length → ∀ j, active p u j → offset p u t j < bounds p.gx
  gradInitial : ∀ j, active p u j → column p u j < bounds p.gs
  gradTransition : ∀ j, active p u j → column p u j < bounds p.gl

def inputValue (p : Layout) (u : BlockState) (length t : Nat) (j : Fin p.B) : ℝ :=
  Reference.inputGradient length (u.readMem p.l (column p u j%p.dim))
    (fun q => u.readMem p.gy (offset p u q j)) t
def initialValue (p : Layout) (u : BlockState) (length : Nat) (j : Fin p.B) : ℝ :=
  Reference.reverseCarry length (u.readMem p.l (column p u j%p.dim))
    (fun q => u.readMem p.gy (offset p u q j)) length
def transitionValue (p : Layout) (u : BlockState) (length : Nat) (j : Fin p.B) : ℝ :=
  Reference.transitionGradient length (u.readMem p.l (column p u j%p.dim)) (u.readMem p.s (column p u j))
    (fun q => u.readMem p.y (offset p u q j)) (fun q => u.readMem p.gy (offset p u q j)) length

def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨
    (∀ t, t<length → ∀ j, active p u j → o≠A.addr p.gx (offset p u t j)) ∧
    (∀ j, active p u j → o≠A.addr p.gs (column p u j)) ∧
    (∀ j, active p u j → o≠A.addr p.gl (column p u j))

/-- Zero length still reads the transition and writes both terminal gradients.
No output separation is required for that constant-zero path. -/
def Implements (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  (0<length → Separation p) →
  ∀ A : FlatAlloc, A.Disjoint → A.regions=[p.s,p.l,p.y,p.gs,p.gx,p.gl,p.gy] →
    (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds p u length A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, active p u j → v.readMem A.flat (A.addr p.gx (offset p u t j))=inputValue p u length t j) ∧
      (∀ j, active p u j → v.readMem A.flat (A.addr p.gs (column p u j))=initialValue p u length j) ∧
      (∀ j, active p u j → v.readMem A.flat (A.addr p.gl (column p u j))=transitionValue p u length j) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)

/-- Snapshot values supplied to the internal finite-stream proof. -/
def inputs (p : Layout) (u : BlockState) (length : Nat) (i : Fin 4) (t : Fin length) (j : Fin p.B) : ℝ :=
  match i with
  | ⟨0,_⟩ => u.readMem p.gy (offset p u t.val j)
  | ⟨1,_⟩ => u.readMem p.y (offset p u (t.val-1) j)
  | ⟨2,_⟩ => u.readMem p.s (column p u j)
  | ⟨_+3,_⟩ => u.readMem p.l (column p u j%p.dim)

theorem gradients_value (p : Layout) (u : BlockState) (length : Nat) (hp : 0<length)
    (o : Fin 3) (t : Fin length) (j : Fin p.B) :
    BackwardReference.gradients (inputs p u length) o t j =
      match o with
      | ⟨0,_⟩ => inputValue p u length t.val j
      | ⟨1,_⟩ => initialValue p u length j
      | ⟨_+2,_⟩ => transitionValue p u length j := by
  have hg (q : Nat) (hq : q<length) :
      BackwardReference.extend (fun v : Fin length => u.readMem p.gy (offset p u v.val j)) q =
        u.readMem p.gy (offset p u q j) := by simp [BackwardReference.extend,hq]
  have hy (q : Nat) (hq : q+1<length) :
      BackwardReference.extend (fun v : Fin length => u.readMem p.y (offset p u (v.val-1) j)) (q+1) =
        u.readMem p.y (offset p u q j) := by simp [BackwardReference.extend,hq]
  have hs : BackwardReference.extend (fun _ : Fin length => u.readMem p.s (column p u j)) 0 =
      u.readMem p.s (column p u j) := by simp [BackwardReference.extend,hp]
  fin_cases o
  · simp only [BackwardReference.gradients,inputs,inputValue,Reference.inputGradient]
    rw [hg t.val t.isLt]
    congr 1
    exact ReferenceCongruence.carry_prefix length _ _ _ (length-1-t.val) (by omega) hg
  · simp only [BackwardReference.gradients,inputs,initialValue]
    exact ReferenceCongruence.carry_prefix length _ _ _ length (le_refl _) hg
  · simp only [BackwardReference.gradients,inputs,transitionValue]
    rw [hs]
    exact ReferenceCongruence.transition_prefix length _ _ _ _ _ _ length (le_refl _) hy hg

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.BackwardIO
