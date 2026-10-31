import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ForwardCorrect
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ZeroIO
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_ReferenceCongruence
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ForwardIO
open VeriTile VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

abbrev Layout := ForwardLayout

def column (p : Layout) (u : BlockState) (j : Fin p.B) : Nat := u.pids 0*p.B+j.val
def active (p : Layout) (u : BlockState) (j : Fin p.B) : Prop := column p u j<p.batch*p.dim
def offset (p : Layout) (u : BlockState) (t : Nat) (j : Fin p.B) : Nat := t*(p.batch*p.dim)+column p u j

structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : ∀ j, active p u j → column p u j < bounds p.s
  transition : ∀ j, active p u j → column p u j%p.dim < bounds p.l
  input : ∀ t, t<length → ∀ j, active p u j → offset p u t j < bounds p.x
  output : ∀ t, t<length → ∀ j, active p u j → offset p u t j < bounds p.y

/-- Independent scalar recurrence over the initial input memory. -/
def value (p : Layout) (u : BlockState) (t : Nat) (j : Fin p.B) : ℝ :=
  Reference.forwardState (u.readMem p.s (column p u j)) (u.readMem p.l (column p u j%p.dim))
    (fun q => u.readMem p.x (offset p u q j)) (t+1)

def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ t, t<length → ∀ j, active p u j → o≠A.addr p.y (offset p u t j)

/-- Uniform real contract, including zero length. The initial-state and
transition bounds remain required because their loads precede the loop. -/
def Implements (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  (0<length → p.x≠p.y) →
  ∀ A : FlatAlloc, A.Disjoint → A.regions=[p.s,p.l,p.x,p.y] →
    (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds p u length A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, active p u j → v.readMem A.flat (A.addr p.y (offset p u t j))=value p u t j) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ForwardIO
