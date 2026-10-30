import bench.tritonbench_g.chunk_cumsum_vector.Proofs.CandidateExecution
import VeriTile.Triton.Float.ScalarArithmeticLaws

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.Rewrite
open VeriTile Triton FP.Structural FPValues
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

/-- The shared strict ADD-COMMUTE certificate, with no empirical assumptions. -/
theorem carry_rule :
    [FP.ScalarArithmetic.lhs .addCommute] ≡[([] : Spec.Assumptions FP.GuardedFragment)]
      [FP.ScalarArithmetic.rhs .addCommute] :=
  Spec.FloatingPoint.ofDerivation rfl trivial
    (FP.ScalarArithmetic.strict_admitted [] .addCommute rfl)

theorem carry_add (M : Algebra α)
    (D : FP.Guarded.Domain α) (model : FP.Guarded.Models [] M D)
    (s : State α) (a b : α) (ha : D .finite a) (hb : D .finite b) :
    M.binary (some .fp32) .real .add a b = M.binary (some .fp32) .real .add b a :=
  FP.ScalarArithmetic.add_comm_strict M D model s a b ha hb

/-- Actual FP32 carry and block-reduction operands, only on active feature
lanes and only when another output chunk reads the updated carry. These are
rewrite-site conditions; the Python input domain is unchanged. -/
def FiniteCarry (M : Algebra α) (D : FP.Guarded.Domain α) (h : Call) (c : Config)
    (p : Geometry.Program h) (initial : State α) : Prop :=
  ∀ q, q + 1 < Geometry.chunks h c → ∀ j : TileIndex [BS], p.feature*BS+j.1.val < h.S →
    D .finite (FPExecution.carry M h c p.feature p.row initial q j) ∧
    D .finite (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false
      (loaded M h c p.feature p.row initial q) j)

@[spec_rule] theorem carry_equal
    (M : Algebra α) (D : FP.Guarded.Domain α) (model : FP.Guarded.Models [] M D)
    (h : Call) (c : Config) (p : Geometry.Program h) (initial : State α)
    (finite : FiniteCarry M D h c p initial) (q : Nat) (hq : q < Geometry.chunks h c)
    (j : TileIndex [BS]) (hj : p.feature*BS+j.1.val < h.S) :
    FPExecution.carry M h c p.feature p.row initial q j =
      CandidateExecution.carry M h c p.feature p.row initial q j := by
  induction q with
  | zero => rfl
  | succ q ih =>
    have eq := ih (by omega)
    obtain ⟨ha,hb⟩ := finite q hq j hj
    exact (carry_add M D model initial _ _ ha hb).trans
      (congrArg (fun x => M.binary (some .fp32) .real .add
        (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false
          (loaded M h c p.feature p.row initial q) j) x) eq)

theorem expected_equal
    (M : Algebra α) (dot : DotBackend α) (D : FP.Guarded.Domain α)
    (model : FP.Guarded.Models [] M D)
    (h : Call) (c : Config) (positive : 0<c.BT) (p : Geometry.Program h) (initial : State α)
    (finite : FiniteCarry M D h c p initial) (time : Nat) (j : Fin BS)
    (ht : time<h.T) (hj : p.feature*BS+j.val<h.S) :
    FPExecution.expected M dot h c positive p initial time j =
      CandidateExecution.expected M dot h c positive p initial time j := by
  have q := (Geometry.coordinate_cover h.T c.BT time positive ht).1
  exact congrArg (fun carry => M.binary (some .fp32) .real .add carry
    (dot (some .fp32) {allowTF32:=some Bool.false} (batch:=[]) (triangular M c)
      (loaded M h c p.feature p.row initial (time/c.BT))
      (⟨time%c.BT,Nat.mod_lt _ positive⟩,j,PUnit.unit)))
    (carry_equal M D model h c p initial finite (time/c.BT) q (j,PUnit.unit) hj)

/-- Complete actual-source executions agree on every typed memory cell.
No rule is used for inactive feature carries or for the unused final update. -/
@[spec_rule] theorem successful_pair
    (M : Algebra α) (dot : DotBackend α) (D : FP.Guarded.Domain α)
    (model : FP.Guarded.Models [] M D)
    (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (positive : 0<c.BT)
    (p : Geometry.Program h) (u : State α) (hf : u.pids 0=p.feature) (hr : u.pids 1=p.row)
    (finite : FiniteCarry M D h c p u) :
    ∃ a b, FP.Structural.exec (machine M dot) (Kernels.source h c) u=some a ∧
      FP.Structural.exec (machine M dot) (Kernels.sumLeft h c) u=some b ∧
      a.mem=b.mem ∧ FPMemory.Frame h p u a ∧ FPMemory.Frame h p u b := by
  obtain ⟨a,ha,va,fa⟩ := FPExecution.source_correct M dot h layout c positive p u hf hr
  obtain ⟨b,hb,vb,fb⟩ := CandidateExecution.source_correct M dot h layout c positive p u hf hr
  refine ⟨a,b,ha,hb,?_,fa,fb⟩
  funext r off
  by_cases written : FPMemory.footprint h p (r,off)
  · obtain ⟨rfl,time,j,ht,hj,rfl⟩ := written
    rw [va time j ht hj,vb time j ht hj,expected_equal M dot D model h c positive p u finite time j ht hj]
  · exact (fa r off written).symm.trans (fb r off written)

#print_fp_assumptions carry_rule
#axiomsClean carry_add
#axiomsClean carry_equal
#axiomsClean expected_equal
#axiomsClean successful_pair
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.Rewrite
