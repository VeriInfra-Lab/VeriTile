import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.PairModel
import VeriTile.Triton.Float.ScalarOpsLaws

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.BlockFirst
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open SourceGraph (Expr Word Memory Parameters PlanSelector engine treeValue sumValues
  row col outputOffset aOffset bOffset blockSum Outside contract contractAdd evaluate)
noncomputable section
set_option maxHeartbeats 500000

/-- A contracted carry starts at zero and is subsequently add/FMA, so it can
never be mistaken for the reduction product by the left-product contraction. -/
def NotProduct (x : Expr) : Prop := ∀ a b, x ≠ .mul a b

theorem contractAdd_notProduct (x y : Expr) : NotProduct (contractAdd x y) := by
  cases x <;> cases y <;> simp [NotProduct,contractAdd]

theorem original_carry_notProduct (plans : PlanSelector) (p : Parameters)
    (s : State Word) (c : Nat) (i : TileIndex [p.BM,p.BN]) :
    NotProduct (contract (SourceGraph.accumulated plans p s c i)) := by
  cases c with
  | zero => simp [SourceGraph.accumulated,contract,NotProduct]
  | succ c => exact contractAdd_notProduct _ _

theorem candidate_carry_notProduct (plans : PlanSelector) (p : Parameters)
    (s : State Word) (c : Nat) (i : TileIndex [p.BM,p.BN]) :
    NotProduct (contract (accumulated plans p s c i)) := by
  cases c with
  | zero => simp [accumulated,contract,NotProduct]
  | succ c => exact contractAdd_notProduct _ _

theorem contractAdd_left (cfg : FP.Config) (a b : Expr) (ha : NotProduct a) :
    evaluate cfg (contractAdd a b) =
      match b with
      | .mul x y => FP.fma cfg (evaluate cfg x) (evaluate cfg y) (evaluate cfg a)
      | _ => FP.add cfg (evaluate cfg a) (evaluate cfg b) := by
  cases a <;> cases b <;> simp_all [NotProduct,contractAdd,evaluate]

theorem contractAdd_right (cfg : FP.Config) (a b : Expr) (hb : NotProduct b) :
    evaluate cfg (contractAdd a b) =
      match a with
      | .mul x y => FP.fma cfg (evaluate cfg x) (evaluate cfg y) (evaluate cfg b)
      | _ => FP.add cfg (evaluate cfg a) (evaluate cfg b) := by
  cases a <;> cases b <;> simp_all [NotProduct,contractAdd,evaluate]

/-- The product and FMA operand order are preserved. Only a genuine separate
addition uses strict FP32 add_comm; no fused/separate equality is assumed. -/
theorem carry_exchange (cfg : FP.Config) (a block b : Expr)
    (ha : NotProduct a) (hb : NotProduct b) (same : evaluate cfg a=evaluate cfg b) :
    evaluate cfg (contractAdd a block)=evaluate cfg (contractAdd block b) := by
  rw [contractAdd_left cfg a block ha,contractAdd_right cfg block b hb,same]
  cases block <;> simp only
  all_goals try exact FP.ScalarOpsLaws.add_comm cfg _ _

theorem accumulated_evaluate (cfg : FP.Config) (plans : PlanSelector)
    (p : Parameters) (s t : State Word) (hp : s.pids=t.pids)
    (ha : s.mem p.A=t.mem p.A) (hb : s.mem p.B=t.mem p.B)
    (c : Nat) (i : TileIndex [p.BM,p.BN]) :
    evaluate cfg (contract (SourceGraph.accumulated plans p s c i)) =
      evaluate cfg (contract (accumulated plans p t c i)) := by
  induction c with
  | zero => rfl
  | succ c ih =>
    have blocks : blockSum plans p s c i=blockSum plans p t c i := by
      simp only [blockSum,aOffset,bOffset,row,col,hp,ha,hb]
    change evaluate cfg (contractAdd
      (contract (SourceGraph.accumulated plans p s c i)) (contract (blockSum plans p s c i))) =
      evaluate cfg (contractAdd (contract (blockSum plans p t c i)) (contract (accumulated plans p t c i)))
    rw [blocks]
    exact carry_exchange cfg _ _ _
      (original_carry_notProduct plans p s c i) (candidate_carry_notProduct plans p t c i) ih

#axiomsClean accumulated_evaluate
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.BlockFirst
