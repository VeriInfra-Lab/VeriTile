import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.RealStep
import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.Memory
namespace VeriTile.Bench.Repairs.BlockSparseAttn.Real
open TritonBenchG.BlockSparseAttn.StreamingMath
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 8000000

def rawValue (x : Layout) (m : Memory ℝ) (b h row col visit j : Nat) : ℝ :=
  let key := x.csr.key x.shape h row visit j
  if x.evenN || decide (key<(x.shape.length:Int)) then
    (inputs x m).value b (kvHead x.shape h) key col else 0

theorem v_real (x : Layout) (m : Memory ℝ) (b h row col visit j : Nat) :
    vLoad x ops (lift m) b h (x.csr.key x.shape h row visit j) col=
      some (rawValue x m b h row col visit j) := by
  unfold vLoad rawValue
  dsimp only
  split <;> rfl

theorem fold_real (x : Layout) (m : Memory ℝ) (scale : ℝ) (w : Work)
    (hr : w.row<x.shape.length) (ad : x.shape.blockD/8*8=x.shape.blockD)
    (an : x.shape.blockN/8*8=x.shape.blockN) (count : Nat) :
    (List.range count).foldl (blockStep x ops (lift m) (some scale) w.batch w.head w.row w.col)
      (initial ops x.reduction.replicas)=
    state (Spec.score x.shape x.csr (inputs x m) scale w.batch w.head w.row)
      (rawValue x m w.batch w.head w.row w.col) count x.reduction.replicas := by
  induction count with
  | zero=>rfl
  | succ count ih=>
    rw [List.range_succ,List.foldl_append,ih]
    simp only [List.foldl_cons,List.foldl_nil,blockStep]
    rw [show score x ops (lift m) (some scale) w.batch w.head w.row count=
      Spec.score x.shape x.csr (inputs x m) scale w.batch w.head w.row count from
        funext (fun j=>score_real x m scale w.batch w.head w.row count j hr ad)]
    simp_rw [v_real x m w.batch w.head w.row w.col count]
    exact normalized_real x.reduction x.valueInstruction an _ _ w.col count

theorem weighted_raw (x : Layout) (m : Memory ℝ) (scale : ℝ) (w : Work)
    (hr : w.row<x.shape.length) (visit : Nat) (j : Fin x.shape.blockN) :
    weight (Spec.score x.shape x.csr (inputs x m) scale w.batch w.head w.row visit j)*
      rawValue x m w.batch w.head w.row w.col visit j.val=
    Spec.weight x.shape x.csr (inputs x m) scale w.batch w.head w.row visit j*
      (inputs x m).value w.batch (kvHead x.shape w.head)
        (x.csr.key x.shape w.head w.row visit j.val) w.col := by
  by_cases vis : x.csr.key x.shape w.head w.row visit j.val≤(w.row:Int)
  · have hk : x.csr.key x.shape w.head w.row visit j.val<(x.shape.length:Int) := by omega
    simp [Spec.score,Spec.weight,vis,rawValue,hk,weight,VeriTile.Triton.WithBot.realExp]
  · simp [Spec.score,Spec.weight,vis,weight,VeriTile.Triton.WithBot.realExp]

theorem row_real (x : Layout) (m : Memory ℝ) (scale : ℝ) (w : Work)
    (hr : w.row<x.shape.length) (ad : x.shape.blockD/8*8=x.shape.blockD)
    (an : x.shape.blockN/8*8=x.shape.blockN) :
    rowValue x ops (lift m) (some scale) w=
      some (Spec.output x.shape x.csr (inputs x m) scale w.batch w.head w.row w.col) := by
  rw [rowValue,fold_real x m scale w hr ad an]
  simp only [state]
  rw [accumulated_eq_formula]
  have hw (visit : Nat) (j : Fin x.shape.blockN) :
      weight (Spec.score x.shape x.csr (inputs x m) scale w.batch w.head w.row visit j)=
        Spec.weight x.shape x.csr (inputs x m) scale w.batch w.head w.row visit j := by
    unfold Spec.score Spec.weight
    dsimp only
    split <;> rfl
  simp only [moment,mass,Spec.output]
  simp_rw [weighted_raw x m scale w hr]
  simp only [hw]

theorem whole_real (x : Layout) (m : Memory ℝ) (scale : ℝ) (ws : List Work)
    (complete : Complete x.shape ws) (ad : x.shape.blockD/8*8=x.shape.blockD)
    (an : x.shape.blockN/8*8=x.shape.blockN) (w : Work) (active : w.active x.shape) :
    execute x ops (some scale) ws (lift m) .o (w.target x.shape)=
      some (Spec.output x.shape x.csr (inputs x m) scale w.batch w.head w.row w.col) := by
  rw [call_value x ops (some scale) ws complete (lift m) w active]
  exact row_real x m scale w active.2.2.1 ad an

#axiomsClean v_real
#axiomsClean fold_real
#axiomsClean weighted_raw
#axiomsClean row_real
#axiomsClean whole_real
end
end VeriTile.Bench.Repairs.BlockSparseAttn.Real
