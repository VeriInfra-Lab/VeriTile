import bench.tritonbench_g.block_sparse_attn.Proofs.Kernels
import bench.tritonbench_g.flash_attn.Proofs.RealBackend
import bench.tritonbench_g.block_sparse_attn.Proofs.Mathematical_RealNormalizationValues
namespace VeriTile.Bench.Repairs.BlockSparseAttn.Real
open VeriTile.Triton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
abbrev R := WithBot ℝ
abbrev SM := TritonBenchG.BlockSparseAttn.StreamingMath.weight
def ops : Ops R where
  zero:=some 0
  negInf:=none
  add:=WithBot.realAdd
  sub:=WithBot.realSub
  mul:=WithBot.realMul
  fma:=fun a b c=>WithBot.realAdd (WithBot.realMul a b) c
  max:=max
  exp:=WithBot.realExp
  div:=WithBot.realDiv
  mma8:=fun _=>FlashAttn.Real.mma8 .n64
def lift (m : Memory ℝ) : Memory R := fun r a=>some (m r a)
def inputs (x : Layout) (m : Memory ℝ) : Inputs where
  query:=fun b h row col=>m .q (address x.q b h row col)
  key:=fun b h row col=>m .k (signedAddress x.k b h row col).toNat
  value:=fun b h row col=>m .v (signedAddress x.v b h row col).toNat

theorem dot_some (instruction : Instruction) (n : Nat) (a b : Nat → ℝ) (seed : ℝ) :
    dot ops instruction n (fun i=>some (a i)) (fun i=>some (b i)) (some seed)=
      some (seed+FlashAttn.RealSum.sumReal (fun i=>a i*b i) (n/8*8)) := by
  exact FlashAttn.Real.dot_some .n64 n a b seed

theorem tree_sum (t : Tree n) (f : Fin n → ℝ) :
    t.eval WithBot.realAdd (fun i=>some (f i))=some ((t.leaves.map f).sum) := by
  induction t with
  | leaf i=>simp [Tree.eval,Tree.leaves]
  | merge a b ha hb=>
    simp only [Tree.eval,ha,hb,Tree.leaves,List.map_append,List.sum_append]
    rfl

theorem reduce_sum (plan : Reduction n) (o : Fin plan.replicas) (f : Fin n → ℝ) :
    (plan.tree o).eval ops.add (fun i=>some (f i))=some (∑ i,f i) := by
  rw [show ops.add=WithBot.realAdd from rfl,tree_sum]
  have he:=((plan.coverage o).map f).sum_eq
  rw [he]
  simp [List.finRange,List.map_ofFn,List.sum_ofFn]

theorem tree_upper (t : Tree n) (f : Fin n → R) (z : R)
    (h : ∀ i∈t.leaves,f i≤z) : t.eval max f≤z := by
  induction t with
  | leaf i=>exact h i (by simp [Tree.leaves])
  | merge a b ha hb=>
    exact max_le (ha (fun i hi=>h i (by simp [Tree.leaves,hi])))
      (hb (fun i hi=>h i (by simp [Tree.leaves,hi])))
theorem tree_member_le (t : Tree n) (f : Fin n → R) (i : Fin n) (h : i∈t.leaves) :
    f i≤t.eval max f := by
  induction t with
  | leaf j=>
    have he : i=j := by simpa [Tree.leaves] using h
    subst i
    exact le_rfl
  | merge a b ha hb=>
    rcases List.mem_append.mp h with h|h
    · exact le_trans (ha h) (le_max_left _ _)
    · exact le_trans (hb h) (le_max_right _ _)
theorem reduce_max (plan : Reduction n) (o : Fin plan.replicas) (f : Fin n → R) :
    (plan.tree o).eval ops.max f=Finset.univ.sup f := by
  apply le_antisymm
  · exact tree_upper _ f _ (fun i _=>Finset.le_sup (Finset.mem_univ i))
  · apply Finset.sup_le
    intro i _
    exact tree_member_le _ f i ((plan.coverage o).mem_iff.mpr (List.mem_finRange i))

theorem q_real (x : Layout) (m : Memory ℝ) (b h row col : Nat) (hr : row<x.shape.length) :
    qLoad x ops (lift m) b h row col=some ((inputs x m).query b h row col) := by
  simp [qLoad,hr,lift,inputs]
theorem k_real (x : Layout) (m : Memory ℝ) (b h : Nat) (key : Int) (col : Nat)
    (hk : key<(x.shape.length:Int)) :
    kLoad x ops (lift m) b h key col=some ((inputs x m).key b (kvHead x.shape h) key col) := by
  simp [kLoad,hk,lift,inputs]

theorem scoreDot_real (x : Layout) (m : Memory ℝ) (b h row : Nat) (key : Int)
    (hr : row<x.shape.length) (hk : key<(x.shape.length:Int))
    (aligned : x.shape.blockD/8*8=x.shape.blockD) :
    scoreDot x ops (lift m) b h row key=some (Spec.dot x.shape (inputs x m) b h row key) := by
  have stepEq (acc : ℝ) (block : Nat) :
      dot ops x.scoreInstruction x.shape.blockD
        (fun d=>qLoad x ops (lift m) b h row (block*x.shape.blockD+d))
        (fun d=>kLoad x ops (lift m) b h key (block*x.shape.blockD+d)) (some acc)=
      some (acc+FlashAttn.RealSum.sumReal (fun d=>
        (inputs x m).query b h row (block*x.shape.blockD+d)*
        (inputs x m).key b (kvHead x.shape h) key (block*x.shape.blockD+d)) x.shape.blockD) := by
    simp_rw [q_real x m b h row _ hr,k_real x m b h key _ hk]
    rw [dot_some,aligned]
  have foldEq (ns : List Nat) (seed : ℝ) :
      ns.foldl (fun acc block=>dot ops x.scoreInstruction x.shape.blockD
        (fun d=>qLoad x ops (lift m) b h row (block*x.shape.blockD+d))
        (fun d=>kLoad x ops (lift m) b h key (block*x.shape.blockD+d)) acc) (some seed)=
      some (ns.foldl (fun acc block=>acc+FlashAttn.RealSum.sumReal (fun d=>
        (inputs x m).query b h row (block*x.shape.blockD+d)*
        (inputs x m).key b (kvHead x.shape h) key (block*x.shape.blockD+d)) x.shape.blockD) seed) := by
    induction ns generalizing seed with
    | nil=>rfl
    | cons block ns ih=>simp only [List.foldl_cons,stepEq,ih]
  unfold scoreDot
  rw [show ops.zero=(some 0 : R) from rfl,foldEq]
  exact congrArg some (Spec.scoreLoop_full x.shape (inputs x m) b h row key)

theorem score_real (x : Layout) (m : Memory ℝ) (scale : ℝ) (b h row visit : Nat)
    (j : Fin x.shape.blockN) (hr : row<x.shape.length)
    (aligned : x.shape.blockD/8*8=x.shape.blockD) :
    score x ops (lift m) (some scale) b h row visit j=
      Spec.score x.shape x.csr (inputs x m) scale b h row visit j := by
  unfold score Spec.score
  dsimp only
  by_cases vis : x.csr.key x.shape h row visit j.val≤(row:Int)
  · have hk : x.csr.key x.shape h row visit j.val<(x.shape.length:Int) := by omega
    rw [if_pos vis,if_pos vis,scoreDot_real x m b h row _ hr hk aligned]
    simp [ops,WithBot.realMul,WithBot.realAdd,mul_comm]
  · rw [if_neg vis,if_neg vis]
    cases he : scoreDot x ops (lift m) b h row (x.csr.key x.shape h row visit j.val) <;>
      simp [ops,WithBot.realMul,WithBot.realAdd]

#axiomsClean dot_some
#axiomsClean reduce_sum
#axiomsClean reduce_max
#axiomsClean scoreDot_real
#axiomsClean score_real
end
end VeriTile.Bench.Repairs.BlockSparseAttn.Real
