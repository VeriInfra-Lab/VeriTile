import bench.repairs.tritonbench_g.flash_attn.KernelOnly.Layout
import bench.repairs.tritonbench_g.flash_attn.Proofs.RealComplete

namespace VeriTile.Bench.Repairs.FlashAttn.KernelReal
open VeriTile Triton FP.Structural ActualSource
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
abbrev R := WithBot ℝ

def binary : Binary→R→R→R
  | .add=>WithBot.realAdd | .sub=>WithBot.realSub | .mul=>WithBot.realMul
  | .div=>WithBot.realDiv | .max=>max | _=>fun _ _=>none

def reductionInputs (shape : TileShape) (ax : Fin shape.length) (keep : Bool)
    (xs : Values R .real shape) :
    TileIndex (TileShape.reduceShape shape ax keep)→Fin (TileShape.axisDim shape ax)→R :=
  match keep with
  | Bool.false=>fun i j=>xs (TileShape.insertAxisIndex shape ax i j)
  | Bool.true=>fun i j=>xs (TileShape.replaceAxisIndex shape ax i j)

def reduce (op : R→R→R) {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values R .real shape) (i : TileIndex (TileShape.reduceShape shape ax keep)) : R :=
  (Reduction.plan 0).eval op (fun j=>if h : j<TileShape.axisDim shape ax then
    reductionInputs shape ax keep xs i ⟨j,h⟩ else none)

def algebra : Algebra R where
  literal:=fun _ _ x=>some x
  negInf:=none
  binary:=fun _ _=>binary
  unary:=fun _ op=>if op=.exp2 then Real.exp2 else if op=.log2 then WithBot.realLog2 else id
  cast:=fun _ _ _=>id
  fromNat:=fun _ n=>some n
  fromInt:=fun _ n=>some n
  fp32Bits:=fun _=>some 0
  fp32Load:=id
  reduceMax:=fun _ {_} ax keep xs=>reduce max ax keep xs
  reduceSum:=fun _ {_} ax keep xs=>reduce WithBot.realAdd ax keep xs

def dotAt : (batch : TileShape)→{m k n : Nat}→
    Values R .real (batch++[m,k])→Values R .real (batch++[k,n])→Values R .real (batch++[m,n])
  | [],_,k,_,a,b,i=>some (RealSum.sumReal (fun j=>if h : j<k then
      (a (i.1,⟨j,h⟩,())).unbotD 0*(b (⟨j,h⟩,i.2.1,())).unbotD 0 else 0) k)
  | _::ds,_,_,_,a,b,i=>dotAt ds (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) i.2

def dot : ActualSource.Dot R where
  apply:=fun _ left right out _ {batch} {_} {_} {_} a b i=>
    ofFloat out (dotAt batch (fun j=>toFloat left (a j)) (fun j=>toFloat right (b j)) i)

def state (x : Layout) (m : Memory ℝ) (pid head : Nat) : State R :=
  stateAt (parameters x) (pid,head) (liftMemory (Real.lift m))

theorem reduction_congr (t : Reduction.Tree) (op : α→α→α) (f g : Nat→α)
    (h : ∀ j,j<64→f j=g j) : t.eval op f=t.eval op g := by
  induction t with
  | leaf i=>exact h i.val i.isLt
  | merge a b ha hb=>simp [Reduction.Tree.eval,ha,hb]
  | shuffle n a ha=>exact ha

theorem reduce_row (op : R→R→R) (xs : TileIndex [128,64]→R) (i : Fin 128) :
    reduce op ⟨1,by simp⟩ Bool.false xs (i,())=
      (Reduction.plan 0).eval op (fun j=>xs (i,⟨j%64,Nat.mod_lt _ (by decide)⟩,())) := by
  apply reduction_congr
  intro j hj
  simp [TileShape.axisDim,reductionInputs,TileShape.insertAxisIndex,hj,Nat.mod_eq_of_lt hj]

theorem dot_some {m k n : Nat} (a : Fin m→Nat→ℝ) (b : Nat→Fin n→ℝ) (i : TileIndex [m,n]) :
    dotAt [] (k:=k) (fun j=>some (a j.1 j.2.1.val)) (fun j=>some (b j.1.val j.2.1)) i=
      some (RealSum.sumReal (fun j=>a i.1 j*b j i.2.1) k) := by
  unfold dotAt
  congr 1
  exact RealSum.sum_congr _ _ k (fun j hj=>by simp [hj,WithBot.unbotD_coe])

theorem query_value (x : Layout) (m : Memory ℝ) (scale : ℝ) (pid head : Nat)
    (i : TileIndex [128,(parameters x).DIM]) :
    ActualSource.query algebra (parameters x) (state x m pid head) (some scale) i=
      some (if pid*128+i.1.val<x.shape.length then
        scale*1.44269504*logical m .q x.q (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.1.val) i.2.1.val else 0) := by
  by_cases hi : pid*128+i.1.val<x.shape.length
  all_goals simp [ActualSource.query,qptr,check0,state,parameters,stateAt,base,algebra,binary,
    liftMemory,sourceRegion,Real.lift,zero,Cell.read,WithBot.realMul,logical,FlashAttn.address,hi,mul_comm]

#axiomsClean reduce_row
#axiomsClean dot_some
#axiomsClean query_value
end
end VeriTile.Bench.Repairs.FlashAttn.KernelReal
