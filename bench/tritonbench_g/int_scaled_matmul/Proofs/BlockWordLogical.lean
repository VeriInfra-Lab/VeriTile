import bench.tritonbench_g.int_scaled_matmul.Proofs.BlockWordComplete
import bench.tritonbench_g.int_scaled_matmul.Sources.LogicalSum
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.BlockWord
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

/-- The independent logical int8 matrix product, read through the supplied views. -/
def logicalProduct {α : Type} [Inhabited α] (p : Parameters) (s : State α) (r c k : Nat) : Int :=
  wrap 8 ((s.mem p.input (r*p.inputRowStride+k*p.inputInnerStride)).read .int) *
  wrap 8 ((s.mem p.weight (k*p.weightInnerStride+c*p.weightColumnStride)).read .int)
def logicalDot {α : Type} [Inhabited α] (p : Parameters) (s : State α) (r c : Nat) : Int :=
  wrap 32 (LogicalSum.sum (logicalProduct p s r c) p.inner)
def blockSum {α : Type} [Inhabited α] (p : Parameters) (s : State α) (k : Nat) (i : CIndex p) : Int :=
  (List.finRange p.blockInner).foldl (fun total t =>
    total+aValue p s k (i.1,t,PUnit.unit)*bValue p s k (t,i.2.1,PUnit.unit)) 0
def exactAccumulation {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat → CIndex p → Int
  | 0 => fun _ => 0
  | k+1 => fun i => exactAccumulation p s k i+blockSum p s k i

theorem dot_low32 {α : Type} [Inhabited α] (p : Parameters) (s : State α) (k : Nat) (i : CIndex p) :
    FP.IntegerDot.wordDot [] (aValue p s k) (bValue p s k) i=BitVec.ofInt 32 (blockSum p s k i) := by
  rcases i with ⟨r,c,u⟩; cases u
  exact FP.IntegerDot.fold_is_low32 _ _ _ 0

theorem accumulation_low32 {α : Type} [Inhabited α] (p : Parameters) (s : State α) (n : Nat) (i : CIndex p) :
    BitVec.ofInt 32 (accumulation p s n i)=BitVec.ofInt 32 (exactAccumulation p s n i) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [accumulation, wrap, BitVec.ofInt_toInt, exactAccumulation, BitVec.ofInt_add]
    rw [ih, dot_low32]

theorem accumulation_value {α : Type} [Inhabited α] (p : Parameters) (s : State α) (n : Nat) (i : CIndex p) :
    accumulation p s n i=wrap 32 (exactAccumulation p s n i) := by
  have h := congrArg BitVec.toInt (accumulation_low32 p s n i)
  cases n with
  | zero => rfl
  | succ n => simpa only [accumulation, wrap, BitVec.ofInt_toInt] using h

theorem lane_product {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : CIndex p) (hi : active p s i) (k : Nat) (t : Fin p.blockInner) :
    aValue p s k (i.1,t,PUnit.unit)*bValue p s k (t,i.2.1,PUnit.unit)=
    if k*p.blockInner+t.val<p.inner then
      logicalProduct p s (rowBlock p s*p.blockRows+i.1.val)
        (columnBlock p s*p.blockColumns+i.2.1.val) (k*p.blockInner+t.val) else 0 := by
  rcases hi with ⟨hr,hc⟩
  simp only [aValue,bValue,loaded,aPointer,bPointer,TileShape.indexToList,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets, Nat.zero_add,decide_eq_true_eq]
  by_cases hk : k*p.blockInner+t.val<p.inner
  · simp [hk,hr,hc,logicalProduct,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc]
  · simp [hk,hr,hc,wrap]

theorem blockSum_logical {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : CIndex p) (hi : active p s i) (k : Nat) :
    blockSum p s k i=LogicalSum.sum (fun t => if k*p.blockInner+t<p.inner then
      logicalProduct p s (rowBlock p s*p.blockRows+i.1.val)
        (columnBlock p s*p.blockColumns+i.2.1.val) (k*p.blockInner+t) else 0) p.blockInner := by
  simp only [blockSum,lane_product p s i hi k]
  simpa only [Int.zero_add] using LogicalSum.fin_fold (fun t => if k*p.blockInner+t<p.inner then
    logicalProduct p s (rowBlock p s*p.blockRows+i.1.val)
      (columnBlock p s*p.blockColumns+i.2.1.val) (k*p.blockInner+t) else 0) p.blockInner 0

theorem exact_logical {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : CIndex p) (hi : active p s i) (n : Nat) :
    exactAccumulation p s n i=LogicalSum.sum (fun k => if k<p.inner then
      logicalProduct p s (rowBlock p s*p.blockRows+i.1.val)
        (columnBlock p s*p.blockColumns+i.2.1.val) k else 0) (n*p.blockInner) := by
  induction n with
  | zero => simp [exactAccumulation,LogicalSum.sum]
  | succ n ih => simp only [exactAccumulation,ih,blockSum_logical p s i hi,Nat.succ_mul,LogicalSum.sum_add]

theorem output_logical {α : Type} [Inhabited α] (p : Parameters) (hBK : 0<p.blockInner)
    (s : State α) (i : CIndex p) (hi : active p s i) :
    outputValue p s i=logicalDot p s (rowBlock p s*p.blockRows+i.1.val)
      (columnBlock p s*p.blockColumns+i.2.1.val) := by
  rw [outputValue,accumulation_value,exact_logical p s i hi,LogicalSum.sum_padding]
  have hc := LogicalSum.ceil_cover p.inner p.blockInner hBK
  rw [show min p.inner (steps p*p.blockInner)=p.inner from Nat.min_eq_left hc]
  rfl

#axiomsClean dot_low32
#axiomsClean accumulation_low32
#axiomsClean accumulation_value
#axiomsClean lane_product
#axiomsClean blockSum_logical
#axiomsClean exact_logical
#axiomsClean output_logical
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.BlockWord
