import bench.tritonbench_g.mixed_sparse_attention.Proofs.OnlineMath
import bench.tritonbench_g.mixed_sparse_attention.Proofs.MixedSparseAttention_Real_RealBackend
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Real
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def logicalScore (c : Config) (s : State R) (μ : RegionName → Nat → ℝ)
    (i : Fin c.BM) (j : Nat) : ℝ :=
  (c.scale*1.44269504)*∑ d : Fin c.BD,
    qAt c s μ (rowStart c s+i.val) d.val*kAt c s μ j d.val

def selected (c : Config) (s : State R) (ph : Bool) (start : Nat) (i : Fin c.BM) (j : Fin c.BN) : Prop :=
  if ph then keyActive c s Bool.true start j
  else keyActive c s Bool.false start j ∧ columnValues c s Bool.false start j≤rowStart c s+i.val

instance (c : Config) (s : State R) (ph : Bool) (start : Nat) (i : Fin c.BM) (j : Fin c.BN) :
    Decidable (selected c s ph start i j) := by unfold selected; infer_instance

def tile (c : Config) (s : State R) (μ : RegionName → Nat → ℝ)
    (ph : Bool) (start : Nat) (i : Fin c.BM) (d : Fin c.BD) : VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.Tile c.BN where
  active:=fun j=>decide (selected c s ph start i j)
  score:=fun j=>logicalScore c s μ i (columnValues c s ph start j)
  value:=fun j=>vAt c s μ (columnValues c s ph start j) d.val

def scores (c : Config) (s : State R) (ph : Bool) (start : Nat) : TileIndex [c.BM,c.BN] → R := fun i=>
  WithBot.realAdd
    (if rowStart c s+i.1.val<sequenceLength c s ∧ selected c s ph start i.1 i.2.1 then some 0 else none)
    (dotValues [] (queryValues algebra c s) (keyValues algebra c s ph start) i)

def maximum (c : Config) (s : State R) (ph : Bool) (start : Nat) (a : Accumulator R c.BM c.BD)
    (i : Fin c.BM) : R := max (a.maximum (i,())) (Finset.univ.sup fun j : Fin c.BN=>scores c s ph start (i,j,()))
def shift (c : Config) (s : State R) (ph : Bool) (start : Nat) (a : Accumulator R c.BM c.BD)
    (i : Fin c.BM) : R :=
  if decide (maximum c s ph start a i=none) then some 0 else maximum c s ph start a i

def probability (c : Config) (s : State R) (ph : Bool) (start : Nat) (a : Accumulator R c.BM c.BD)
    (i : TileIndex [c.BM,c.BN]) : R :=
  exp2 (WithBot.realSub (scores c s ph start i) (shift c s ph start a i.1))
def rescale (c : Config) (s : State R) (ph : Bool) (start : Nat) (a : Accumulator R c.BM c.BD)
    (i : Fin c.BM) : R :=
  exp2 (WithBot.realSub (a.maximum (i,())) (shift c s ph start a i))

theorem selected_key (c : Config) (s : State R) (ph : Bool) (start : Nat) (i : Fin c.BM) (j : Fin c.BN)
    (h : selected c s ph start i j) : keyActive c s ph start j := by
  cases ph
  · exact h.1
  · exact h

theorem query_finite (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads c s μ)
    (i : TileIndex [c.BM,c.BD]) :
    ∃ q,queryValues algebra c s i=some q := by
  by_cases hi : rowStart c s+i.1.val<sequenceLength c s
  · exact ⟨_,query_real c s μ h i hi⟩
  · exact ⟨0,by norm_num [queryValues,hi,algebra]⟩

theorem dot_real (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads c s μ)
    (ph : Bool) (start : Nat) (i : TileIndex [c.BM,c.BN]) (hi : rowStart c s+i.1.val<sequenceLength c s) :
    dotValues [] (queryValues algebra c s) (keyValues algebra c s ph start) i=
      some (if keyActive c s ph start i.2.1 then logicalScore c s μ i.1 (columnValues c s ph start i.2.1) else 0) := by
  change sum c.BD (fun d=>WithBot.realMul (queryValues algebra c s (i.1,d,())) (keyValues algebra c s ph start (d,i.2.1,())))=_
  have hp : (fun d : Fin c.BD=>WithBot.realMul (queryValues algebra c s (i.1,d,())) (keyValues algebra c s ph start (d,i.2.1,())))=
      (fun d=>some ((qAt c s μ (rowStart c s+i.1.val) d.val*(c.scale*1.44269504))*
        (if keyActive c s ph start i.2.1 then kAt c s μ (columnValues c s ph start i.2.1) d.val else 0))) := by
    funext d
    rw [query_real c s μ h _ hi,key_real c s μ h]
    rfl
  rw [hp,sum_some]
  by_cases hj : keyActive c s ph start i.2.1
  · simp only [hj,if_pos,logicalScore,Finset.mul_sum]
    congr 1
    apply Finset.sum_congr rfl
    intro d _
    ring
  · simp [hj]

theorem score_real (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads c s μ)
    (ph : Bool) (start : Nat) (i : TileIndex [c.BM,c.BN]) (hi : rowStart c s+i.1.val<sequenceLength c s) :
    scores c s ph start i=if selected c s ph start i.1 i.2.1 then
      some (logicalScore c s μ i.1 (columnValues c s ph start i.2.1)) else none := by
  unfold scores
  rw [dot_real c s μ h ph start i hi]
  by_cases hj : selected c s ph start i.1 i.2.1
  · simp [hi,hj,selected_key c s ph start i.1 i.2.1 hj]
  · simp [hi,hj]

theorem shift_finite (c : Config) (s : State R) (ph : Bool) (start : Nat) (a : Accumulator R c.BM c.BD)
    (i : Fin c.BM) : shift c s ph start a i=some ((maximum c s ph start a i).unbotD 0) := by
  unfold shift
  cases maximum c s ph start a i <;> simp [WithBot.unbotD] <;> rfl

theorem rescale_real (c : Config) (s : State R) (ph : Bool) (start : Nat) (a : Accumulator R c.BM c.BD)
    (i : Fin c.BM) : rescale c s ph start a i=some (VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.alpha (a.maximum (i,())) ((maximum c s ph start a i).unbotD 0)) := by
  unfold rescale
  rw [shift_finite]
  cases a.maximum (i,()) <;> rfl

theorem maximum_real (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads c s μ)
    (ph : Bool) (start : Nat) (i : Fin c.BM) (d : Fin c.BD) (a : Accumulator R c.BM c.BD)
    (b : VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.Acc) (hi : rowStart c s+i.val<sequenceLength c s) (hm : a.maximum (i,())=b.maximum) :
    maximum c s ph start a i=VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.nextMaximum b (tile c s μ ph start i d) := by
  unfold maximum VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.nextMaximum
  rw [hm]
  congr 1
  congr 1
  funext j
  rw [score_real c s μ h ph start _ hi]
  simp only [tile,decide_eq_true_eq]
  rfl

theorem probability_real (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads c s μ)
    (ph : Bool) (start : Nat) (i : Fin c.BM) (d : Fin c.BD) (a : Accumulator R c.BM c.BD)
    (hi : rowStart c s+i.val<sequenceLength c s) (j : Fin c.BN) :
    probability c s ph start a (i,j,())=some (VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.weight (tile c s μ ph start i d)
      ((maximum c s ph start a i).unbotD 0) j) := by
  unfold probability
  rw [score_real c s μ h ph start _ hi,shift_finite]
  by_cases hj : selected c s ph start i j <;> simp [hj,exp2,VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.weight,tile]

#axiomsClean score_real
#axiomsClean maximum_real
#axiomsClean probability_real
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Real
