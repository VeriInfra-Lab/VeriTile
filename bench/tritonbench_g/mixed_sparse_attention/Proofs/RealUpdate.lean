import bench.tritonbench_g.mixed_sparse_attention.Proofs.RealStep
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Real
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def RowAgrees (c : Config) (i : Fin c.BM) (d : Fin c.BD)
    (a : Accumulator R c.BM c.BD) (b : VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.Acc) : Prop :=
  a.maximum (i,())=b.maximum ∧ a.normalizer (i,())=some b.denominator ∧ a.output (i,d,())=some b.numerator

theorem normalizer_real (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads c s μ)
    (ph : Bool) (start : Nat) (i : Fin c.BM) (d : Fin c.BD) (a : Accumulator R c.BM c.BD)
    (L : ℝ) (hi : rowStart c s+i.val<sequenceLength c s) (hl : a.normalizer (i,())=some L) :
    (update algebra dotModel c s ph start a).normalizer (i,())=some
      (L*VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.alpha (a.maximum (i,())) ((maximum c s ph start a i).unbotD 0)+
        ∑ j,VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.weight (tile c s μ ph start i d) ((maximum c s ph start a i).unbotD 0) j) := by
  change WithBot.realAdd (WithBot.realMul (a.normalizer (i,())) (rescale c s ph start a i))
    (sum c.BN (fun j=>probability c s ph start a (i,j,())))=_
  rw [hl,rescale_real]
  have hp : (fun j : Fin c.BN=>probability c s ph start a (i,j,()))=
      (fun j=>some (VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.weight (tile c s μ ph start i d) ((maximum c s ph start a i).unbotD 0) j)) :=
    funext (probability_real c s μ h ph start i d a hi)
  rw [hp,sum_some]
  rfl

theorem contribution_real (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads c s μ)
    (ph : Bool) (start : Nat) (i : Fin c.BM) (d : Fin c.BD) (a : Accumulator R c.BM c.BD)
    (hi : rowStart c s+i.val<sequenceLength c s) :
    dotValues [] (probability c s ph start a) (valueValues algebra c s ph start) (i,d,())=some
      (∑ j,VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.weight (tile c s μ ph start i d) ((maximum c s ph start a i).unbotD 0) j*
        (tile c s μ ph start i d).value j) := by
  change sum c.BN (fun j=>WithBot.realMul (probability c s ph start a (i,j,()))
    (valueValues algebra c s ph start (j,d,())))=_
  have hp : (fun j : Fin c.BN=>WithBot.realMul (probability c s ph start a (i,j,()))
      (valueValues algebra c s ph start (j,d,())))=
      (fun j=>some (VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.weight (tile c s μ ph start i d) ((maximum c s ph start a i).unbotD 0) j*
        (tile c s μ ph start i d).value j)) := by
    funext j
    rw [probability_real c s μ h ph start i d a hi j,value_real c s μ h]
    by_cases hj : selected c s ph start i j
    · simp [VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.weight,tile,hj,selected_key c s ph start i j hj]
    · simp [VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.weight,tile,hj]
  rw [hp,sum_some]

theorem output_real (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads c s μ)
    (ph : Bool) (start : Nat) (i : Fin c.BM) (d : Fin c.BD) (a : Accumulator R c.BM c.BD)
    (L T : ℝ) (hi : rowStart c s+i.val<sequenceLength c s)
    (hl : a.normalizer (i,())=some L) (ht : a.output (i,d,())=some T) :
    (update algebra dotModel c s ph start a).output (i,d,())=some
      (T*(L*0+VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.alpha (a.maximum (i,())) ((maximum c s ph start a i).unbotD 0))+
        ∑ j,VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.weight (tile c s μ ph start i d) ((maximum c s ph start a i).unbotD 0) j*
          (tile c s μ ph start i d).value j) := by
  change WithBot.realAdd (WithBot.realMul (a.output (i,d,()))
    (WithBot.realAdd (WithBot.realMul (a.normalizer (i,())) (some 0)) (rescale c s ph start a i)))
    (dotValues [] (probability c s ph start a) (valueValues algebra c s ph start) (i,d,()))=_
  rw [ht,hl,rescale_real,contribution_real c s μ h ph start i d a hi]
  rfl

/-- One actual typed source recurrence step agrees with the independently
proved online Real update, including a wholly masked initial tile. -/
theorem update_agrees (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads c s μ)
    (ph : Bool) (start : Nat) (i : Fin c.BM) (d : Fin c.BD) (a : Accumulator R c.BM c.BD)
    (b : VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.Acc) (hi : rowStart c s+i.val<sequenceLength c s) (hab : RowAgrees c i d a b) :
    RowAgrees c i d (update algebra dotModel c s ph start a) (VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.step b (tile c s μ ph start i d)) := by
  obtain ⟨hm,hl,ht⟩:=hab
  refine ⟨?_,?_,?_⟩
  · change maximum c s ph start a i=VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Online.nextMaximum b (tile c s μ ph start i d)
    exact maximum_real c s μ h ph start i d a b hi hm
  · rw [normalizer_real c s μ h ph start i d a b.denominator hi hl,
      maximum_real c s μ h ph start i d a b hi hm,hm]
    rfl
  · rw [output_real c s μ h ph start i d a b.denominator b.numerator hi hl ht,
      maximum_real c s μ h ph start i d a b hi hm,hm]
    rfl

theorem initial_agrees (c : Config) (i : Fin c.BM) (d : Fin c.BD) :
    RowAgrees c i d (initialAccumulator algebra c) ⟨⊥,0,0⟩ := by
  exact ⟨rfl,rfl,rfl⟩

#axiomsClean normalizer_real
#axiomsClean output_real
#axiomsClean update_agrees
#axiomsClean initial_agrees
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Real
