import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.NativeLocality

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.ArrayLocality
open VeriTile Triton FP.Structural Reference
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host Ownership
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

def IsReal (cell : Cell α) : Prop := ∃ value,cell=.mk .real value

theorem fold_real_preserved {I : Type} (is : List I) (rg : RegionName) (offset : I → Nat)
    (values : I → α) (s : State α) (r : RegionName) (off : Nat) (before : IsReal (s.mem r off)) :
    IsReal ((is.foldl (fun t i => t.write rg (offset i) (.mk .real (values i))) s).mem r off) := by
  induction is generalizing s with
  | nil => exact before
  | cons i rest ih =>
    apply ih
    by_cases same : r=rg ∧ off=offset i
    · obtain ⟨rfl,rfl⟩ := same
      exact ⟨values i,State.write_same _ _ _ _⟩
    · rw [State.write_other _ _ _ _ _ _ (by tauto)]
      exact before

theorem fold_real_hit {I : Type} (is : List I) (rg : RegionName) (offset : I → Nat)
    (values : I → α) (s : State α) (i : I) (member : i∈is) :
    IsReal ((is.foldl (fun t j => t.write rg (offset j) (.mk .real (values j))) s).mem rg (offset i)) := by
  induction is generalizing s with
  | nil => simp at member
  | cons j rest ih =>
    rcases List.mem_cons.mp member with rfl|tail
    · exact fold_real_preserved rest rg offset values _ rg (offset i)
        ⟨values i,State.write_same _ _ _ _⟩
    · exact ih _ tail

def laneAddress (c : Call) (a : Inputs α) : Lane c.BM c.D → RegionName × Nat
  | .center r => ("M",carryAddress c a r.val)
  | .denominator r => ("L",carryAddress c a r.val)
  | .output r d => ((optr c a).region,(optr c a).address (TileShape.indexToList [c.BM,c.D] (r,d,PUnit.unit)))

theorem prepared_address (c : Call) (s : State α) (idx : GridIndex c.grid) (lane : Lane c.BM c.D) :
    laneAddress c (inputs (Composition.prepare c.grid idx s)) lane=Ownership.address c idx lane := by
  cases lane <;>
    simp [laneAddress,inputs,Composition.prepare,Ownership.address,carryAddress,row,optr,batch,head,
      Call.carryAddress,Call.outAddress,Call.row,Geometry.Strides.at,Call.tile,Call.hz,Nat.add_assoc,
      GridIndex.toPids,Call.grid,Grid.rank]

theorem scatter_real_preserved (c : Call) (o : Ops α) (a : Inputs α) (old : Carry c α)
    (s : State α) (rg : RegionName) (off : Nat) (before : IsReal (s.mem rg off)) :
    IsReal ((scatter c o a old s).mem rg off) := by
  unfold scatter writeOutput
  apply fold_real_preserved
  unfold writeRows
  apply fold_real_preserved
  split_ifs
  · exact fold_real_preserved _ _ _ _ _ _ _ before
  · exact before

theorem scatter_real_hit (c : Call) (o : Ops α) (a : Inputs α) (old : Carry c α)
    (s : State α) (lane : Lane c.BM c.D) (active : Ownership.active c lane) :
    IsReal ((scatter c o a old s).mem (laneAddress c a lane).1 (laneAddress c a lane).2) := by
  cases lane with
  | center r =>
    unfold scatter writeOutput
    apply fold_real_preserved
    exact fold_real_hit _ _ _ _ _ (r,PUnit.unit) (TileShape.mem_allIndices _ _)
  | denominator r =>
    change c.finish=0 at active
    unfold scatter writeOutput
    apply fold_real_preserved
    unfold writeRows
    apply fold_real_preserved
    rw [if_pos active]
    exact fold_real_hit _ _ _ _ _ (r,PUnit.unit) (TileShape.mem_allIndices _ _)
  | output r d =>
    exact fold_real_hit _ _ _ _ _ (r,d,PUnit.unit) (TileShape.mem_allIndices _ _)

theorem scatter_frame (c : Call) (o : Ops α) (a : Inputs α) (old : Carry c α)
    (s : State α) (rg : RegionName) (off : Nat)
    (outside : ∀ lane,Ownership.active c lane → laneAddress c a lane≠(rg,off)) :
    (scatter c o a old s).mem rg off=s.mem rg off := by
  have hO : rg≠(optr c a).region ∨ ∀ i : TileIndex [c.BM,c.D],
      off≠(optr c a).address (TileShape.indexToList [c.BM,c.D] i) := by
    by_cases h : rg=(optr c a).region
    · right
      intro i eq
      apply outside (.output i.1 i.2.1) trivial
      exact Prod.ext h.symm (by simpa only [Reference.pair_index] using eq.symm)
    · exact Or.inl h
  have hM : rg≠"M" ∨ ∀ i : TileIndex [c.BM],off≠carryAddress c a i.1.val := by
    by_cases h : rg="M"
    · right
      intro i eq
      exact outside (.center i.1) trivial (Prod.ext h.symm eq.symm)
    · exact Or.inl h
  unfold scatter writeOutput
  rw [State.scatter_frame _ _ _ _ _ _ (hO.imp_right (fun h i _ => h i))]
  unfold writeRows
  rw [State.scatter_frame _ _ _ _ _ _ (hM.imp_right (fun h i _ => h i))]
  split_ifs with hf
  · apply State.scatter_frame
    by_cases h : rg="L"
    · right
      intro i _ eq
      exact outside (.denominator i.1) hf (Prod.ext h.symm eq.symm)
    · exact Or.inl h
  · rfl

theorem schedule_real_preserved (c : Call) (o : Ops α) (jobs : List (GridIndex c.grid))
    (s : State α) (rg : RegionName) (off : Nat) (before : IsReal (s.mem rg off)) :
    IsReal ((referenceSchedule c o jobs s).mem rg off) := by
  induction jobs generalizing s with
  | nil => exact before
  | cons idx rest ih => exact ih _ (scatter_real_preserved c o _ _ _ rg off before)

theorem schedule_real_hit (c : Call) (o : Ops α) (jobs : List (GridIndex c.grid))
    (s : State α) (idx : GridIndex c.grid) (member : idx∈jobs)
    (lane : Lane c.BM c.D) (active : Ownership.active c lane) :
    IsReal ((referenceSchedule c o jobs s).mem (Ownership.address c idx lane).1
      (Ownership.address c idx lane).2) := by
  induction jobs generalizing s with
  | nil => simp at member
  | cons jdx rest ih =>
    simp only [referenceSchedule]
    rcases List.mem_cons.mp member with rfl|tail
    · apply schedule_real_preserved
      have hit := scatter_real_hit c o (inputs (Composition.prepare c.grid idx s))
        (result c o (inputs (Composition.prepare c.grid idx s)))
        (Composition.prepare c.grid idx s) lane active
      simpa only [prepared_address] using hit
    · exact ih _ tail

theorem schedule_frame (c : Call) (o : Ops α) (jobs : List (GridIndex c.grid))
    (s : State α) (rg : RegionName) (off : Nat)
    (outside : ∀ idx∈jobs,∀ lane,Ownership.active c lane → Ownership.address c idx lane≠(rg,off)) :
    (referenceSchedule c o jobs s).mem rg off=s.mem rg off := by
  induction jobs generalizing s with
  | nil => rfl
  | cons idx rest ih =>
    simp only [referenceSchedule]
    rw [ih _ (fun jdx member => outside jdx (List.mem_cons_of_mem idx member))]
    apply scatter_frame
    intro lane active
    rw [prepared_address]
    exact outside idx List.mem_cons_self lane active

#axiomsClean schedule_real_hit
#axiomsClean schedule_frame
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.ArrayLocality
