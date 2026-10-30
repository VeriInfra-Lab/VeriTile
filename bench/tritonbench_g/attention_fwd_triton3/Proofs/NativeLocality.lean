import bench.tritonbench_g.attention_fwd_triton3.Proofs.Grid

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.ArrayLocality
open VeriTile Triton FP.Structural Reference
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

def buffers : List RegionName := ["Q","K","V","M","L","O"]

def SameInputs (a b : Inputs α) : Prop := a.tile=b.tile ∧ a.hz=b.hz ∧
  ∀ rg,rg∈buffers → ∀ off,a.read rg off=b.read rg off

theorem initial_congr (c : Call) (o : Ops α) (a b : Inputs α) (same : SameInputs a b) :
    initial c o a=initial c o b := by
  rcases a with ⟨ra,ta,ha⟩
  rcases b with ⟨rb,tb,hb⟩
  rcases same with ⟨rfl,rfl,read⟩
  have hm := read "M" (by simp [buffers])
  have hl := read "L" (by simp [buffers])
  have ho := read "O" (by simp [buffers])
  dsimp only at hm hl ho
  simp only [initial,carryAddress,row,kptr,vptr,optr,batch,head,kvhead,lower,hm,hl,ho]

theorem query_congr (c : Call) (o : Ops α) (a b : Inputs α) (same : SameInputs a b) :
    query c o a=query c o b := by
  rcases a with ⟨ra,ta,ha⟩
  rcases b with ⟨rb,tb,hb⟩
  rcases same with ⟨rfl,rfl,read⟩
  have hq := read "Q" (by simp [buffers])
  dsimp only at hq
  funext i
  simp only [query,Reference.loadBlock,qptr,batch,head,hq]

def PointerRegions (c : Call) (old : Carry c α) : Prop :=
  old.keyPointer.region="K" ∧ old.valuePointer.region="V"

theorem initial_regions (c : Call) (o : Ops α) (a : Inputs α) :
    PointerRegions c (initial c o a) := by
  by_cases h : c.window≠0 ∧ c.complement=0 <;> simp [PointerRegions,initial,kptr,vptr,h]

theorem advance_regions (c : Call) (o : Ops α) (a : Inputs α) (cur : Int)
    (old : Carry c α) (regions : PointerRegions c old) :
    PointerRegions c (advance c o a cur old) := by
  simpa only [PointerRegions,advance,BlockPtr.advance] using regions

theorem advance_congr (c : Call) (o : Ops α) (a b : Inputs α) (same : SameInputs a b)
    (cur : Int) (old : Carry c α) (regions : PointerRegions c old) :
    advance c o a cur old=advance c o b cur old := by
  have hq := query_congr c o a b same
  have hk : loadBlock o a old.keyPointer [c.D,c.BN] (if c.params.evenN≠0 then [] else [0,1])=
      loadBlock o b old.keyPointer [c.D,c.BN] (if c.params.evenN≠0 then [] else [0,1]) := by
    funext i
    simp only [Reference.loadBlock,regions.1,same.2.2 "K" (by simp [buffers])]
  have hv : loadBlock o a old.valuePointer [c.BN,c.D] (if c.params.evenN≠0 then [] else [0,1])=
      loadBlock o b old.valuePointer [c.BN,c.D] (if c.params.evenN≠0 then [] else [0,1]) := by
    funext i
    simp only [Reference.loadBlock,regions.2,same.2.2 "V" (by simp [buffers])]
  simp only [advance,hk,hv,hq,windowMask,same.1]

theorem iterate_congr (c : Call) (o : Ops α) (a b : Inputs α) (same : SameInputs a b)
    (cur stop : Int) (old : Carry c α) (regions : PointerRegions c old) :
    iterate c o a cur stop old=iterate c o b cur stop old := by
  conv_lhs => rw [Reference.iterate]
  conv_rhs => rw [Reference.iterate]
  split_ifs with hz hc
  · rfl
  · rw [advance_congr c o a b same cur old regions]
    exact iterate_congr c o a b same (cur+c.BN) stop _ (advance_regions c o b cur old regions)
  · rfl
termination_by (stop-cur).toNat
decreasing_by omega

theorem result_congr (c : Call) (o : Ops α) (a b : Inputs α) (same : SameInputs a b) :
    result c o a=result c o b := by
  have hlo : lower c a=lower c b := by simp only [lower,same.1]
  have hhi : upper c a=upper c b := by simp only [upper,same.1]
  unfold result
  rw [hlo,hhi,initial_congr c o a b same]
  exact iterate_congr c o a b same _ _ _ (initial_regions c o b)

def SameReads (s t : State α) : Prop :=
  ∀ rg,rg∈buffers → ∀ off,(s.mem rg off).read .real=(t.mem rg off).read .real

theorem prepared_inputs (c : Call) (idx : GridIndex c.grid) (s t : State α)
    (same : SameReads s t) :
    SameInputs (inputs (Composition.prepare c.grid idx s))
      (inputs (Composition.prepare c.grid idx t)) := ⟨rfl,rfl,same⟩

theorem write_same_reads (s t : State α) (same : SameReads s t)
    (rg : RegionName) (off : Nat) (v : Cell α) :
    SameReads (s.write rg off v) (t.write rg off v) := by
  intro r member i
  by_cases hr : r=rg
  · subst r
    by_cases hi : i=off
    · simp [State.write,hi]
    · simpa [State.write,hi] using same rg member i
  · simpa [State.write,hr] using same r member i

theorem fold_same_reads {I : Type} (is : List I) (rg : RegionName) (off : I → Nat)
    (v : I → Cell α) (s t : State α) (same : SameReads s t) :
    SameReads (is.foldl (fun u i => u.write rg (off i) (v i)) s)
      (is.foldl (fun u i => u.write rg (off i) (v i)) t) := by
  induction is generalizing s t with
  | nil => exact same
  | cons i rest ih => exact ih _ _ (write_same_reads s t same _ _ _)

theorem scatter_same_reads (c : Call) (o : Ops α) (a b : Inputs α)
    (sameInputs : SameInputs a b) (old : Carry c α) (s t : State α)
    (same : SameReads s t) : SameReads (scatter c o a old s) (scatter c o b old t) := by
  have hcarry : carryAddress c a=carryAddress c b := by
    funext i
    simp only [carryAddress,row,sameInputs.1,sameInputs.2.1]
  have hop : optr c a=optr c b := by
    simp only [optr,batch,head,sameInputs.1,sameInputs.2.1]
  unfold scatter writeOutput
  rw [hop]
  apply fold_same_reads
  unfold writeRows
  rw [hcarry]
  apply fold_same_reads
  split_ifs
  · exact fold_same_reads _ _ _ _ _ _ same
  · exact same

theorem schedule_same_reads (c : Call) (o : Ops α) (jobs : List (GridIndex c.grid))
    (s t : State α) (same : SameReads s t) :
    SameReads (referenceSchedule c o jobs s) (referenceSchedule c o jobs t) := by
  induction jobs generalizing s t with
  | nil => exact same
  | cons idx rest ih =>
    simp only [referenceSchedule]
    have h := prepared_inputs c idx s t same
    rw [result_congr c o _ _ h]
    exact ih _ _ (scatter_same_reads c o _ _ h _ _ _ same)

#axiomsClean schedule_same_reads
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.ArrayLocality
