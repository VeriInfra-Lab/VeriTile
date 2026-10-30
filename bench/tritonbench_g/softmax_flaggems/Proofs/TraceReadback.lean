import bench.tritonbench_g.softmax_flaggems.Proofs.FPTraceCore

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.TraceReadback
open VeriTile.Triton FP.Structural FPTrace
noncomputable section
set_option maxHeartbeats 4000000
variable {α ι : Type} [Inhabited α]

theorem fold_reads (xs : List ι) (region : RegionName) (offset : ι → Nat)
    (value : ι → Cell α) (target : RegionName → Nat → Cell α)
    (uniform : ∀ i, i∈xs → value i=target region (offset i)) (s : State α) (r : RegionName) (o : Nat) :
    (xs.foldl (fun st i => st.write region (offset i) (value i)) s).mem r o =
      if (r,o)∈xs.map (fun i => (region,offset i)) then target r o else s.mem r o := by
  induction xs generalizing s with
  | nil => simp
  | cons i xs ih =>
    rw [List.foldl_cons,ih (fun j hj => uniform j (List.mem_cons_of_mem i hj))]
    by_cases hit : r=region ∧ o=offset i
    · rcases hit with ⟨hr,ho⟩
      subst r
      subst o
      by_cases later : (region,offset i)∈xs.map (fun j => (region,offset j))
      · simp [later]
      · simp [later,uniform i (List.mem_cons_self)]
    · have miss : (r,o)≠(region,offset i) :=
        fun h => hit ⟨congrArg Prod.fst h,congrArg Prod.snd h⟩
      have same := State.write_other s region r (offset i) o (value i)
        (by simpa only [not_and_or] using hit)
      simp [miss,same]

def Values (target : RegionName → Nat → Cell α) (t : Tile α) (s : State α) : Prop :=
  ∀ i, t.active i=Bool.true → Cell.mk .real (t.value s i)=target t.region (t.offset i)

theorem tile_reads (target : RegionName → Nat → Cell α) (t : Tile α) (s : State α)
    (hv : Values target t s) (r : RegionName) (o : Nat) :
    (t.execute s).mem r o = if (r,o)∈t.events then target r o else s.mem r o := by
  rw [Tile.exact_stores]
  apply fold_reads
  intro i hi
  apply hv i
  exact (List.mem_filter.mp hi).2

def Sound (Inv : State α → Prop) (target : RegionName → Nat → Cell α) (t : Tile α) : Prop :=
  ∀ s, Inv s → Inv (t.execute s) ∧ Values target t s

theorem execute_reads (Inv : State α → Prop) (target : RegionName → Nat → Cell α)
    (ts : List (Tile α)) (sound : ∀ t, t∈ts → Sound Inv target t) (s : State α) (hs : Inv s) :
    Inv (execute ts s) ∧ ∀ r o, (execute ts s).mem r o =
      if (r,o)∈events ts then target r o else s.mem r o := by
  induction ts generalizing s with
  | nil => exact ⟨hs,by intro r o; simp [execute,events]⟩
  | cons t ts ih =>
    obtain ⟨hm,hv⟩ := sound t (List.mem_cons_self) s hs
    obtain ⟨hf,hr⟩ := ih (fun u hu => sound u (List.mem_cons_of_mem t hu)) (t.execute s) hm
    refine ⟨hf,?_⟩
    intro r o
    change (execute ts (t.execute s)).mem r o = _
    rw [hr,tile_reads target t s hv]
    by_cases ht : (r,o)∈t.events <;> by_cases hs : (r,o)∈events ts <;>
      simp [events,List.mem_append,ht,hs] at *

/-- Store traces preserve every read region separately, even when read tensors
are physically aliased. Fresh output allocation supplies region separation. -/
def Reads (regions : List RegionName) (base : State α) (s : State α) : Prop :=
  ∀ r, r∈regions → ∀ o, s.mem r o=base.mem r o

theorem reads_preserved (regions : List RegionName) (base s : State α) (t : Tile α)
    (outside : t.region∉regions) (same : Reads regions base s) : Reads regions base (t.execute s) := by
  intro r hr o
  rw [Tile.frame t s r o]
  · exact same r hr o
  · intro hit
    obtain ⟨i,hi,he⟩ := (Tile.mem_events t (r,o)).mp hit
    have eq : r=t.region := congrArg Prod.fst he
    exact outside (eq ▸ hr)

#axiomsClean fold_reads
#axiomsClean execute_reads
#axiomsClean reads_preserved
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.TraceReadback
