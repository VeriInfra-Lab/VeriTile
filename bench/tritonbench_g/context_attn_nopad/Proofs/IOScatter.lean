import bench.tritonbench_g.context_attn_nopad.Proofs.IOLogicalOutput

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOLogical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
open Classical
set_option maxHeartbeats 3000000
variable {α I : Type} [Inhabited α]

def cells (region : RegionName) (offset : I → Nat) (on : I → Prop) (values : I → α)
    (lanes : List I) (s : State α) : State α :=
  lanes.foldl (fun t i => if on i then t.write region (offset i) (.mk .real (values i)) else t) s

def values (region : RegionName) (offset : I → Nat) (on : I → Prop) (v : I → α)
    (lanes : List I) (m : Memory α) : Memory α :=
  lanes.foldl (fun t i => if on i then
    fun r off => if r=region ∧ off=offset i then v i else t r off else t) m

def Hit (region : RegionName) (offset : I → Nat) (on : I → Prop) (lanes : List I)
    (r : RegionName) (off : Nat) : Prop := ∃ i∈lanes,on i ∧ r=region ∧ off=offset i

theorem scatter_reads (region : RegionName) (offset : I → Nat) (on : I → Prop) (v : I → α)
    (lanes : List I) (s : State α) (m : Memory α) (r : RegionName) (off : Nat)
    (agree : (s.mem r off).read .real=m r off) :
    ((cells region offset on v lanes s).mem r off).read .real=values region offset on v lanes m r off := by
  induction lanes generalizing s m with
  | nil => exact agree
  | cons i rest ih =>
    simp only [cells,values,List.foldl_cons]
    by_cases hi : on i
    · simp only [if_pos hi]
      apply ih
      simp only [State.write]
      by_cases hit : r=region ∧ off=offset i
      · simp only [if_pos hit,Cell.read_same]
      · simp only [if_neg hit]
        exact agree
    · simp only [if_neg hi]
      exact ih s m agree

theorem scatter_cells (region : RegionName) (offset : I → Nat) (on : I → Prop) (v : I → α)
    (lanes : List I) (s : State α) (m : Memory α) (r : RegionName) (off : Nat)
    (agree : s.mem r off=Cell.mk .real (m r off) ∨ Hit region offset on lanes r off) :
    (cells region offset on v lanes s).mem r off=Cell.mk .real (values region offset on v lanes m r off) := by
  induction lanes generalizing s m with
  | nil => exact agree.elim id (by simp [Hit])
  | cons i rest ih =>
    simp only [cells,values,List.foldl_cons]
    by_cases hi : on i
    · simp only [if_pos hi]
      apply ih
      by_cases hit : r=region ∧ off=offset i
      · left
        simp only [State.write,if_pos hit]
      · rcases agree with eq|found
        · left
          simp only [State.write,if_neg hit]
          exact eq
        · right
          rcases found with ⟨j,hj,hon,hr,ho⟩
          rcases List.mem_cons.mp hj with rfl|hj
          · exact False.elim (hit ⟨hr,ho⟩)
          · exact ⟨j,hj,hon,hr,ho⟩
    · simp only [if_neg hi]
      apply ih
      rcases agree with eq|found
      · exact Or.inl eq
      · right
        rcases found with ⟨j,hj,hon,hr,ho⟩
        rcases List.mem_cons.mp hj with rfl|hj
        · exact False.elim (hi hon)
        · exact ⟨j,hj,hon,hr,ho⟩

theorem tile_memory (c : Call) (r : Regions) (o : FPBackend.Ops α) (idx : GridIndex c.grid)
    (s : State α) (metadata : FPHostSource.Metadata c r s) :
    (FPReference.tile c r o idx s).mem=
      (cells r.out (FPReference.laneOffset c idx) (FPReference.laneActive c idx)
        (output c r o idx (fun region off => (s.mem region off).read .real))
        (TileShape.allIndices [c.layout.BM,c.layout.D]) s).mem := by
  unfold FPReference.tile FPReference.scatter
  simp_rw [FPReference.active_matches c r s metadata idx,FPReference.offset_matches c r s metadata idx]
  rw [source_arguments c r idx s metadata]
  exact masked_fold_mem_congr _ _ _ _ _ _ _ rfl

#axiomsClean scatter_cells
#axiomsClean tile_memory
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOLogical
