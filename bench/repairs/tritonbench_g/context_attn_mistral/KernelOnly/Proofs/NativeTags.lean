import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Proofs.NativeBindings

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeBindings
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnMistral
noncomputable section
open scoped Classical
variable {α : Type} [Inhabited α]

def realCell (cell : Cell α) : Prop := cell=Cell.mk .real (cell.read .real)

theorem fold_real {I : Type} (region : RegionName) (offset : I → Nat)
    (value : I → α) (on : I → Prop) [DecidablePred on] (lanes : List I)
    (s : State α) (off : Nat)
    (admitted : realCell (s.mem region off) ∨ ∃ i∈lanes,on i ∧ offset i=off) :
    realCell ((lanes.foldl (fun t i=>if on i then t.write region (offset i) (.mk .real (value i)) else t) s).mem region off) := by
  induction lanes generalizing s with
  | nil=>simpa using admitted
  | cons i rest ih=>
    simp only [List.foldl_cons]
    by_cases active:on i
    · simp only [if_pos active]
      apply ih
      by_cases hit:offset i=off
      · left
        subst off
        simp [State.write_same,realCell]
      · rcases admitted with old | ⟨j,member,hj,ej⟩
        · left
          rw [State.write_other _ _ _ _ _ _ (Or.inr (Ne.symm hit))]
          exact old
        · right
          refine ⟨j,?_,hj,ej⟩
          rcases List.mem_cons.mp member with same | tail
          · subst j; exact (hit ej).elim
          · exact tail
    · simp only [if_neg active]
      apply ih
      rcases admitted with old | ⟨j,member,hj,ej⟩
      · exact Or.inl old
      · right
        refine ⟨j,?_,hj,ej⟩
        rcases List.mem_cons.mp member with same | tail
        · subst j; exact (active hj).elim
        · exact tail

theorem tile_real (c : Host.Call) (o : FPBackend.Ops α) (idx : GridIndex c.grid)
    (s : State α) (metadata : KernelFootprint.Metadata c s) (off : Nat)
    (admitted : realCell (s.mem c.regions.out off) ∨ ∃ i,c.active idx i ∧ c.offset idx i=off) :
    realCell ((KernelReference.tile c o idx s).mem c.regions.out off) := by
  apply fold_real
  rcases admitted with old | ⟨i,on,address⟩
  · exact Or.inl old
  · exact Or.inr ⟨i,TileShape.mem_allIndices _ i,
      (KernelFootprint.active_matches c s metadata idx i).mpr on,
      (KernelFootprint.offset_matches c s metadata idx i).trans address⟩

theorem grid_real (c : Host.Call) (o : FPBackend.Ops α) (jobs : List (GridIndex c.grid))
    (s : State α) (metadata : KernelFootprint.Metadata c s)
    (separate : c.regions.out≠c.regions.start ∧ c.regions.out≠c.regions.sequence)
    (off : Nat) (admitted : realCell (s.mem c.regions.out off) ∨ KernelFootprint.writes c jobs c.regions.out off) :
    realCell ((KernelReference.grid c o jobs s).mem c.regions.out off) := by
  induction jobs generalizing s with
  | nil=>
    rcases admitted with old | ⟨idx,member,_⟩
    · exact old
    · simp at member
  | cons idx rest ih=>
    apply ih _ (tile_metadata c o idx s metadata separate)
    rcases admitted with old | ⟨job,member,i,on,region,address⟩
    · exact Or.inl (tile_real c o idx s metadata off (Or.inl old))
    · rcases List.mem_cons.mp member with same | tail
      · subst job
        exact Or.inl (tile_real c o idx s metadata off (Or.inr ⟨i,on,address.symm⟩))
      · exact Or.inr ⟨job,tail,i,on,region,address⟩

#axiomsClean grid_real
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeBindings
