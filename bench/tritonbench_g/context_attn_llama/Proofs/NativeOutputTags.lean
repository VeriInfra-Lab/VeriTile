import bench.tritonbench_g.context_attn_llama.Proofs.NativeBindings

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeBindings
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
variable {α I : Type} [Inhabited α]

theorem fold_preserves_real (region : RegionName) (offset : I → Nat) (value : I → α)
    (active : I → Prop) (lanes : List I) (s : State α) (address : Nat)
    (initial : ∃ x,s.mem region address=.mk .real x) :
    ∃ x,(lanes.foldl (fun t i=>if active i then t.write region (offset i) (.mk .real (value i)) else t) s).mem
      region address=.mk .real x := by
  induction lanes generalizing s with
  | nil=>exact initial
  | cons i rest ih=>
    simp only [List.foldl_cons]
    apply ih
    by_cases on:active i
    · rw [if_pos on]
      by_cases same:address=offset i
      · exact ⟨value i,by simp [State.write,same]⟩
      · simpa [State.write,same] using initial
    · simpa only [if_neg on] using initial

theorem fold_hit_real (region : RegionName) (offset : I → Nat) (value : I → α)
    (active : I → Prop) (lanes : List I) (s : State α) (address : Nat)
    (hit : ∃ i∈lanes,active i ∧ offset i=address) :
    ∃ x,(lanes.foldl (fun t i=>if active i then t.write region (offset i) (.mk .real (value i)) else t) s).mem
      region address=.mk .real x := by
  induction lanes generalizing s with
  | nil=>simpa using hit
  | cons i rest ih=>
    obtain ⟨j,member,on,addressEq⟩:=hit
    rcases List.mem_cons.mp member with same|member
    · subst j
      simp only [List.foldl_cons,if_pos on]
      exact fold_preserves_real region offset value active rest _ address
        ⟨value i,by simp [State.write,addressEq]⟩
    · simp only [List.foldl_cons]
      exact ih _ ⟨j,member,on,addressEq⟩

theorem tile_real (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (idx : GridIndex c.grid) (s : State α) (metadata : FPSuccess.Metadata c s)
    (i : TileIndex [c.layout.BM,c.layout.D]) (active : c.active idx i) :
    ∃ x,(FPReference.tile c M dots idx s).mem c.regions.out (c.offset idx i)=.mk .real x := by
  apply fold_hit_real
  refine ⟨i,TileShape.mem_allIndices _ i,(FPSuccess.active_matches c s metadata idx i).mpr active,?_⟩
  exact FPSuccess.offset_matches c s metadata idx i

theorem grid_real (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (order : List (GridIndex c.grid)) (s : State α) (metadata : FPSuccess.Metadata c s)
    (separate : c.regions.start≠c.regions.out ∧ c.regions.sequence≠c.regions.out ∧ c.regions.prompt≠c.regions.out)
    (off : Nat) (written : FPSuccess.writes c order c.regions.out off) :
    ∃ x,(FPReference.grid c M dots order s).mem c.regions.out off=.mk .real x := by
  induction order generalizing s with
  | nil=>simpa [FPSuccess.writes] using written
  | cons idx rest ih=>
    have next:=tile_metadata c M dots idx s metadata separate
    by_cases later:FPSuccess.writes c rest c.regions.out off
    · exact ih _ next later
    · rw [FPReference.grid,RawKernel.reference_frame c M dots rest _ next separate _ _ later]
      obtain ⟨job,member,i,on,_,address⟩:=written
      rcases List.mem_cons.mp member with same|member
      · subst job
        rw [address]
        exact tile_real c M dots idx s metadata i on
      · exact False.elim (later ⟨job,member,i,on,rfl,address⟩)

#axiomsClean fold_hit_real
#axiomsClean grid_real
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.NativeBindings
