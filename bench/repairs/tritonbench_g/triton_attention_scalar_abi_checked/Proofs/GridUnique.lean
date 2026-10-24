import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.GridCoverage
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Grid
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

theorem programs_nodup (grid : Program) : (programs grid).Nodup := by
  change ((List.range grid.1).product (List.range grid.2)).Nodup
  exact List.Nodup.product List.nodup_range List.nodup_range

theorem schedule_nodup (grid : Program) (work : List Program) (h : FullSchedule grid work) : work.Nodup :=
  h.nodup_iff.mpr (programs_nodup grid)

/-- The unique target CTA may consume/overwrite part of its input invariant,
as backward does with initially zero DQ. Other CTAs preserve that invariant. -/
theorem fold_readback_unique {α : Type} [Inhabited α] (step : Program → Memory α → Memory α)
    (inv : Memory α → Prop) (work : List Program) (mem : Memory α)
    (target : Program) (r : RegionName) (o : Nat) (value : Cell α)
    (hn : work.Nodup) (ht : target∈work) (hm : inv mem)
    (hs : ∀ pid∈work,pid≠target → ∀ m,inv m → inv (step pid m))
    (hw : ∀ m,inv m → step target m r o=value)
    (hf : ∀ pid∈work,pid≠target → ∀ m,step pid m r o=m r o) :
    (work.foldl (fun m pid=>step pid m) mem) r o=value := by
  induction work generalizing mem with
  | nil=>simp at ht
  | cons p ps ih=>
    have hnodup:=List.nodup_cons.mp hn
    by_cases he : p=target
    · rw [List.foldl_cons,fold_frame]
      · rw [he]
        exact hw mem hm
      · intro pid hp m
        exact hf pid (by simp [hp]) (by intro ht;subst pid;exact hnodup.1 (he ▸ hp)) m
    · have hh : target∈ps := by simpa [Ne.symm he] using ht
      exact ih (step p mem) hnodup.2 hh (hs p (by simp) he mem hm)
        (by intro pid hp hn m h;exact hs pid (by simp [hp]) hn m h)
        (by intro pid hp hn m;exact hf pid (by simp [hp]) hn m)

#axiomsClean schedule_nodup
#axiomsClean fold_readback_unique
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Grid
