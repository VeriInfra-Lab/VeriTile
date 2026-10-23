import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.GridFrame
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Grid
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 4000000

theorem mem_programs (grid pid : Program) : pid∈programs grid ↔ Legal grid pid := by
  rcases pid with ⟨x,y⟩
  simp [programs,Legal]

def FullSchedule (grid : Program) (work : List Program) : Prop := work.Perm (programs grid)

theorem FullSchedule.legal (grid : Program) (work : List Program) (h : FullSchedule grid work) :
    ∀ pid∈work,Legal grid pid := by
  intro pid hp
  exact (mem_programs grid pid).mp ((h.mem_iff).mp hp)

theorem FullSchedule.covers (grid : Program) (work : List Program) (h : FullSchedule grid work)
    (pid : Program) (hp : Legal grid pid) : pid∈work :=
  (h.mem_iff).mpr ((mem_programs grid pid).mpr hp)

theorem fold_invariant {α : Type} (step : Program → Memory α → Memory α)
    (inv : Memory α → Prop) (work : List Program) (mem : Memory α)
    (hm : inv mem) (hs : ∀ pid∈work,∀ m,inv m → inv (step pid m)) :
    inv (work.foldl (fun m pid=>step pid m) mem) := by
  induction work generalizing mem with
  | nil=>exact hm
  | cons p ps ih=>
    exact ih (step p mem) (hs p (by simp) mem hm)
      (by intro pid hp m h;exact hs pid (by simp [hp]) m h)

/-- Readback of a particular cell from a complete launch. All source programs
see current memory; the input invariant is preserved through the launch. -/
theorem fold_readback {α : Type} [Inhabited α] (step : Program → Memory α → Memory α)
    (inv : Memory α → Prop) (work : List Program) (mem : Memory α)
    (target : Program) (r : RegionName) (o : Nat) (value : Cell α)
    (ht : target∈work) (hm : inv mem)
    (hs : ∀ pid∈work,∀ m,inv m → inv (step pid m))
    (hw : ∀ m,inv m → step target m r o=value)
    (hf : ∀ pid∈work,pid≠target → ∀ m,step pid m r o=m r o) :
    (work.foldl (fun m pid=>step pid m) mem) r o=value := by
  induction work generalizing mem with
  | nil=>simp at ht
  | cons p ps ih=>
    by_cases hh : target∈ps
    · exact ih (step p mem) hh (hs p (by simp) mem hm)
        (by intro pid hp m h;exact hs pid (by simp [hp]) m h)
        (by intro pid hp hn m;exact hf pid (by simp [hp]) hn m)
    · have he : p=target := by simpa [hh,eq_comm] using ht
      rw [List.foldl_cons,fold_frame]
      · rw [he]
        exact hw mem hm
      · intro pid hp m
        exact hf pid (by simp [hp]) (by intro he;subst pid;exact hh hp) m

#axiomsClean mem_programs
#axiomsClean fold_readback
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Grid
