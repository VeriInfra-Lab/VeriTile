import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Autotune
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Autotune
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Physical (Role)
noncomputable section
set_option maxHeartbeats 6000000

theorem run_congr {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout) (valid : Private.Valid p) (indices : List Nat)
    (unique : indices.Nodup) (bounded : ∀ i∈indices,i<p.total) (a b : M)
    (same : SamePrivate ops p a b) :
    SamePrivate ops p (Machine.run ops model p indices a) (Machine.run ops model p indices b) := by
  intro role i active
  by_cases member : i∈indices
  · have values (output : Output) :
        ops.read (Machine.run ops model p indices a) ((p.allocation (Private.outputRole output)).key i)=
        ops.read (Machine.run ops model p indices b) ((p.allocation (Private.outputRole output)).key i) := by
      rw [Machine.values ops model p valid indices unique bounded a output i member,
        Machine.values ops model p valid indices unique bounded b output i member,
        same .a i active,same .third i active,same .b i active]
    cases role
    · exact values .parameter
    · exact values .momentum
    · have outside : ∀ j∈indices,∀ output,Role.third≠Private.outputRole output ∨ i≠j := by
        intro j member output; left; cases output <;> decide
      rw [Machine.read_frame ops model p valid indices bounded a .third i active outside,
        Machine.read_frame ops model p valid indices bounded b .third i active outside]
      exact same .third i active
  · have outside : ∀ j∈indices,∀ output,role≠Private.outputRole output ∨ i≠j := by
      intro j hj output
      exact Or.inr (fun eq => member (eq.symm ▸ hj))
    rw [Machine.read_frame ops model p valid indices bounded a role i active outside,
      Machine.read_frame ops model p valid indices bounded b role i active outside]
    exact same role i active

theorem launched_congr {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout) (valid : Private.Valid p) (chunk : Index.Chunk)
    (fits : chunk.first+chunk.count≤p.total) (callerBytes : Nat → Nat → Prop)
    (before after : M) (launch : Launched ops model p chunk callerBytes before after) :
    SamePrivate ops p after (Machine.run ops model p chunk.indices before) := by
  rw [launch.selectedSource.physical]
  apply run_congr ops model p valid chunk.indices List.nodup_range' _ launch.restored before
    (trials_private ops model p valid chunk fits callerBytes _ _ _ launch.trials)
  intro i member
  have := List.mem_range'_1.mp member
  omega

theorem launched_bytes {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout) (chunk : Index.Chunk) (callerBytes : Nat → Nat → Prop)
    (before after : M) (launch : Launched ops model p chunk callerBytes before after)
    (owner address : Nat) (kept : callerBytes owner address) (outside : Outside p chunk owner address) :
    ops.observe after owner address=ops.observe before owner address := by
  rw [launch.selectedSource.physical,Machine.byte_frame ops model p chunk.indices launch.restored owner address outside]
  exact trials_bytes ops model p chunk callerBytes _ _ _ launch.trials owner address kept outside

inductive Launches {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout) (callerBytes : Nat → Nat → Prop) : List Index.Chunk → M → M → Prop
  | nil (s) : Launches ops model p callerBytes [] s s
  | cons (chunk rest before middle after)
      (head : Launched ops model p chunk callerBytes before middle)
      (tail : Launches ops model p callerBytes rest middle after) :
      Launches ops model p callerBytes (chunk::rest) before after

theorem chunks_congr {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout) (valid : Private.Valid p) (parts : List Index.Chunk)
    (fits : ∀ chunk∈parts,chunk.first+chunk.count≤p.total) (a b : M) (same : SamePrivate ops p a b) :
    SamePrivate ops p (Machine.chunks ops model p parts a) (Machine.chunks ops model p parts b) := by
  induction parts generalizing a b with
  | nil => exact same
  | cons head rest ih =>
    apply ih (fun chunk member => fits chunk (by simp [member]))
    apply run_congr ops model p valid head.indices List.nodup_range' _ a b same
    intro i member
    have bound := fits head (by simp)
    have := List.mem_range'_1.mp member
    omega

theorem launches_congr {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout) (valid : Private.Valid p) (callerBytes : Nat → Nat → Prop)
    (parts : List Index.Chunk) (fits : ∀ chunk∈parts,chunk.first+chunk.count≤p.total) (before after : M)
    (runs : Launches ops model p callerBytes parts before after) :
    SamePrivate ops p after (Machine.chunks ops model p parts before) := by
  induction runs with
  | nil => intro role i active; rfl
  | cons chunk rest before middle after head tail ih =>
    have tailFits : ∀ part∈rest,part.first+part.count≤p.total :=
      fun part member => fits part (by simp [member])
    have h1 := ih tailFits
    have h2 := chunks_congr ops model p valid rest tailFits middle
      (Machine.run ops model p chunk.indices before)
      (launched_congr ops model p valid chunk (fits chunk (by simp)) callerBytes before middle head)
    intro role i active
    exact (h1 role i active).trans (h2 role i active)

theorem launches_bytes {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout) (callerBytes : Nat → Nat → Prop) (parts : List Index.Chunk)
    (before after : M) (runs : Launches ops model p callerBytes parts before after)
    (owner address : Nat) (kept : callerBytes owner address)
    (outside : ∀ chunk∈parts,Outside p chunk owner address) :
    ops.observe after owner address=ops.observe before owner address := by
  induction runs with
  | nil => rfl
  | cons chunk rest before middle after head tail ih =>
    exact (ih (fun part member => outside part (by simp [member]))).trans
      (launched_bytes ops model p chunk callerBytes before middle head owner address kept (outside chunk (by simp)))

theorem whole_values {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout) (valid : Private.Valid p) (callerBytes : Nat → Nat → Prop)
    (before after : M) (runs : Launches ops model p callerBytes (Index.chunks 0 p.total) before after)
    (output : Output) (i : Nat) (active : i<p.total) :
    ops.read after ((p.allocation (Private.outputRole output)).key i)=
      Machine.scalar model output (ops.read before ((p.allocation .a).key i))
        (ops.read before ((p.allocation .third).key i)) (ops.read before ((p.allocation .b).key i)) := by
  have same := launches_congr ops model p valid callerBytes (Index.chunks 0 p.total)
    (fun chunk member => by simpa using (Index.chunks_bounds 0 p.total chunk member).2.2.2) before after runs
  exact (same (Private.outputRole output) i active).trans (Machine.whole_values ops model p valid before output i active)

#axiomsClean whole_values
#axiomsClean launches_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Autotune
