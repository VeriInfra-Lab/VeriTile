import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Machine
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Autotune
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Physical (Role)
open VeriTile.Bench.TritonBenchG.AdamUpdateTriton.HostAutotune (Config)
noncomputable section
set_option maxHeartbeats 6000000

def Selected (chunk : Index.Chunk) (role : Role) (i : Nat) : Prop :=
  (role=.a ∨ role=.b) ∧ i∈chunk.indices

def SamePrivate {M α β : Type} (ops : Ops M α β) (p : Private.Layout) (a b : M) : Prop :=
  ∀ role i,i<p.total → ops.read a ((p.allocation role).key i)=ops.read b ((p.allocation role).key i)

def Outside (p : Private.Layout) (chunk : Index.Chunk) (owner address : Nat) : Prop :=
  ∀ i∈chunk.indices,∀ output,¬((p.allocation (Private.outputRole output)).key i).covers owner address

/-- The default Triton pre-hook clones both actual pointer slices; the
post-hook copy_ restores those slices after every trial. This local copy
contract refers only to saved entry words and the post-trial state. callerBytes
denotes live caller bytes, excluding fresh autotuner clone allocations. -/
structure Restore {M α β : Type} (ops : Ops M α β) (p : Private.Layout) (chunk : Index.Chunk)
    (callerBytes : Nat → Nat → Prop) (saved trial restored : M) : Prop where
  snapshot : ∀ role i,i<p.total → Selected chunk role i →
    ops.read restored ((p.allocation role).key i)=ops.read saved ((p.allocation role).key i)
  readFrame : ∀ role i,i<p.total → ¬Selected chunk role i →
    ops.read restored ((p.allocation role).key i)=ops.read trial ((p.allocation role).key i)
  byteFrame : ∀ owner address,callerBytes owner address → Outside p chunk owner address →
    ops.observe restored owner address=ops.observe trial owner address

theorem restored_private {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β)
    (model : Machine.Model α τ) (p : Private.Layout) (valid : Private.Valid p)
    (chunk : Index.Chunk) (fits : chunk.first+chunk.count≤p.total)
    (callerBytes : Nat → Nat → Prop) (saved trial restored : M)
    (ran : trial=Machine.run ops model p chunk.indices saved)
    (copy : Restore ops p chunk callerBytes saved trial restored) : SamePrivate ops p restored saved := by
  intro role i active
  by_cases selected : Selected chunk role i
  · exact copy.snapshot role i active selected
  · rw [copy.readFrame role i active selected,ran]
    have bounded : ∀ j∈chunk.indices,j<p.total := by
      intro j member
      have := List.mem_range'_1.mp member
      omega
    apply Machine.read_frame ops model p valid chunk.indices bounded saved role i active
    intro j member output
    by_cases same : role=Private.outputRole output
    · right
      intro eq
      apply selected
      refine ⟨?_,eq.symm ▸ member⟩
      cases output <;> simp [Private.outputRole] at same <;> simp [same]
    · exact Or.inl same

theorem restored_bytes {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β)
    (model : Machine.Model α τ) (p : Private.Layout)
    (chunk : Index.Chunk) (callerBytes : Nat → Nat → Prop) (saved trial restored : M)
    (ran : trial=Machine.run ops model p chunk.indices saved)
    (copy : Restore ops p chunk callerBytes saved trial restored)
    (owner address : Nat) (kept : callerBytes owner address) (outside : Outside p chunk owner address) :
    ops.observe restored owner address=ops.observe saved owner address := by
  rw [copy.byteFrame owner address kept outside,ran]
  exact Machine.byte_frame ops model p chunk.indices saved owner address outside

/-- Ordinary clone/setup/benchmark-administration effects may write fresh
storage. They preserve all live private input words and caller bytes. This
local copy/frame contract contains no kernel result or numerical law. -/
structure Prepare {M α β : Type} (ops : Ops M α β) (p : Private.Layout)
    (callerBytes : Nat → Nat → Prop) (before after : M) : Prop where
  privateReads : SamePrivate ops p after before
  callerBytes : ∀ owner address,callerBytes owner address →
    ops.observe after owner address=ops.observe before owner address

/-- Empty history is a cache hit; repeats and any finite benchmark/warmup
history are allowed. Each successful trial runs actual source before restore. -/
inductive Trials {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout) (chunk : Index.Chunk)
    (callerBytes : Nat → Nat → Prop) : List Config → M → M → Prop
  | nil (s) : Trials ops model p chunk callerBytes [] s s
  | cons (config rest schedule before prepared trial restored after)
      (prepare : Prepare ops p callerBytes before prepared)
      (source : Machine.SourceStep ops model p chunk config schedule prepared trial)
      (restore : Restore ops p chunk callerBytes prepared trial restored)
      (tail : Trials ops model p chunk callerBytes rest restored after) :
      Trials ops model p chunk callerBytes (config::rest) before after
  | administrative (history before prepared after)
      (effect : Prepare ops p callerBytes before prepared)
      (tail : Trials ops model p chunk callerBytes history prepared after) :
      Trials ops model p chunk callerBytes history before after

theorem trials_private {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β)
    (model : Machine.Model α τ) (p : Private.Layout) (valid : Private.Valid p)
    (chunk : Index.Chunk) (fits : chunk.first+chunk.count≤p.total) (callerBytes : Nat → Nat → Prop)
    (history : List Config) (before after : M)
    (trials : Trials ops model p chunk callerBytes history before after) : SamePrivate ops p after before := by
  induction trials with
  | nil => intro role i active; rfl
  | cons config rest schedule before prepared trial restored after prepare source restore tail ih =>
    intro role i active
    exact (ih role i active).trans
      ((restored_private ops model p valid chunk fits callerBytes prepared trial restored source.physical restore role i active).trans
        (prepare.privateReads role i active))
  | administrative history before prepared after effect tail ih =>
    intro role i active
    exact (ih role i active).trans (effect.privateReads role i active)

theorem trials_bytes {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β)
    (model : Machine.Model α τ) (p : Private.Layout)
    (chunk : Index.Chunk) (callerBytes : Nat → Nat → Prop) (history : List Config) (before after : M)
    (trials : Trials ops model p chunk callerBytes history before after)
    (owner address : Nat) (kept : callerBytes owner address) (outside : Outside p chunk owner address) :
    ops.observe after owner address=ops.observe before owner address := by
  induction trials with
  | nil => rfl
  | cons config rest schedule before prepared trial restored after prepare source restore tail ih =>
    exact ih.trans ((restored_bytes ops model p chunk callerBytes prepared trial restored source.physical restore owner address kept outside).trans
      (prepare.callerBytes owner address kept))
  | administrative history before prepared after effect tail ih =>
    exact ih.trans (effect.callerBytes owner address kept)

structure Launched {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β) (model : Machine.Model α τ) (p : Private.Layout) (chunk : Index.Chunk)
    (callerBytes : Nat → Nat → Prop) (before after : M) where
  key : List Nat
  originalKey : key=[p.total]
  dtypeKey : List String
  originalDtypes : dtypeKey=["torch.float32","torch.float32","torch.float32"]
  history : List Config
  selected : Config
  schedule : List Nat
  restored : M
  trials : Trials ops model p chunk callerBytes history before restored
  selectedSource : Machine.SourceStep ops model p chunk selected schedule restored after

theorem selected_values {M α β τ : Type} [Inhabited α] [Inhabited τ] (ops : Ops M α β)
    (model : Machine.Model α τ) (p : Private.Layout) (valid : Private.Valid p)
    (chunk : Index.Chunk) (fits : chunk.first+chunk.count≤p.total) (callerBytes : Nat → Nat → Prop)
    (before after : M) (launch : Launched ops model p chunk callerBytes before after)
    (output : Output) (i : Nat) (member : i∈chunk.indices) :
    ops.read after ((p.allocation (Private.outputRole output)).key i)=
      Machine.scalar model output (ops.read before ((p.allocation .a).key i))
        (ops.read before ((p.allocation .third).key i)) (ops.read before ((p.allocation .b).key i)) := by
  have bounded : ∀ j∈chunk.indices,j<p.total := by
    intro j member
    have := List.mem_range'_1.mp member
    omega
  have same := trials_private ops model p valid chunk fits callerBytes launch.history before launch.restored launch.trials
  rw [launch.selectedSource.physical,Machine.values ops model p valid chunk.indices
    (List.nodup_range') bounded launch.restored output i member,
    same .a i (bounded i member),same .third i (bounded i member),same .b i (bounded i member)]

#axiomsClean trials_private
#axiomsClean trials_bytes
#axiomsClean selected_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Autotune
