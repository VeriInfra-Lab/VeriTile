import bench.repairs.tritonbench_g.rope_embedding.Proofs.Native
namespace VeriTile.Bench.Repairs.RopeEmbedding.Native
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

def prior (r : Request) (a g : Nat) : List (Location r) :=
  (schedule r).filter fun l => decide (row l<a ∨ row l=a ∧ group l<g)
theorem prior_scheduled (r : Request) (a g : Nat) (l : Location r) (hl : l∈prior r a g) : l∈schedule r :=
  (List.mem_filter.mp hl).1
theorem prior_other (r : Request) (a g : Nat) (l : Location r) (hl : l∈prior r a g) : row l≠a ∨ group l≠g := by
  have h := (List.mem_filter.mp hl).2
  simp only [decide_eq_true_eq] at h
  omega

def prefixMemory {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (back : Bool) (s : State α) (call : Preparation api embed r s)
    (a g : Nat) :=
  Memory.runWrites api (Memory.events project op r back s call.output.base (prior r a g)) call.memory

def snapshot {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α)
    (r : Request) (s : State α) (call : Preparation api embed r s) (m : Store) : State α :=
  {s with mem:=fun region o =>
    if region="Q" then .mk .real (embed (api.read m (call.output.base+4*o)))
    else if region="C" then .mk .real (embed (api.read m (call.cosine.base+4*o)))
    else if region="S" then .mk .real (embed (api.read m (call.sine.base+4*o)))
    else s.mem region o}

theorem prefix_coefficient {α : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (embed : β→α) (project : α→β) (op : Operations α) (r : Request) (back : Bool)
    (s : State α) (call : Preparation api embed r s) (a g base count k : Nat)
    (hk : k<count) (sep : call.output.base+4*r.size≤base ∨ base+4*count≤call.output.base) :
    api.read (prefixMemory api embed project op r back s call a g) (base+4*k)=api.read call.memory (base+4*k) := by
  apply Memory.writes_read_frame
  intro e he
  have h := Memory.active_window project op r back s call.output.base (prior r a g) e he
  omega

theorem snapshot_Q {α : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (embed : β→α) (project : α→β) (op : Operations α) (r : Request) (back : Bool)
    (s : State α) (call : Preparation api embed r s) (a g h c : Nat)
    (ha : a<r.rows) (hh : h<r.heads) (hc : c<r.dimension) (hg : h/4=g) :
    ((snapshot api embed r s call (prefixMemory api embed project op r back s call a g)).mem "Q" (r.offset a h c)).read .real=
      (s.mem "Q" (r.offset a h c)).read .real := by
  change embed (api.read (prefixMemory api embed project op r back s call a g)
    (call.output.base+4*r.offset a h c))=(s.mem "Q" (r.offset a h c)).read .real
  rw [prefixMemory,Memory.prefix_current api project op r back s call.output.base call.memory
    (prior r a g) a g h c hh hc hg (prior_scheduled r a g) (prior_other r a g)]
  rw [call.preparedInput a h c ha hh hc,call.inputBinding a h c ha hh hc]

theorem snapshot_C {α : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (embed : β→α) (project : α→β) (op : Operations α) (r : Request) (back : Bool)
    (s : State α) (call : Preparation api embed r s) (a g k : Nat) (hk : k<r.sequence*r.half) :
    ((snapshot api embed r s call (prefixMemory api embed project op r back s call a g)).mem "C" k).read .real=
      (s.mem "C" k).read .real := by
  change embed (api.read (prefixMemory api embed project op r back s call a g)
    (call.cosine.base+4*k))=(s.mem "C" k).read .real
  rw [prefix_coefficient api embed project op r back s call a g call.cosine.base _ k hk call.separateCosine,
    call.preservedCosine k hk,call.cosineBinding k hk]

theorem snapshot_S {α : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (embed : β→α) (project : α→β) (op : Operations α) (r : Request) (back : Bool)
    (s : State α) (call : Preparation api embed r s) (a g k : Nat) (hk : k<r.sequence*r.half) :
    ((snapshot api embed r s call (prefixMemory api embed project op r back s call a g)).mem "S" k).read .real=
      (s.mem "S" k).read .real := by
  change embed (api.read (prefixMemory api embed project op r back s call a g)
    (call.sine.base+4*k))=(s.mem "S" k).read .real
  rw [prefix_coefficient api embed project op r back s call a g call.sine.base _ k hk call.separateSine,
    call.preservedSine k hk,call.sineBinding k hk]

theorem coefficient_bound (r : Request) (a j : Nat) (ha : a<r.rows) (hj : j<r.half) :
    a%r.sequence*r.half+j<r.sequence*r.half := by
  have hs : 0<r.sequence := by
    unfold Request.rows at ha
    by_contra h
    have hz : r.sequence=0 := by omega
    simp [hz] at ha
  have hm := Nat.mod_lt a hs
  have hp := Nat.mul_le_mul_right r.half (show a%r.sequence+1≤r.sequence by omega)
  rw [Nat.add_mul,Nat.one_mul] at hp
  omega

/-- Every program executes the actual typed source from the physical memory
after all preceding programs. Both Q halves and coefficients are bound to
those real prefix reads. Its final stores match the enumerated source events. -/
theorem source_execution_prefix {α : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (embed : β→α) (project : α→β) (op : Operations α) (r : Request) (back : Bool)
    (s : State α) (call : Preparation api embed r s) (l : Location r) (hl : l∈schedule r) :
    ∃ final,FP.Structural.exec op.backend (Source.source (parameters r) back)
      (programState (snapshot api embed r s call
        (prefixMemory api embed project op r back s call (row l) (group l))) (row l) (group l))=some final ∧
      (active l → final.mem "Q" (r.offset (row l) (head l) (column l))=.mk .real (sourceValue op r back s l)) := by
  let t := snapshot api embed r s call (prefixMemory api embed project op r back s call (row l) (group l))
  obtain ⟨final,he,hv⟩ := source_execution op r back t l hl
  refine ⟨final,he,?_⟩
  intro ha
  rw [hv ha,source_value op r back s l ha]
  congr 1
  rw [column,scalar_slot op r back _ _ _ _ _ _ ha.2.2,scalar_slot op r back _ _ _ _ _ _ ha.2.2]
  have hj := ha.2.2
  have hdim : 2*r.half≤r.dimension := by unfold Request.half; omega
  have hg := group_coordinate r l hl
  have hc := coefficient_bound r (row l) (lane l).1.val ha.1 hj
  dsimp only [t]
  rw [snapshot_Q api embed project op r back s call _ _ _ _ ha.1 ha.2.1 (by omega) hg,
    snapshot_Q api embed project op r back s call _ _ _ _ ha.1 ha.2.1 (by omega) hg,
    snapshot_C api embed project op r back s call _ _ _ hc,
    snapshot_S api embed project op r back s call _ _ _ hc]

#axiomsClean snapshot_Q
#axiomsClean snapshot_C
#axiomsClean snapshot_S
#axiomsClean source_execution_prefix
end
end VeriTile.Bench.Repairs.RopeEmbedding.Native
