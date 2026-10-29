import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.Global
import bench.repairs.tritonbench_g.triton_argmax.Proofs.DimMemory

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Native
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction Memory
noncomputable section
set_option maxHeartbeats 2000000

/-- Three distinct complete reduction sites occur in the repaired source. -/
structure Sites (n : Nat) where
  flags : Tree n
  present : Tree n
  numeric : Tree n
  flagsComplete : flags.Complete
  presentComplete : present.Complete
  numericComplete : numeric.Complete

def Sites.pick (sites : Sites n) (input : Fin n → Word) : Fin n :=
  nanReduce input sites.flags sites.present sites.numeric

theorem Sites.pick_eq (sites : Sites n) (input : Fin n → Word) :
    sites.pick input=sites.numeric.pick greater equal input :=
  nanReduce_eq input _ _ _ sites.flagsComplete sites.presentComplete sites.numericComplete

/-- The floating scratch sequence ranks decoded values only after the exact
requested conversion. The discrete 0/1 reductions are kept separate. -/
def keyFlag {n : Nat} (keys : Fin n → WithTop Int) (i : Fin n) : Int :=
  if keys i=⊤ then 1 else 0

def keySelect {n : Nat} (sites : Sites n) (padding : WithTop Int) (keys : Fin n → WithTop Int) : Fin n :=
  let first := sites.flags.pick orderedGt orderedEq (keyFlag keys)
  let present := sites.present.pick orderedGt orderedEq (keyFlag keys)
  if 0<keyFlag keys present then first
  else sites.numeric.pick orderedGt orderedEq (fun i => if keys i=⊤ then padding else keys i)

theorem keySelect_correct {n : Nat} (sites : Sites n) (padding : WithTop Int)
    (keys : Fin n → WithTop Int) : IsLeastMaximum keys (keySelect sites padding keys) := by
  have flags := mathematical_maximum (keyFlag keys) sites.flags sites.flagsComplete
  have present := mathematical_maximum (keyFlag keys) sites.present sites.presentComplete
  by_cases any : ∃ i,keys i=⊤
  · obtain ⟨i,hi⟩ := any
    have hfp : 0<keyFlag keys (sites.flags.pick orderedGt orderedEq (keyFlag keys)) := by
      exact (show 0<keyFlag keys i by simp [keyFlag,hi]).trans_le (flags.1 i)
    have hpp : 0<keyFlag keys (sites.present.pick orderedGt orderedEq (keyFlag keys)) := by
      exact (show 0<keyFlag keys i by simp [keyFlag,hi]).trans_le (present.1 i)
    have top : keys (sites.flags.pick orderedGt orderedEq (keyFlag keys))=⊤ := by
      by_contra h
      simp [keyFlag,h] at hfp
    simp only [keySelect,if_pos hpp]
    constructor
    · intro j; rw [top]; exact le_top
    · intro j same
      apply flags.2 j
      simp [keyFlag,top,same]
  · have noTop : ∀ i,keys i≠⊤ := fun i hi => any ⟨i,hi⟩
    have empty : ¬0<keyFlag keys (sites.present.pick orderedGt orderedEq (keyFlag keys)) := by
      simp [keyFlag,noTop]
    simp only [keySelect,if_neg empty,if_neg (noTop _)]
    exact mathematical_maximum keys sites.numeric sites.numericComplete

theorem keySelect_eq {n : Nat} (sites : Sites n) (padding : WithTop Int)
    (keys : Fin n → WithTop Int) :
    keySelect sites padding keys=sites.numeric.pick orderedGt orderedEq keys :=
  least_maximum_unique _ _ _ (keySelect_correct sites padding keys)
    (mathematical_maximum keys sites.numeric sites.numericComplete)

structure GlobalPrograms (c : GlobalConfig) where
  first : Nat → Sites c.block
  second : Sites c.midBlock
  firstNumeric : ∀ pid,(first pid).numeric=c.firstTree pid
  secondNumeric : second.numeric=c.secondTree

/-- This is the source's selected-address reread, not the sanitized max value. -/
def firstIndex (g : GlobalPrograms c) (input : Nat → Word) (pid : Nat) : Nat :=
  ((g.first pid).pick (c.loaded input 0xff800000 pid)).val+pid*c.block

def firstValue {d : MidFormat} (scratch : Scratch d) (g : GlobalPrograms c)
    (input : Nat → Word) (pid : Nat) : MidWord d := scratch.cast (input (firstIndex g input pid))

theorem first_index {d : MidFormat} (scratch : Scratch d) (g : GlobalPrograms c)
    (input : Nat → Word) (pid : Nat) :
    firstIndex g input pid=stage1Index scratch.scalars c input pid := by
  simp only [firstIndex,Sites.pick_eq,g.firstNumeric]
  rfl

theorem first_value {d : MidFormat} (scratch : Scratch d) (g : GlobalPrograms c)
    (input : Nat → Word) (pid : Nat) (positive : 0<c.block) (active : pid*c.block<c.size) :
    firstValue scratch g input pid=stage1Value scratch.scalars c input pid := by
  have valid := selected_valid c.size (pid*c.block) c.block input (c.firstTree pid)
    (g.firstNumeric pid ▸ (g.first pid).numericComplete) active positive
  have loaded : c.loaded input 0xff800000 pid=padded c.size (pid*c.block) c.block input := rfl
  rw [firstValue,first_index scratch g]
  unfold stage1Value stage1Index GlobalConfig.selected
  dsimp only [Scratch.scalars]
  rw [loaded]
  change scratch.cast (input (_+pid*c.block))=scratch.cast (padded c.size (pid*c.block) c.block input _)
  dsimp only [selected] at valid
  rw [padded,if_pos valid,Nat.add_comm]

def firstProgram {M : Type} {d : MidFormat} (mem : Primitives M) (scratch : Scratch d)
    (g : GlobalPrograms c) (p : GlobalMemory.Buffers) (s : M) (pid : Nat) : M :=
  let input := GlobalMemory.input mem p s
  mem.write 8 (mem.write d.bytes s (address p.midValue d.bytes pid) (firstValue scratch g input pid))
    (address p.midIndex 8 pid) (BitVec.ofNat 64 (firstIndex g input pid))

def secondSelected {d : MidFormat} (scratch : Scratch d) (floating : Bool) (g : GlobalPrograms c)
    (mid : Nat → MidWord d) : Fin c.midBlock :=
  let keys := fun lane => scratch.key (if (lane : Fin c.midBlock).val<c.blocks then mid lane.val else scratch.padding)
  if floating then keySelect g.second (scratch.key scratch.padding) keys
  else g.second.numeric.pick orderedGt orderedEq keys

theorem second_selected {d : MidFormat} (scratch : Scratch d) (floating : Bool) (g : GlobalPrograms c)
    (mid : Nat → MidWord d) :
    secondSelected scratch floating g mid=stage2Selected scratch.scalars c mid := by
  unfold secondSelected
  split <;> simp only [keySelect_eq,g.secondNumeric]
  all_goals symm; apply pick_key

def secondProgram {M : Type} {d : MidFormat} (mem : Primitives M) (scratch : Scratch d)
    (floating : Bool) (g : GlobalPrograms c) (p : GlobalMemory.Buffers) (s : M) : M :=
  let selected := secondSelected scratch floating g (fun i => mem.read d.bytes s (address p.midValue d.bytes i))
  let payload := mem.read 8 s (address p.midIndex 8 selected.val)
  mem.write 8 s (address p.output 8 0) payload

def global {M : Type} {d : MidFormat} (mem : Primitives M) (scratch : Scratch d)
    (floating : Bool) (g : GlobalPrograms c) (p : GlobalMemory.Buffers) (s : M) : M :=
  secondProgram mem scratch floating g p ((List.range c.blocks).foldl (firstProgram mem scratch g p) s)

theorem global_source {M : Type} {d : MidFormat} (mem : Primitives M) (scratch : Scratch d)
    (floating : Bool) (g : GlobalPrograms c) (p : GlobalMemory.Buffers) (s : M)
    (positive : 0<c.block) (active : ∀ pid,pid<c.blocks → pid*c.block<c.size) :
    global mem scratch floating g p s=GlobalMemory.run mem scratch.scalars c p s := by
  have first : ∀ state pid,pid∈List.range c.blocks →
      firstProgram mem scratch g p state pid=GlobalMemory.firstLive mem scratch.scalars c p state pid := by
    intro state pid hp
    simp only [firstProgram,first_value scratch g _ pid positive (active pid (List.mem_range.mp hp)),
      first_index scratch g]
    rfl
  unfold global GlobalMemory.run
  have foldAux : ∀ jobs : List Nat,(∀ pid,pid∈jobs → pid<c.blocks) → ∀ state,
      jobs.foldl (firstProgram mem scratch g p) state=
        jobs.foldl (GlobalMemory.firstLive mem scratch.scalars c p) state := by
    intro jobs valid
    induction jobs with
    | nil => intro state; rfl
    | cons pid rest ih =>
      intro state
      rw [List.foldl_cons,ih (fun j hj => valid j (by simp [hj])),
        first state pid (List.mem_range.mpr (valid pid (by simp))),List.foldl_cons]
  have fold := foldAux (List.range c.blocks) (fun pid hp => List.mem_range.mp hp) s
  rw [fold]
  simp only [secondProgram,second_selected]
  rfl

structure DimPrograms (c : DimConfig) where
  sites : Nat → Nat → Nat → Sites 128
  numeric : ∀ row inner tile,(sites row inner tile).numeric=c.tree row inner tile

def dimLocal (g : DimPrograms c) (input : Nat → Word) (row inner tile : Nat) : Word × Nat :=
  let index := (g.sites row inner tile).pick (c.loaded input 0xff800000 row inner tile)
  let col := tile*128+index.val
  (if row<c.outer ∧ col<c.extent then input (dimInputOffset c row col inner) else 0xff800000,col)

def dimStep (g : DimPrograms c) (input : Nat → Word) (row inner : Nat)
    (current : Word × Nat) (tile : Nat) : Word × Nat :=
  let next := dimLocal g input row inner tile
  if ((!VeriTile.Bench.TritonBenchG.MatrixReduction.MachineMax.isNaN .fp32 current.1) &&
      (VeriTile.Bench.TritonBenchG.MatrixReduction.MachineMax.isNaN .fp32 next.1 || ScalarBinding.gt32 next.1 current.1))
  then next else current

def dimIndex (g : DimPrograms c) (input : Nat → Word) (row inner : Nat) : Nat :=
  ((List.range ((c.extent+127)/128)).foldl (dimStep g input row inner) (0xff800000,0)).2

theorem dim_local (g : DimPrograms c) (input : Nat → Word) (row inner tile : Nat) :
    dimLocal g input row inner tile=Source.dimLocal fp32Scalars c input row inner tile := by
  simp only [dimLocal,Sites.pick_eq,g.numeric]
  rfl

theorem dim_index (g : DimPrograms c) (input : Nat → Word) (row inner : Nat) :
    dimIndex g input row inner=Source.dimIndex fp32Scalars c input row inner := by
  have step : dimStep g input row inner=Source.dimStep fp32Scalars c input row inner := by
    funext current tile
    simp only [dimStep,update_exact,dim_local]
    rfl
  simp only [dimIndex,step,Source.dimIndex,dimTiles,blockN,fp32Scalars,
    show c.extent+128-1=c.extent+127 by omega]

def dimension {M : Type} (mem : Primitives M) (g : DimPrograms c)
    (p : DimMemory.Buffers) (s : M) : M :=
  (DimMemory.jobs c).foldl (fun state job =>
    if job.1<c.outer then mem.write 8 state (address p.output 8 (dimOutputOffset c job.1 job.2))
      (BitVec.ofNat 64 (dimIndex g (DimMemory.input mem p state) job.1 job.2)) else state) s

theorem dimension_source {M : Type} (mem : Primitives M) (g : DimPrograms c)
    (p : DimMemory.Buffers) (s : M) :
    dimension mem g p s=DimMemory.run mem fp32Scalars c p s := by
  simp only [dimension,dim_index]
  rfl

#axiomsClean keySelect_correct
#axiomsClean first_value
#axiomsClean global_source
#axiomsClean dimension_source
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Native
