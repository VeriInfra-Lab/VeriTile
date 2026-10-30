import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Address
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Shared.Execution

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing
open VeriTile.Bench.TritonBenchG.NestedLoopsProcessing
open VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Address (Dense address_apart)
open VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Execution (Disjoint disjoint_words)
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section
open scoped Classical

structure Placement (count : Nat) where
  input : Dense count
  output : Dense count
  apart : Disjoint output input

structure Copy (count : Nat) where
  destination : Fin count
  source : Nat

def laneCopy (count cols outputShift inputShift : Nat) (lane : Lane) : Option (Copy count) :=
  if h:address cols outputShift lane<count ∧ activeLane cols lane then
    some ⟨⟨address cols outputShift lane,h.1⟩,address cols inputShift lane⟩ else none

def tileCopies (count cols outputShift inputShift : Nat) : List (Copy count) :=
  lanes.filterMap (laneCopy count cols outputShift inputShift)

def copies (count cols : Nat) : List (Copy count) :=
  (List.range 24).flatMap (fun t=>tileCopies count cols (2*t) (CopyMath.inputShift t))

def copyValue (zero : α) (count : Nat) (input : Nat → α) (copy : Copy count) : α :=
  if copy.source<count then input copy.source else zero

def wordStep (zero : α) (count : Nat) (input : Nat → α) (memory : Nat → α) (copy : Copy count) : Nat → α :=
  Function.update memory copy.destination.val (copyValue zero count input copy)

def wordRun (zero : α) (count : Nat) (input : Nat → α) (order : List (Copy count)) (initial : Nat → α) : Nat → α :=
  order.foldl (wordStep zero count input) initial

theorem unique_tile_copies (zero : α) (count cols outputShift inputShift : Nat)
    (input memory : Nat → α) :
    wordRun zero count input (tileCopies count cols outputShift inputShift) memory=
      uniqueTile count cols outputShift (inputTile zero count cols input inputShift) memory := by
  have fold : ∀ ls : List Lane,∀ memory : Nat → α,
      (ls.filterMap (laneCopy count cols outputShift inputShift)).foldl (wordStep zero count input) memory=
      ls.foldl (fun m lane=>if activeLane cols lane then
        put count (address cols outputShift lane) (inputTile zero count cols input inputShift lane) m else m) memory := by
    intro ls
    induction ls with
    | nil=>intro;rfl
    | cons lane ls ih=>
      intro memory
      by_cases active:activeLane cols lane <;> by_cases bound:address cols outputShift lane<count <;>
        simpa [laneCopy,active,bound,put,wordStep,copyValue,inputTile] using
          ih (if activeLane cols lane then
            put count (address cols outputShift lane) (inputTile zero count cols input inputShift lane) memory else memory)
  exact fold lanes memory

theorem word_run_source (zero : α) (count cols : Nat) (input initial : Nat → α) :
    wordRun zero count input (copies count cols) initial=logicalRun zero count cols input initial := by
  unfold logicalRun
  rw [source_store_schedule,List.foldl_map]
  have fold : ∀ ts : List Nat,∀ initial : Nat → α,
      wordRun zero count input (ts.flatMap (fun t=>tileCopies count cols (2*t) (CopyMath.inputShift t))) initial=
      ts.foldl (fun memory t=>uniqueTile count cols (2*t)
        (inputTile zero count cols input (CopyMath.inputShift t)) memory) initial := by
    intro ts
    induction ts with
    | nil=>intro;rfl
    | cons t ts ih=>
      intro initial
      change wordRun zero count input
        (tileCopies count cols (2*t) (CopyMath.inputShift t)++
          ts.flatMap (fun t=>tileCopies count cols (2*t) (CopyMath.inputShift t))) initial=_
      simp only [wordRun,List.foldl_append]
      rw [show (tileCopies count cols (2*t) (CopyMath.inputShift t)).foldl (wordStep zero count input) initial=
        uniqueTile count cols (2*t) (inputTile zero count cols input (CopyMath.inputShift t)) initial from
          unique_tile_copies zero count cols (2*t) (CopyMath.inputShift t) input initial]
      exact ih _
  exact fold (List.range 24) initial

def readInput (api : API α Store β) (p : Placement count) (memory : Store) : Nat → α :=
  fun i=>api.read memory (p.input.address i)

def copyStep (api : API α Store β) (zero : α) (p : Placement count)
    (memory : Store) (copy : Copy count) : Store :=
  api.write memory (p.output.address copy.destination.val) (copyValue zero count (readInput api p memory) copy)

def execute (api : API α Store β) (zero : α) (p : Placement count)
    (order : List (Copy count)) (memory : Store) : Store :=
  order.foldl (copyStep api zero p) memory

def StableInput (api : API α Store β) (p : Placement count) (before after : Store) : Prop :=
  ∀ i,i<count → readInput api p after i=readInput api p before i

theorem step_input (api : API α Store β) (zero : α) (p : Placement count)
    (memory : Store) (copy : Copy count) : StableInput api p memory (copyStep api zero p memory copy) := by
  intro i hi
  apply api.read_apart
  exact disjoint_words p.output p.input p.apart _ _ copy.destination.isLt hi

theorem execute_input (api : API α Store β) (zero : α) (p : Placement count)
    (order : List (Copy count)) (memory : Store) : StableInput api p memory (execute api zero p order memory) := by
  induction order generalizing memory with
  | nil=>intro i hi;rfl
  | cons copy rest ih=>
    intro i hi
    exact (ih (copyStep api zero p memory copy) i hi).trans (step_input api zero p memory copy i hi)

theorem copy_value_stable (api : API α Store β) (zero : α) (p : Placement count)
    (before after : Store) (stable : StableInput api p before after) (copy : Copy count) :
    copyValue zero count (readInput api p after) copy=copyValue zero count (readInput api p before) copy := by
  by_cases active:copy.source<count
  · simp [copyValue,active,stable copy.source active]
  · simp [copyValue,active]

def physicalEvent (zero : α) (p : Placement count) (input : Nat → α) (copy : Copy count) : Write α :=
  ⟨p.output.address copy.destination.val,copyValue zero count input copy⟩

theorem execute_static_aux (api : API α Store β) (zero : α) (p : Placement count)
    (order : List (Copy count)) (initial memory : Store) (stable : StableInput api p initial memory) :
    execute api zero p order memory=run api (order.map (physicalEvent zero p (readInput api p initial))) memory := by
  induction order generalizing memory with
  | nil=>rfl
  | cons copy rest ih=>
    change execute api zero p rest (copyStep api zero p memory copy)=_
    have next:StableInput api p initial (copyStep api zero p memory copy) :=
      fun i hi=>(step_input api zero p memory copy i hi).trans (stable i hi)
    rw [ih _ next]
    simp only [run,List.map_cons,List.foldl_cons,physicalEvent,copyStep,
      copy_value_stable api zero p initial memory stable copy]

theorem execute_static (api : API α Store β) (zero : α) (p : Placement count)
    (order : List (Copy count)) (memory : Store) :
    execute api zero p order memory=run api (order.map (physicalEvent zero p (readInput api p memory))) memory :=
  execute_static_aux api zero p order memory memory (fun _ _=>rfl)

theorem write_read_map (api : API α Store β) (buffer : Dense count) (memory : Store)
    (destination : Nat) (value : α) :
    (fun i=>api.read (api.write memory (buffer.address destination) value) (buffer.address i))=
      Function.update (fun i=>api.read memory (buffer.address i)) destination value := by
  funext i
  by_cases same:i=destination
  · subst i;simp [api.read_same]
  · rw [api.read_apart _ _ _ _ (address_apart buffer (Ne.symm same))]
    simp [Function.update_of_ne same]

theorem static_readback (api : API α Store β) (zero : α) (p : Placement count)
    (input : Nat → α) (order : List (Copy count)) (memory : Store) :
    (fun i=>api.read (run api (order.map (physicalEvent zero p input)) memory) (p.output.address i))=
      wordRun zero count input order (fun i=>api.read memory (p.output.address i)) := by
  induction order generalizing memory with
  | nil=>rfl
  | cons copy rest ih=>
    change (fun i=>api.read (run api (rest.map (physicalEvent zero p input))
      (api.write memory (p.output.address copy.destination.val) (copyValue zero count input copy))) (p.output.address i))=_
    rw [ih,write_read_map]
    rfl

theorem execute_readback (api : API α Store β) (zero : α) (p : Placement count)
    (cols : Nat) (positive : 0<cols) (memory : Store) (i : Nat) (bound : i<count) :
    api.read (execute api zero p (copies count cols) memory) (p.output.address i)=
      target zero count cols (readInput api p memory) (fun j=>api.read memory (p.output.address j)) i := by
  rw [execute_static]
  have readback:=congrFun (static_readback api zero p (readInput api p memory) (copies count cols) memory) i
  rw [word_run_source] at readback
  exact readback.trans (logical_readback zero count cols positive _ _ i bound)

theorem word_run_congr (zero : α) (count : Nat) (order : List (Copy count))
    (a b x y : Nat → α) (input : Inside count a b) (initial : Inside count x y) :
    Inside count (wordRun zero count a order x) (wordRun zero count b order y) := by
  induction order generalizing x y with
  | nil=>exact initial
  | cons copy rest ih=>
    apply ih
    intro i hi
    by_cases same:i=copy.destination.val
    · subst i
      by_cases bound:copy.source<count
      · simp [wordStep,copyValue,bound,input copy.source bound]
      · simp [wordStep,copyValue,bound]
    · simp [wordStep,Function.update_of_ne same,initial i hi]

theorem target_congr_inside {α : Type} (zero : α) (count cols : Nat) (positive : 0<cols)
    (a b x y : Nat → α) (input : Inside count a b) (initial : Inside count x y) :
    Inside count (target zero count cols a x) (target zero count cols b y) := by
  have h:=word_run_congr zero count (copies count cols) a b x y input initial
  rw [word_run_source,word_run_source] at h
  intro i hi
  exact (logical_readback zero count cols positive a x i hi).symm.trans
    ((h i hi).trans (logical_readback zero count cols positive b y i hi))

theorem execute_frame (api : API α Store β) (zero : α) (p : Placement count)
    (order : List (Copy count)) (memory : Store) (byte : Nat) (outside : ¬p.output.contains byte) :
    api.byte (execute api zero p order memory) byte=api.byte memory byte := by
  rw [execute_static]
  apply run_byte_frame
  intro event member
  obtain ⟨copy,_,rfl⟩:=List.mem_map.mp member
  have bound:=copy.destination.isLt
  simp only [physicalEvent,Dense.contains,Dense.address] at *
  omega

/-- Cached source loads and a later dynamic reread agree after every actual
completed store prefix. This justifies cache elimination from the mutable
physical interpreter; it is derived from fresh allocation separation. -/
theorem cached_read_after_prefix (api : API α Store β) (zero : α) (p : Placement count)
    (cols completed shift : Nat) (memory : Store) :
    inputTile zero count cols (readInput api p (execute api zero p ((copies count cols).take completed) memory)) shift=
      inputTile zero count cols (readInput api p memory) shift := by
  funext lane
  by_cases active:address cols shift lane<count
  · simp [inputTile,active,execute_input api zero p ((copies count cols).take completed) memory _ active]
  · simp [inputTile,active]

/-- Ordinary little-endian int32 storage, a local memory representation law.
It is neither an arithmetic port nor a whole-output premise. -/
structure Codec (api : API Word Store Byte) : Prop where
  byte : ∀ memory address offset,offset<4 →
    api.byte memory (address+offset)=(api.read memory address).extractLsb' (8*offset) 8

theorem execute_output_bytes (api : API Word Store Byte) (codec : Codec api)
    (p : Placement count) (cols : Nat) (positive : 0<cols) (memory : Store)
    (i offset : Nat) (bound : i<count) (small : offset<4) :
    api.byte (execute api 0 p (copies count cols) memory) (p.output.address i+offset)=
      (target 0 count cols (readInput api p memory) (fun j=>api.read memory (p.output.address j)) i).extractLsb' (8*offset) 8 := by
  rw [codec.byte _ _ _ small,execute_readback api 0 p cols positive memory i bound]

#axiomsClean word_run_source
#axiomsClean execute_static
#axiomsClean execute_input
#axiomsClean execute_readback
#axiomsClean target_congr_inside
#axiomsClean execute_frame
#axiomsClean cached_read_after_prefix
#axiomsClean execute_output_bytes
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing
