import bench.tritonbench_g.adam_update_triton.PairIO
import bench.tritonbench_g.adam_update_triton.Proofs.NativeWords
import bench.tritonbench_g.adam_update_triton.Proofs.Grid
import bench.tritonbench_g.adam_update_triton.Proofs.CommutedGrid
import bench.tritonbench_g.adam_update_triton.Proofs.AccessSafety
import bench.tritonbench_g.adam_update_triton.Proofs.CommutedSafety
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! The two actual runtime-scalar sources on a complete grid, in any grid
permutation. Every invocation executes the quoted body. The logical inputs are
arbitrary FP32 words, including all NaN payloads. The native observation keeps
source FMA contraction; no arithmetic rewrite or finite-word guard is assumed. -/
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.WholePair
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

def execute {α : Type} [Inhabited α] (choice : Choice) (sem : Tile.Semantics α)
    (params : Scalars α) (total n B : Nat) (pids : List Nat) (s : State α) : Option (State α) :=
  match choice with
  | .original => Grid.runPrograms sem params total n B pids s
  | .commuted => CommutedGrid.runPrograms sem params total n B pids s

def scalar (choice : Choice) (sem : Tile.Semantics α) (params : Scalars α)
    (output : Output) (p g m : α) : α :=
  match choice with
  | .original => Tile.scalar sem params output p g m
  | .commuted => CommutedTile.scalar sem params output p g m

theorem invocation_run {α : Type} [Inhabited α] (choice : Choice)
    (sem : Tile.Semantics α) (params : Scalars α) (total n B pid : Nat) (s : State α) :
    (invocation choice sem params total n B pid).run s =
      run sem.backend (kernel choice total n B).surfaceBody (Tile.programState params pid s) := by
  cases choice <;> rfl

theorem schedule_execution {α : Type} [Inhabited α] (choice : Choice)
    (sem : Tile.Semantics α) (params : Scalars α) (total n B : Nat)
    (pids : List Nat) (s : State α) :
    runSchedule (pids.map (invocation choice sem params total n B)) s =
      execute choice sem params total n B pids s := by
  induction pids generalizing s with
  | nil => cases choice <;> rfl
  | cons pid rest ih =>
    simp only [List.map_cons, runSchedule, invocation_run]
    cases choice <;> simp only [execute, Grid.runPrograms, CommutedGrid.runPrograms, kernel]
    · exact congrArg (fun f => (run sem.backend (source total n B).surfaceBody
        (Tile.programState params pid s)).bind f) (funext ih)
    · exact congrArg (fun f => (run sem.backend (commutedSource total n B).surfaceBody
        (Tile.programState params pid s)).bind f) (funext ih)

/-- Both complete bodies retain the same full-window read/write access safety. -/
theorem schedule_safe {α : Type} [Inhabited α] (choice : Choice)
    (sem : Tile.Semantics α) (params : Scalars α) (total n B : Nat)
    (bounds : RegionName → Nat)
    (hp : ∀ i, i < n → i < bounds "P")
    (hg : ∀ i, i < n → i < bounds "G")
    (hm : ∀ i, i < n → i < bounds "M")
    (pids : List Nat) (s : State α) :
    ScheduleSafe bounds (pids.map (invocation choice sem params total n B)) s := by
  induction pids generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    refine ⟨?_, fun t _ => ih t⟩
    cases choice
    · exact Tile.source_traceSafe sem params total n B pid bounds s
        (fun j h => hp _ h) (fun j h => hg _ h) (fun j h => hm _ h)
    · exact CommutedTile.source_traceSafe sem params total n B pid bounds s
        (fun j h => hp _ h) (fun j h => hg _ h) (fun j h => hm _ h)

theorem whole_execution {α : Type} [Inhabited α] (choice : Choice)
    (sem : Tile.Semantics α) (params : Scalars α) (total n B : Nat)
    (positive : 0 < B) (pids : List Nat)
    (complete : pids.Perm (List.range ((n+B-1)/B))) (s : State α) :
    ∃ final, execute choice sem params total n B pids s = some final ∧
      (∀ output i, i < n → final.mem (Tile.outputRegion output) i = Cell.mk .real
        (scalar choice sem params output ((s.mem "P" i).read .real)
          ((s.mem "G" i).read .real) ((s.mem "M" i).read .real))) ∧
      (∀ r offset, Grid.Outside n r offset → final.mem r offset = s.mem r offset) := by
  cases choice
  · exact Grid.whole sem params total n B positive pids complete s
  · exact CommutedGrid.whole sem params total n B positive pids complete s

def target (native : NativeWords.Instructions) (params : Scalars Native.Word)
    {n : Nat} (values : Interface.Inputs Native.Word n) : Interface.Outputs Native.Word n :=
  Interface.result
    (fun j => Native.reference (NativeWords.ports native) params .parameter
      (Interface.parameter values j) (Interface.gradient values j) (Interface.momentum values j))
    (fun j => Native.reference (NativeWords.ports native) params .momentum
      (Interface.parameter values j) (Interface.gradient values j) (Interface.momentum values j))

theorem graph_lowering (choice : Choice) (native : NativeWords.Instructions)
    (params : Scalars Native.Word) (output : Output) (p g m : Native.Word) :
    Native.interpret (NativeWords.ports native)
      (scalar choice (Native.semantics (NativeWords.ports native)) (Native.arguments params)
        output (.input p) (.input g) (.input m)) =
      Native.reference (NativeWords.ports native) params output p g m := by
  cases choice
  · exact Native.lowering _ _ _ _ _ _
  · exact Native.commuted_lowering _ _ _ _ _ _

theorem outside_grid (choice : Choice) (native : NativeWords.Instructions)
    (params : Scalars Native.Word) (total n B : Nat) (pids : List Nat)
    (values : Interface.Inputs Native.Word n) (r : RegionName) (offset : Nat)
    (outside : (programIO choice native params total n B pids).Outside values r offset) :
    Grid.Outside n r offset := by
  by_cases active : offset < n
  · left
    constructor
    · intro same
      have h := outside.1 ⟨0, by simp [programIO,Interface.memory,Interface.outputs]⟩
        (⟨offset,active⟩, PUnit.unit) (by simpa [programIO,Interface.memory,Interface.outputLane])
      simpa [programIO,Interface.memory,Interface.outputs,Interface.outputLane,same] using h
    · intro same
      have h := outside.1 ⟨1, by simp [programIO,Interface.memory,Interface.outputs]⟩
        (⟨offset,active⟩, PUnit.unit) (by simpa [programIO,Interface.memory,Interface.outputLane])
      simpa [programIO,Interface.memory,Interface.outputs,Interface.outputLane,same] using h
  · exact Or.inr (by omega)

/-- Full-grid implementation: safety, successful source execution, every typed
output cell, and the exact exterior cell frame. Its domain contains every word. -/
theorem implements (choice : Choice) (native : NativeWords.Instructions)
    (params : Scalars Native.Word) (total n B : Nat) (positive : 0 < B)
    (pids : List Nat) (complete : pids.Perm (List.range ((n+B-1)/B))) :
    (programIO choice native params total n B pids).Implements (target native params) := by
  apply ProgramIO.Implements.intro
  · intro values bounds hb s _
    apply schedule_safe
    · intro i hi
      simpa [programIO,Interface.memory,Interface.inputs,Interface.inputLane] using
        hb.2.1 ⟨0, by simp [programIO,Interface.memory,Interface.inputs]⟩ (⟨i,hi⟩, PUnit.unit)
          (by simpa [programIO,Interface.memory,Interface.inputLane])
    · intro i hi
      simpa [programIO,Interface.memory,Interface.inputs,Interface.inputLane] using
        hb.2.1 ⟨1, by simp [programIO,Interface.memory,Interface.inputs]⟩ (⟨i,hi⟩, PUnit.unit)
          (by simpa [programIO,Interface.memory,Interface.inputLane])
    · intro i hi
      simpa [programIO,Interface.memory,Interface.inputs,Interface.inputLane] using
        hb.2.1 ⟨2, by simp [programIO,Interface.memory,Interface.inputs]⟩ (⟨i,hi⟩, PUnit.unit)
          (by simpa [programIO,Interface.memory,Interface.inputLane])
  · intro values s hin
    have hp (j : Fin n) : (s.mem "P" j.val).read .real = .input (Interface.parameter values j) := by
      simpa [programIO,Interface.memory,Interface.inputs,Interface.inputLane,Interface.parameter,mapValue] using
        hin ⟨0, by simp [programIO,Interface.memory,Interface.inputs]⟩ (j,PUnit.unit)
          (by simpa [programIO,Interface.memory,Interface.inputLane] using j.isLt)
    have hg (j : Fin n) : (s.mem "G" j.val).read .real = .input (Interface.gradient values j) := by
      simpa [programIO,Interface.memory,Interface.inputs,Interface.inputLane,Interface.gradient,mapValue] using
        hin ⟨1, by simp [programIO,Interface.memory,Interface.inputs]⟩ (j,PUnit.unit)
          (by simpa [programIO,Interface.memory,Interface.inputLane] using j.isLt)
    have hm (j : Fin n) : (s.mem "M" j.val).read .real = .input (Interface.momentum values j) := by
      simpa [programIO,Interface.memory,Interface.inputs,Interface.inputLane,Interface.momentum,mapValue] using
        hin ⟨2, by simp [programIO,Interface.memory,Interface.inputs]⟩ (j,PUnit.unit)
          (by simpa [programIO,Interface.memory,Interface.inputLane] using j.isLt)
    obtain ⟨final,ran,valuesOut,frame⟩ := whole_execution choice
      (Native.semantics (NativeWords.ports native)) (Native.arguments params)
      total n B positive pids complete s
    refine ⟨final, ?_, ?_, ?_⟩
    · exact (schedule_execution choice _ _ total n B pids s).trans ran
    · intro o j active
      fin_cases o
      · dsimp [programIO,Interface.memory,Interface.outputs,Interface.outputLane,target,Interface.result]
        simp only [Nat.zero_mul,Nat.zero_add]
        change mapCell (Native.interpret (NativeWords.ports native)) (final.mem "P" j.1.val) =
          Cell.mk .real (Native.reference (NativeWords.ports native) params .parameter
            (Interface.parameter values j.1) (Interface.gradient values j.1) (Interface.momentum values j.1))
        have raw := valuesOut .parameter j.1.val j.1.isLt
        simp only [Tile.outputRegion] at raw
        rw [raw, hp j.1, hg j.1, hm j.1]
        exact congrArg (Cell.mk .real) (graph_lowering choice native params .parameter _ _ _)
      · dsimp [programIO,Interface.memory,Interface.outputs,Interface.outputLane,target,Interface.result]
        simp only [Nat.zero_mul,Nat.zero_add]
        change mapCell (Native.interpret (NativeWords.ports native)) (final.mem "M" j.1.val) =
          Cell.mk .real (Native.reference (NativeWords.ports native) params .momentum
            (Interface.parameter values j.1) (Interface.gradient values j.1) (Interface.momentum values j.1))
        have raw := valuesOut .momentum j.1.val j.1.isLt
        simp only [Tile.outputRegion] at raw
        rw [raw, hp j.1, hg j.1, hm j.1]
        exact congrArg (Cell.mk .real) (graph_lowering choice native params .momentum _ _ _)
    · intro r offset outside
      exact frame r offset (outside_grid choice native params total n B pids values r offset outside)

#axiomsClean implements
#axiomsClean graph_lowering
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.WholePair
