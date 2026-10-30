import bench.tritonbench_g.int8_matmul_kernel.Proofs.TileOutput

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 8000000

abbrev Backing := RegionName × Nat → Cell Word32
def flatten (state : State Word32) : Backing := fun loc => state.mem loc.1 loc.2
def tileState (template : State Word32) (pid : Nat) : State Word32 :=
  { template with pids := fun ax => if ax = 0 then pid else template.pids ax }
def programState (template : State Word32) (memory : Backing) (pid : Nat) : State Word32 :=
  { tileState template pid with mem := fun r o => memory (r,o) }

/-- Program execution uses the full source quotation projection and evolving
common memory. This relation contains actual evaluator runs, not a presumed
native/kernel lowering equation. -/
inductive GridExec (cast : Word32 → Byte) (rest : Algebra Word32) (p : Parameters)
    (template : State Word32) : List Nat → Backing → Backing → Prop
  | nil (memory) : GridExec cast rest p template [] memory memory
  | cons {pid tail before after} (next : State Word32)
      (step : FP.Structural.exec (backend cast rest) (source p)
        (programState template before pid) = some next)
      (following : GridExec cast rest p template tail (flatten next) after) :
      GridExec cast rest p template (pid :: tail) before after

def tileWrites (layout : Layout) (s : Shape) (c : Fin 15) (template : State Word32)
    (pid : Nat) (loc : RegionName × Nat) : Prop :=
  ∃ i : CIndex (parameters layout s c), active (parameters layout s c) (tileState template pid) i ∧
    loc = (layout.output, outputOffset (parameters layout s c) (tileState template pid) i)
instance (layout : Layout) (s : Shape) (c : Fin 15) (template : State Word32) (pid : Nat) :
    DecidablePred (tileWrites layout s c template pid) := Classical.decPred _

def denseCell (layout : Layout) (s : Shape) (initial : State Word32) (loc : RegionName × Nat) : Cell Word32 :=
  int32Cell (matrixWordTarget layout s initial (loc.2 / max s.n 1) (loc.2 % max s.n 1))
def frozenStep (layout : Layout) (s : Shape) (c : Fin 15) (initial : State Word32) :
    Nat → Backing → Backing :=
  Schedule.put (tileWrites layout s c initial) (fun _ => denseCell layout s initial)

def SameInputs (layout : Layout) (initial : State Word32) (memory : Backing) : Prop :=
  ∀ r o, (r = layout.input ∨ r = layout.weight) → memory (r,o) = initial.mem r o

theorem program_frozen (cast : Word32 → Byte)
    (castLeaf : ∀ word, DefinedCast word → cast word = referenceCast word)
    (rest : Algebra Word32) (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (initial : State Word32) (mpos : 0 < s.m) (npos : 0 < s.n)
    (defined : FP32Domain layout s initial) (bytes : UInt8Domain layout s initial)
    (hout : OutputLayout layout s) (memory : Backing) (same : SameInputs layout initial memory)
    (pid : Nat) :
    ∃ next, FP.Structural.exec (backend cast rest) (source (parameters layout s c))
        (programState initial memory pid) = some next ∧
      flatten next = frozenStep layout s c initial pid memory := by
  have aw : inputWords layout (programState initial memory pid) = inputWords layout initial := by
    funext row k
    exact congrArg (Cell.read .real) (same _ _ (Or.inl rfl))
  have bw : packedValues layout (programState initial memory pid) = packedValues layout initial := by
    funext col q
    exact congrArg (Cell.read .nat) (same _ _ (Or.inr rfl))
  have defined' : FP32Domain layout s (programState initial memory pid) := by
    simpa only [FP32Domain, aw] using defined
  have bytes' : UInt8Domain layout s (programState initial memory pid) := by
    simpa only [UInt8Domain, bw] using bytes
  obtain ⟨next,run,stores,frame⟩ := tile_low32 cast castLeaf rest layout s c legal
    (programState initial memory pid) mpos npos defined' bytes' hout
  refine ⟨next,run,?_⟩
  funext loc
  by_cases written : tileWrites layout s c initial pid loc
  · obtain ⟨i,hi,he⟩ := written
    subst loc
    have coords := Whole001.output_coordinates s
      (outputRow (parameters layout s c) (tileState initial pid) i)
      (outputColumn (parameters layout s c) (tileState initial pid) i) hi.2
    rw [← output_coordinates_source layout s c (tileState initial pid) i hout] at coords
    change next.mem layout.output (outputOffset (parameters layout s c) (programState initial memory pid) i) = _
    rw [stores i hi]
    simp only [frozenStep, Schedule.put, if_pos (show tileWrites layout s c initial pid
      (layout.output,outputOffset (parameters layout s c) (tileState initial pid) i) from ⟨i,hi,rfl⟩),
      denseCell, Prod.snd, coords.1, coords.2]
    unfold int32Cell matrixWordTarget packedWords
    rw [aw,bw]
    rfl
  · rw [show frozenStep layout s c initial pid memory loc = memory loc from if_neg written]
    apply frame
    by_cases hr : loc.1 = layout.output
    · right
      intro i hi ho
      apply written
      refine ⟨i,hi,?_⟩
      exact Prod.ext hr ho
    · exact Or.inl hr

theorem frozen_inputs (layout : Layout) (s : Shape) (c : Fin 15) (initial : State Word32)
    (separate : layout.output ≠ layout.input ∧ layout.output ≠ layout.weight)
    (memory : Backing) (same : SameInputs layout initial memory) (pid : Nat) :
    SameInputs layout initial (frozenStep layout s c initial pid memory) := by
  intro r o hr
  have miss : ¬ tileWrites layout s c initial pid (r,o) := by
    rintro ⟨i,hi,he⟩
    have eq := congrArg Prod.fst he
    rcases hr with ha | hb
    · exact separate.1 (eq.symm.trans ha)
    · exact separate.2 (eq.symm.trans hb)
  exact (if_neg miss).trans (same r o hr)

theorem schedule_execution (cast : Word32 → Byte)
    (castLeaf : ∀ word, DefinedCast word → cast word = referenceCast word)
    (rest : Algebra Word32) (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (initial : State Word32) (mpos : 0 < s.m) (npos : 0 < s.n)
    (defined : FP32Domain layout s initial) (bytes : UInt8Domain layout s initial)
    (hout : OutputLayout layout s) (separate : layout.output ≠ layout.input ∧ layout.output ≠ layout.weight)
    (schedule : List Nat) (memory : Backing) (same : SameInputs layout initial memory) :
    GridExec cast rest (parameters layout s c) initial schedule memory
      (Schedule.run (frozenStep layout s c initial) schedule memory) := by
  induction schedule generalizing memory with
  | nil => exact GridExec.nil memory
  | cons pid tail ih =>
    obtain ⟨next,run,mem⟩ := program_frozen cast castLeaf rest layout s c legal initial mpos npos
      defined bytes hout memory same pid
    apply GridExec.cons next run
    rw [mem]
    exact ih _ (frozen_inputs layout s c initial separate memory same pid)

#axiomsClean program_frozen
#axiomsClean frozen_inputs
#axiomsClean schedule_execution
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
