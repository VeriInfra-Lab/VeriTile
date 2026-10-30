import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.Commit
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.GemmIdeal
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Coverage

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Grid
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSectionVars false
set_option linter.unusedSimpArgs false
variable {I F H : Type} [Inhabited F]

def atProgram (initial : State F) (pid split : Nat) : State F :=
  { initial with pids := fun ax => if ax = 0 then pid else if ax = 1 then split else initial.pids ax }

theorem reads_program (s : Shape) (c : Config) (columnMajor : Bool) (rowScale : Precision)
    (input : Gemm.Input F) (initial : State F)
    (reads : Gemm.Reads (Gemm.parameters s c columnMajor) rowScale input initial) (pid split : Nat) :
    Gemm.Reads (Gemm.parameters s c columnMajor) rowScale input (atProgram initial pid split) :=
  ⟨reads.A, reads.B, reads.aScale, reads.bScale⟩

theorem physical_row (s : Shape) (c : Config) (columnMajor : Bool) (initial : State F) (e : PhysicalCommit s c) :
    KernelGemm.rowBlock (Gemm.parameters s c columnMajor) (atProgram initial e.pid e.split)*c.BM+e.lane.1.val =
      e.logical.1.1.val := rfl
theorem physical_column (s : Shape) (c : Config) (columnMajor : Bool) (initial : State F) (e : PhysicalCommit s c) :
    KernelGemm.columnBlock (Gemm.parameters s c columnMajor) (atProgram initial e.pid e.split)*c.BN+e.lane.2.1.val =
      e.logical.1.2.val := rfl

theorem physical_active (s : Shape) (c : Config) (columnMajor : Bool) (initial : State F) (e : PhysicalCommit s c) :
    KernelGemm.active (Gemm.parameters s c columnMajor) (atProgram initial e.pid e.split) e.lane := e.active

def outputOffset (s : Shape) (cell : MatrixCell s) : Nat := cell.1.val*s.N+cell.2.val

theorem physical_offset (s : Shape) (c : Config) (columnMajor : Bool) (initial : State F) (e : PhysicalCommit s c) :
    KernelGemm.outputOffset (Gemm.parameters s c columnMajor) (atProgram initial e.pid e.split) e.lane =
      outputOffset s e.logical.1 := by
  have positive : 1 ≤ s.N := by have := e.logical.1.2.isLt; omega
  change max s.N 1*(KernelGemm.rowBlock (Gemm.parameters s c columnMajor) (atProgram initial e.pid e.split)*c.BM+e.lane.1.val) +
    1*(KernelGemm.columnBlock (Gemm.parameters s c columnMajor) (atProgram initial e.pid e.split)*c.BN+e.lane.2.1.val) = _
  rw [physical_row, physical_column, max_eq_left positive, Nat.one_mul, Nat.mul_comm s.N]
  rfl

/-- One independently specified source-ordered contribution at one logical
cell/split, applied to exactly its current output cell. -/
def action (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (atomicHalf atomicSingle : F → F → F) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (input : Gemm.Input F)
    (e : Int8MatmulQuantization.Commit s c) (old : FP.Structural.Cell F) : FP.Structural.Cell F :=
  let value := KernelGemm.storedValue M output (Gemm.partialWord M carrier ops plans rowScale
    (Gemm.parameters s c columnMajor) input e.1.1 e.1.2 e.2)
  match output with
  | .half => .mk .fp16 (if c.S = 1 then value else atomicHalf (old.read .fp16) value)
  | .single => .mk .real (if c.S = 1 then value else atomicSingle (old.read .real) value)

/-- Every logical event is realized by the full readable source at its
unique physical owner/lane. The arithmetic and old output are never premises. -/
theorem source_event (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (atomicHalf atomicSingle : F → F → F)
    (half : M.atomicAdd .fp16 = some atomicHalf) (single : M.atomicAdd .real = some atomicSingle)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (input : Gemm.Input F)
    (initial : State F) (reads : Gemm.Reads (Gemm.parameters s c columnMajor) rowScale input initial)
    (e : Int8MatmulQuantization.Commit s c) :
    ∃ physical : PhysicalCommit s c, ∃ final,
      physical.logical = e ∧
      KernelExecution.exec M carrier ops plans (Gemm.source rowScale output (Gemm.parameters s c columnMajor))
        (atProgram initial physical.pid physical.split) = some final ∧
      final.mem "C" (outputOffset s e.1) =
        action M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor input e
          (initial.mem "C" (outputOffset s e.1)) := by
  obtain ⟨physical, logical⟩ := physical_covers s c e
  have positive : 1 ≤ s.N := by have := e.1.2.isLt; omega
  obtain ⟨final, he, hv, _⟩ := Gemm.program_complete M carrier ops plans atomicHalf atomicSingle half single
    rowScale output (Gemm.parameters s c columnMajor) input (atProgram initial physical.pid physical.split)
    (reads_program s c columnMajor rowScale input initial reads _ _) (max_eq_left positive) rfl
  refine ⟨physical, final, logical, he, ?_⟩
  have value := hv physical.lane (physical_active s c columnMajor initial physical)
  rw [physical_offset, logical] at value
  convert value using 1
  unfold Gemm.action action
  simp only [Gemm.parameters, atProgram, ite_true, show (1:Nat) ≠ 0 by decide, ite_false]
  have hr := physical_row s c columnMajor initial physical
  have hc := physical_column s c columnMajor initial physical
  rw [logical] at hr hc
  simp only [Gemm.parameters, atProgram] at hr hc
  rw [hr, hc]
  have hz : physical.split = e.2 := congrArg Prod.snd logical
  rw [hz]
  cases output <;> rfl

def advance (s : Shape) (c : Config)
    (action : Int8MatmulQuantization.Commit s c → FP.Structural.Cell F → FP.Structural.Cell F)
    (state : State F) (e : Int8MatmulQuantization.Commit s c) : State F :=
  state.write "C" (outputOffset s e.1) (action e (state.mem "C" (outputOffset s e.1)))

def run (s : Shape) (c : Config)
    (action : Int8MatmulQuantization.Commit s c → FP.Structural.Cell F → FP.Structural.Cell F)
    (events : List (Int8MatmulQuantization.Commit s c)) (state : State F) : State F :=
  events.foldl (advance s c action) state

theorem output_injective (s : Shape) : Function.Injective (outputOffset s) := by
  intro x y h
  obtain ⟨hr,hc⟩ := RowwiseQuantizationTriton.linear_injective s.N _ _ _ _ x.2.isLt y.2.isLt h
  exact Prod.ext (Fin.ext hr) (Fin.ext hc)

theorem frame (s : Shape) (c : Config)
    (action : Int8MatmulQuantization.Commit s c → FP.Structural.Cell F → FP.Structural.Cell F)
    (events : List (Int8MatmulQuantization.Commit s c)) (state : State F) (region : RegionName) (offset : Nat)
    (outside : region ≠ "C" ∨ s.M*s.N ≤ offset) :
    (run s c action events state).mem region offset = state.mem region offset := by
  induction events generalizing state with
  | nil => rfl
  | cons e events ih =>
    change (run s c action events (advance s c action state e)).mem region offset = _
    rw [ih]
    apply State.write_other
    rcases outside with hr | ho
    · exact Or.inl hr
    · right
      have bound := RowwiseQuantizationTriton.linear_bound s.M s.N e.1.1 e.1.2 e.1.1.isLt e.1.2.isLt
      unfold outputOffset
      omega

/-- Per-cell order is the projection of the actual selected interleaving.
No commutation or associativity assumption appears in this fold theorem. -/
theorem cell_order (s : Shape) (c : Config)
    (action : Int8MatmulQuantization.Commit s c → FP.Structural.Cell F → FP.Structural.Cell F)
    (events : List (Int8MatmulQuantization.Commit s c)) (state : State F) (cell : MatrixCell s) :
    (run s c action events state).mem "C" (outputOffset s cell) =
      (projection cell events).foldl (fun old z => action (cell,z) old) (state.mem "C" (outputOffset s cell)) := by
  induction events generalizing state with
  | nil => rfl
  | cons e events ih =>
    change (run s c action events (advance s c action state e)).mem "C" (outputOffset s cell) = _
    rw [ih]
    by_cases same : e.1 = cell
    · rcases e with ⟨x,z⟩
      dsimp only at same
      subst x
      simp [projection, advance, State.write]
    · have different : outputOffset s cell ≠ outputOffset s e.1 := fun h => same ((output_injective s h).symm)
      simp [projection, same, advance, State.write, different]

#axiomsClean physical_active
#axiomsClean physical_offset
#axiomsClean source_event
#axiomsClean frame
#axiomsClean cell_order
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Grid
