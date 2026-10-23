import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.GridResults

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Host
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSectionVars false
variable {I F H : Type} [Inhabited F]

structure Invocation (s : Shape) where
  config : Config
  admitted : Int8MatmulQuantization.Source.Admitted config
  plans : Plans
  schedule : Schedule s config
  dispatch : schedule.events.Perm (allCommits s config)

def act (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (atomicHalf atomicSingle : F → F → F) (rowScale output : Precision) (s : Shape)
    (columnMajor : Bool) (input : Gemm.Input F) (v : Invocation s) :=
  Grid.action M carrier ops v.plans atomicHalf atomicSingle rowScale output s v.config columnMajor input

def invoke (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (atomicHalf atomicSingle : F → F → F) (rowScale output : Precision) (s : Shape)
    (columnMajor : Bool) (input : Gemm.Input F) (zero : FP.Structural.Cell F) (v : Invocation s) (state : State F) :=
  Grid.launch s v.config (act M carrier ops atomicHalf atomicSingle rowScale output s columnMajor input v) v.schedule zero state

def trials (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (atomicHalf atomicSingle : F → F → F) (rowScale output : Precision) (s : Shape)
    (columnMajor : Bool) (input : Gemm.Input F) (zero : FP.Structural.Cell F) : List (Invocation s) → State F → State F
  | [],state => state
  | v::vs,state => trials M carrier ops atomicHalf atomicSingle rowScale output s columnMajor input zero vs
      (invoke M carrier ops atomicHalf atomicSingle rowScale output s columnMajor input zero v state)

theorem trials_frame (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (atomicHalf atomicSingle : F → F → F) (rowScale output : Precision) (s : Shape)
    (columnMajor : Bool) (input : Gemm.Input F) (zero : FP.Structural.Cell F)
    (vs : List (Invocation s)) (state : State F) (region : RegionName) (offset : Nat)
    (outside : region ≠ "C" ∨ s.M*s.N ≤ offset) :
    (trials M carrier ops atomicHalf atomicSingle rowScale output s columnMajor input zero vs state).mem region offset =
      state.mem region offset := by
  induction vs generalizing state with
  | nil => rfl
  | cons v vs ih =>
    rw [trials, ih]
    exact Grid.launch_frame _ _ _ _ _ _ _ _ outside

/-- Literal wrapper/autotuner lifecycle: initial zero; each benchmark's
reset; post-benchmark reset on a cold run; final source grid. -/
def runtime (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (atomicHalf atomicSingle : F → F → F) (rowScale output : Precision) (s : Shape)
    (columnMajor : Bool) (input : Gemm.Input F) (zero : FP.Structural.Cell F)
    (vs : List (Invocation s)) (chosen : Invocation s) (state : State F) : State F :=
  let initial := Grid.reset s zero state
  let done := trials M carrier ops atomicHalf atomicSingle rowScale output s columnMajor input zero vs initial
  let ready := if vs=[] then done else Grid.reset s zero done
  Grid.run s chosen.config (act M carrier ops atomicHalf atomicSingle rowScale output s columnMajor input chosen)
    chosen.schedule.events ready

theorem runtime_cell (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (atomicHalf atomicSingle : F → F → F) (rowScale output : Precision) (s : Shape)
    (columnMajor : Bool) (input : Gemm.Input F) (zero : FP.Structural.Cell F)
    (vs : List (Invocation s)) (chosen : Invocation s) (state : State F) (cell : MatrixCell s) :
    (runtime M carrier ops atomicHalf atomicSingle rowScale output s columnMajor input zero vs chosen state).mem
      "C" (Grid.outputOffset s cell) =
      (projection cell chosen.schedule.events).foldl
        (fun old z => act M carrier ops atomicHalf atomicSingle rowScale output s columnMajor input chosen (cell,z) old) zero := by
  unfold runtime
  rw [Grid.cell_order]
  cases vs with
  | nil => simp only [trials, ite_true, Grid.reset_cell]
  | cons v vs => simp only [reduceCtorEq, ite_false, Grid.reset_cell]

theorem runtime_frame (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (atomicHalf atomicSingle : F → F → F) (rowScale output : Precision) (s : Shape)
    (columnMajor : Bool) (input : Gemm.Input F) (zero : FP.Structural.Cell F)
    (vs : List (Invocation s)) (chosen : Invocation s) (state : State F) (region : RegionName) (offset : Nat)
    (outside : region ≠ "C" ∨ s.M*s.N ≤ offset) :
    (runtime M carrier ops atomicHalf atomicSingle rowScale output s columnMajor input zero vs chosen state).mem region offset =
      state.mem region offset := by
  unfold runtime
  rw [Grid.frame _ _ _ _ _ _ _ outside]
  split_ifs
  all_goals try rw [Grid.reset_frame _ _ _ _ _ outside]
  all_goals rw [trials_frame _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ outside, Grid.reset_frame _ _ _ _ _ outside]

theorem empty_events (s : Shape) (v : Invocation s) (empty : s.M=0 ∨ s.N=0) : v.schedule.events=[] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro e _
  have hm:=e.1.1.isLt
  have hn:=e.1.2.isLt
  rcases empty with h|h <;> omega

theorem configs_preserved : Int8MatmulQuantization.Source.configs.length=34 ∧
    Int8MatmulQuantization.Source.quantizerWarpsStages.length=4 := ⟨rfl,rfl⟩

#axiomsClean runtime_cell
#axiomsClean runtime_frame
#axiomsClean empty_events
#axiomsClean configs_preserved
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Host
