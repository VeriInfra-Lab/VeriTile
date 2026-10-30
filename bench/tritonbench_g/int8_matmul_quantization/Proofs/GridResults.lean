import bench.tritonbench_g.int8_matmul_quantization.Proofs.Grid
import bench.tritonbench_g.int8_matmul_quantization.Proofs.NativeInteger

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Grid
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false
variable {I F H : Type} [Inhabited F]

def zeroCell (output : Precision) (zero : F) : FP.Structural.Cell F :=
  match output with | .half => .mk .fp16 zero | .single => .mk .real zero

def reset (s : Shape) (zero : FP.Structural.Cell F) (state : State F) : State F :=
  { state with mem := fun region offset => if region = "C" ∧ offset < s.M*s.N then zero else state.mem region offset }

theorem reset_cell (s : Shape) (zero : FP.Structural.Cell F) (state : State F) (cell : MatrixCell s) :
    (reset s zero state).mem "C" (outputOffset s cell) = zero := by
  have h := RowwiseQuantizationTriton.linear_bound s.M s.N cell.1 cell.2 cell.1.isLt cell.2.isLt
  simp [reset, outputOffset, h]

theorem reset_frame (s : Shape) (zero : FP.Structural.Cell F) (state : State F) (region : RegionName) (offset : Nat)
    (outside : region ≠ "C" ∨ s.M*s.N ≤ offset) :
    (reset s zero state).mem region offset = state.mem region offset := by
  rcases outside with h | h
  · simp [reset, h]
  · simp [reset, show ¬offset < s.M*s.N by omega]

theorem reads_of_frame (s : Shape) (c : Config) (columnMajor : Bool) (rowScale : Precision)
    (input : Gemm.Input F) (initial current : State F)
    (reads : Gemm.Reads (Gemm.parameters s c columnMajor) rowScale input initial)
    (frame : ∀ (region : RegionName) offset, region ≠ "C" → current.mem region offset = initial.mem region offset) :
    Gemm.Reads (Gemm.parameters s c columnMajor) rowScale input current := by
  constructor
  · intro r k hr hk
    change (current.mem "A" _).read .int = _
    rw [frame _ _ (by decide)]
    exact reads.A r k hr hk
  · intro k n hk hn
    change (current.mem "B" _).read .int = _
    rw [frame _ _ (by decide)]
    exact reads.B k n hk hn
  · intro r hr
    have h := reads.aScale r hr
    cases rowScale <;> simp only [Gemm.parameters] at h ⊢
    all_goals rw [frame "AS" _ (by decide)]; exact h
  · intro n hn
    change (current.mem "BS" _).read .real = _
    rw [frame _ _ (by decide)]
    exact reads.bScale n hn

def launch (s : Shape) (c : Config)
    (act : Int8MatmulQuantization.Commit s c → FP.Structural.Cell F → FP.Structural.Cell F)
    (q : Schedule s c) (zero : FP.Structural.Cell F) (state : State F) :=
  run s c act q.events (reset s zero state)

theorem launch_cell (s : Shape) (c : Config)
    (act : Int8MatmulQuantization.Commit s c → FP.Structural.Cell F → FP.Structural.Cell F)
    (q : Schedule s c) (zero : FP.Structural.Cell F) (state : State F) (cell : MatrixCell s) :
    (launch s c act q zero state).mem "C" (outputOffset s cell) =
      (projection cell q.events).foldl (fun old z => act (cell,z) old) zero := by
  rw [launch, cell_order, reset_cell]

theorem launch_frame (s : Shape) (c : Config)
    (act : Int8MatmulQuantization.Commit s c → FP.Structural.Cell F → FP.Structural.Cell F)
    (q : Schedule s c) (zero : FP.Structural.Cell F) (state : State F) (region : RegionName) (offset : Nat)
    (outside : region ≠ "C" ∨ s.M*s.N ≤ offset) :
    (launch s c act q zero state).mem region offset = state.mem region offset := by
  rw [launch, frame _ _ _ _ _ _ _ outside, reset_frame _ _ _ _ _ outside]

/-- At every position of the actual interleaving, its event is the effect
of the full readable source on that event's current old output cell. -/
theorem source_at_prefix (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (atomicHalf atomicSingle : F → F → F)
    (half : M.atomicAdd .fp16 = some atomicHalf) (single : M.atomicAdd .real = some atomicSingle)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (input : Gemm.Input F)
    (initial : State F) (reads : Gemm.Reads (Gemm.parameters s c columnMajor) rowScale input initial)
    (zero : FP.Structural.Cell F) (before : List (Int8MatmulQuantization.Commit s c))
    (e : Int8MatmulQuantization.Commit s c) :
    let act := action M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor input
    let current := run s c act before (reset s zero initial)
    ∃ physical : PhysicalCommit s c, ∃ final,
      physical.logical = e ∧
      KernelExecution.exec M carrier ops plans (Gemm.source rowScale output (Gemm.parameters s c columnMajor))
        (atProgram current physical.pid physical.split) = some final ∧
      final.mem "C" (outputOffset s e.1) = (advance s c act current e).mem "C" (outputOffset s e.1) := by
  dsimp only
  have currentReads := reads_of_frame s c columnMajor rowScale input initial
    (run s c (action M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor input)
      before (reset s zero initial)) reads (by
        intro region offset different
        rw [frame _ _ _ _ _ _ _ (Or.inl different), reset_frame _ _ _ _ _ (Or.inl different)])
  obtain ⟨physical, final, he, hx, hv⟩ := source_event M carrier ops plans atomicHalf atomicSingle half single
    rowScale output s c columnMajor input _ currentReads e
  exact ⟨physical, final, he, hx, by simpa [advance, State.write] using hv⟩

def realAction (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (plans : Plans)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (input : Gemm.Input ℝ) :=
  action (KernelScalarProfile.mathematical tree rest) mathematicalIntegers KernelIntegerIdeal.operations.toOps
    plans (·+·) (·+·) rowScale output s c columnMajor input

theorem real_action (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (plans : Plans)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (input : Gemm.Input ℝ)
    (e : Int8MatmulQuantization.Commit s c) (old : ℝ) :
    realAction tree rest plans rowScale output s c columnMajor input e (zeroCell output old) =
      zeroCell output (if c.S=1 then realPartial s c (Gemm.realInput s input) e.1.1 e.1.2 e.2
        else old+realPartial s c (Gemm.realInput s input) e.1.1 e.1.2 e.2) := by
  unfold realAction action
  rw [Gemm.partial_real]
  cases output <;> simp [KernelGemm.storedValue, KernelScalarProfile.mathematical, zeroCell]

theorem real_fold (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (plans : Plans)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (input : Gemm.Input ℝ)
    (cell : MatrixCell s) (zs : List (Fin c.S)) (old : ℝ) :
    zs.foldl (fun old z => realAction tree rest plans rowScale output s c columnMajor input (cell,z) old)
      (zeroCell output old) = zeroCell output
      (zs.foldl (fun old (z : Fin c.S) => if c.S=1 then realPartial s c (Gemm.realInput s input) cell.1 cell.2 z
        else old+realPartial s c (Gemm.realInput s input) cell.1 cell.2 z) old) := by
  induction zs generalizing old with
  | nil => rfl
  | cons z zs ih => rw [List.foldl_cons, real_action, ih]; rfl

/-- Complete ideal grid result. Schedule permutations are used only in this
Real interpretation, after the full source-to-event connection above. -/
theorem real_complete (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (plans : Plans)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (input : Gemm.Input ℝ)
    (q : Schedule s c) (state : State ℝ) (cell : MatrixCell s) :
    (launch s c (realAction tree rest plans rowScale output s c columnMajor input) q (zeroCell output 0) state).mem
      "C" (outputOffset s cell) = zeroCell output (target s (Gemm.realInput s input) cell.1 cell.2) := by
  rw [launch_cell, real_fold]
  apply congrArg (zeroCell output)
  let f : Fin c.S → ℝ := fun z => realPartial s c (Gemm.realInput s input) cell.1 cell.2 z
  have perm := ((q.complete cell).map f).sum_eq
  by_cases single : c.S=1
  · have len : (projection cell q.events).length=1 := by simpa [single] using (q.complete cell).length_eq
    obtain ⟨z,hz⟩ : ∃ z,projection cell q.events=[z] := by
      cases h : projection cell q.events with
      | nil => simp [h] at len
      | cons z zs => cases zs with
        | nil => exact ⟨z,rfl⟩
        | cons z' zs => simp [h] at len
    rw [hz,List.foldl_cons,List.foldl_nil,if_pos single]
    have hz0 : z.val=0 := by have h:=z.isLt; omega
    simpa [single,Int8DequantMatmul.rsum,hz0] using partials_target s c (Gemm.realInput s input) cell.1 cell.2
  · simp only [if_neg single]
    rw [Int8DequantMatmul.fold_add, zero_add, perm]
    change ((List.finRange c.S).map (fun z => realPartial s c (Gemm.realInput s input) cell.1 cell.2 z.val)).sum = _
    rw [Int8DequantMatmul.finRange_rsum]
    exact partials_target _ _ _ _ _

local instance : Inhabited Word32 := ⟨FP.Value.zero .fp32⟩
def nativeAction (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (plans : Plans) (rowScale output : Precision) (s : Shape) (c : Config)
    (columnMajor : Bool) (input : Gemm.Input Word32) :=
  action (KernelScalarProfile.native f tree rest) signed32 (KernelNativeModel.operations f).toOps plans
    (fun x y => KernelScalarProfile.pack16 (f.atomic16 (KernelScalarProfile.unpack16 x) (KernelScalarProfile.unpack16 y)))
    f.atomic32 rowScale output s c columnMajor input

/-- Exact native result preserves every rounding, stored half, widening,
and each cell's projected typed atomic order. -/
def nativeWord (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (plans : Plans) (rowScale output : Precision) (s : Shape) (c : Config)
    (columnMajor : Bool) (input : Gemm.Input Word32) (q : Schedule s c) (cell : MatrixCell s) :=
  (projection cell q.events).foldl
    (fun old z => nativeAction f tree rest plans rowScale output s c columnMajor input (cell,z) old)
    (zeroCell output (FP.Value.zero .fp32))

theorem native_complete (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (plans : Plans) (rowScale output : Precision) (s : Shape) (c : Config)
    (columnMajor : Bool) (input : Gemm.Input Word32) (q : Schedule s c) (state : State Word32) (cell : MatrixCell s) :
    (launch s c (nativeAction f tree rest plans rowScale output s c columnMajor input) q
      (zeroCell output (FP.Value.zero .fp32)) state).mem "C" (outputOffset s cell) =
      nativeWord f tree rest plans rowScale output s c columnMajor input q cell :=
  launch_cell _ _ _ _ _ _ _

#axiomsClean source_at_prefix
#axiomsClean real_complete
#axiomsClean native_complete
#axiomsClean launch_frame
#specNonCircular nativeWord avoiding [Gemm.source, KernelExecution.exec, launch, run, advance]
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Grid
