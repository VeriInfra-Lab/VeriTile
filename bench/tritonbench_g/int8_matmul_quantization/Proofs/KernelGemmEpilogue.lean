import bench.tritonbench_g.int8_matmul_quantization.Proofs.KernelGemmLoop

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelGemm
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {I F H : Type} [Inhabited F]

def rowScaleType : Precision → FloatDType | .half => .fp16 | .single => .real

/-- Independent source-order partial: actual signed accumulation, FP32
conversion, row scale, column scale, and mandatory FP16 narrowing. -/
def partialResult (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (plans : Plans) (rowScale : Precision)
    (p : Parameters) (s : State F) (i : CIndex p) : F :=
  M.cast none .real .fp16 (M.binary (some .fp32) .real .mul
    (M.binary (some .fp32) .real .mul
      (M.fromInt (some .fp32) (accumulation carrier ops plans p s p.steps i))
      (M.cast (some .fp32) (rowScaleType rowScale) .real (aScale M rowScale p s i.1)))
    (bScale M p s i.2.1))

/-- FP32 destinations widen the already narrowed FP16 partial. -/
def storedValue (M : Algebra F) : Precision → F → F
  | .half, x => M.cast none .fp16 .fp16 x
  | .single, x => M.cast (some .fp32) .fp16 .real x
def outputOffset (p : Parameters) (s : State F) (i : CIndex p) :=
  p.outputRowStride*(rowBlock p s*p.blockRows+i.1.val)+
    p.outputColumnStride*(columnBlock p s*p.blockColumns+i.2.1.val)
def active (p : Parameters) (s : State F) (i : CIndex p) : Prop :=
  rowBlock p s*p.blockRows+i.1.val < p.rows ∧
    columnBlock p s*p.blockColumns+i.2.1.val < p.columns
instance (p : Parameters) (s : State F) : DecidablePred (active p s) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))

def epilogueExecution (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (plans : Plans)
    (atomicHalf atomicSingle : F → F → F)
    (half : M.atomicAdd .fp16 = some atomicHalf) (single : M.atomicAdd .real = some atomicSingle)
    (rowScale output : Precision) (p : Parameters) (initial current : State F)
    (h : Invariant M carrier ops plans rowScale p initial p.steps current) :
    {t : State F // KernelExecution.run M carrier ops plans (epilogue rowScale output p) current = some t} := by
  cases rowScale <;> cases output <;> by_cases hs : p.split = 1
  all_goals (refine ⟨?_, ?_⟩; swap
             simp [epilogue, source, KernelSource.gemmHalfHalf, KernelSource.gemmHalfSingle,
    KernelSource.gemmSingleHalf, KernelSource.gemmSingleSingle, ComputeKernel.surfaceBody,
    KernelExecution.run, KernelExecution.step, localStep, FP.Structural.step,
    evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, atomicAdd, atomicAddition,
    numeric, FP.Structural.bop, natLt, numericEq, ComputeDType.eraseDType,
    ofFloat, toFloat, FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, h.pidm, h.pidn, h.acc, h.ascale, h.bscale,
    h.memory, half, single, hs]
             rfl)

/-- The masked source store or atomic read-modify-write, retaining each
actual per-lane old output. No zero-output or associativity premise occurs. -/
def written (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (plans : Plans)
    (atomicHalf atomicSingle : F → F → F) (rowScale output : Precision)
    (p : Parameters) (initial : State F) : State F :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl (fun t i =>
    if active p initial i then
      let value := storedValue M output (partialResult M carrier ops plans rowScale p initial i)
      let offset := outputOffset p initial i
      match output with
      | .half => t.write p.output offset (.mk .fp16
          (if p.split = 1 then value else atomicHalf ((t.mem p.output offset).read .fp16) value))
      | .single => t.write p.output offset (.mk .real
          (if p.split = 1 then value else atomicSingle ((t.mem p.output offset).read .real) value))
    else t) initial

private theorem fold_memory {α ι : Type} (xs : List ι) (r : RegionName)
    (off : ι → Nat) (value : (RegionName → Nat → Cell α) → ι → Cell α)
    (active : ι → Prop) [DecidablePred active] (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (value t.mem i) else t) s).mem =
      xs.foldl (fun mem i => fun r' o =>
        if active i ∧ r' = r ∧ o = off i then value mem i else mem r' o) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih =>
    by_cases hi : active i
    · simp only [List.foldl_cons, if_pos hi, ih]
      congr 1
      funext r' o
      simp [hi, State.write]
    · simp only [List.foldl_cons, if_neg hi, ih]
      congr 1
      funext r' o
      simp [hi]

private theorem fold_expanded {α ι : Type} (xs : List ι) (r : RegionName)
    (off : ι → Nat) (value : (RegionName → Nat → Cell α) → ι → Cell α)
    (active : ι → Prop) [DecidablePred active] (s : State α) :
    (xs.foldl (fun t i => if active i then
      { t with mem := fun r' o => if r' = r ∧ o = off i then value t.mem i else t.mem r' o }
      else t) s).mem =
    xs.foldl (fun mem i => fun r' o => if active i ∧ r' = r ∧ o = off i then value mem i else mem r' o) s.mem :=
  fold_memory xs r off value active s

private theorem foldl_mem_congr {α ι : Type} (xs : List ι)
    (f g : State α → ι → State α) (s t : State α) (initial : s.mem = t.mem)
    (step : ∀ a b, a.mem = b.mem → ∀ i, (f a i).mem = (g b i).mem) :
    (xs.foldl f s).mem = (xs.foldl g t).mem := by
  induction xs generalizing s t with
  | nil => exact initial
  | cons i xs ih => exact ih _ _ (step _ _ initial i)

private theorem index_eta {m n : Nat} (i : TileIndex [m,n]) :
    (i.1,i.2.1,PUnit.unit) = i := by
  rcases i with ⟨r,c,u⟩
  cases u
  rfl

theorem epilogueMemory (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (plans : Plans)
    (atomicHalf atomicSingle : F → F → F)
    (half : M.atomicAdd .fp16 = some atomicHalf) (single : M.atomicAdd .real = some atomicSingle)
    (rowScale output : Precision) (p : Parameters) (initial current : State F)
    (h : Invariant M carrier ops plans rowScale p initial p.steps current) :
    (epilogueExecution M carrier ops plans atomicHalf atomicSingle half single rowScale output p initial current h).val.mem =
      (written M carrier ops plans atomicHalf atomicSingle rowScale output p initial).mem := by
  cases rowScale <;> cases output <;> by_cases hs : p.split = 1
  all_goals simp +contextual (config := { maxSteps := 1000000 }) [epilogueExecution,
    written, partialResult, storedValue, rowScaleType, active, outputOffset,
    fold_memory, fold_expanded, State.write, hs, h.memory, index_eta,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    and_assoc, and_comm, and_left_comm]
  all_goals try rfl
  all_goals apply foldl_mem_congr
  all_goals try simpa using h.memory
  all_goals
    intro a b same i
    simp +contextual [apply_ite, same]

/-- Success and output-memory formula for the entire actual quotation,
including its K loop and the selected store/atomic branch. -/
theorem source_execution (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (plans : Plans)
    (atomicHalf atomicSingle : F → F → F)
    (half : M.atomicAdd .fp16 = some atomicHalf) (single : M.atomicAdd .real = some atomicSingle)
    (rowScale output : Precision) (p : Parameters) (initial : State F) :
    ∃ final, KernelExecution.exec M carrier ops plans (source rowScale output p) initial = some final ∧
      final.mem = (written M carrier ops plans atomicHalf atomicSingle rowScale output p initial).mem := by
  let start := preludeExecution M carrier ops plans rowScale output p initial
  obtain ⟨mid, hm, hi⟩ := loopExecution M carrier ops plans rowScale p initial start.val 0 (by omega)
    (preludeInvariant M carrier ops plans rowScale output p initial)
  let finish := epilogueExecution M carrier ops plans atomicHalf atomicSingle half single rowScale output p initial mid hi
  refine ⟨finish.val, ?_, epilogueMemory M carrier ops plans atomicHalf atomicSingle half single rowScale output p initial mid hi⟩
  change KernelExecution.run M carrier ops plans (source rowScale output p).surfaceBody initial = some finish.val
  rw [source_decomposition, KernelExecution.run_append, KernelExecution.run_append, start.property]
  have hsingle (s : State F) :
      KernelExecution.run M carrier ops plans [.forRange "k" 0 p.steps 1 (body p)] s =
        KernelExecution.range M carrier ops plans "k" 0 p.steps 1 (body p) s := by
    simp [KernelExecution.run, KernelExecution.step]
  simp only [Option.bind_some, hsingle, hm]
  exact finish.property

#axiomsClean epilogueExecution
#axiomsClean epilogueMemory
#axiomsClean source_execution
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelGemm
