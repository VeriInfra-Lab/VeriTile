import bench.tritonbench_g.int8_matmul_quantization.Proofs.GemmIOSafetyComplete
import bench.tritonbench_g.int8_matmul_quantization.Proofs.GemmIOTrace

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Gemm
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 28000
variable {F I H : Type} [Inhabited F]

theorem programs_safe (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (atomicHalf atomicSingle : F→F→F)
    (half : M.atomicAdd .fp16=some atomicHalf) (single : M.atomicAdd .real=some atomicSingle)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (q : Schedule s c)
    (xs : Inputs F s rowScale) (base : State F) (signed : SignedInputs s rowScale xs)
    (binding : (layout F s rowScale output columnMajor).InputCells id xs base) :
    (io M carrier ops zero plans rowScale output s c columnMajor q).Programs xs base := by
  intro current event _
  constructor
  · intro bounds bounded
    exact GemmSafety.invocation_safe M carrier ops zero plans rowScale output s c columnMajor event xs bounds bounded _
  · exact source_result M carrier ops zero plans atomicHalf atomicSingle half single rowScale output s c columnMajor
      event xs _ signed (patch_inputs s rowScale output columnMajor xs current base binding)

/-- The traced finite result is the exact old reset/interleaved launch result. -/
theorem finalValues_launch (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (atomicHalf atomicSingle : F→F→F) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (q : Schedule s c) (xs : Inputs F s rowScale)
    (initialZero : F) (base : State F) (cell : MatrixCell s) :
    (Grid.launch s c (Grid.action M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor
      (input default s rowScale xs)) q (Grid.zeroCell output initialZero) base).mem "C" (Grid.outputOffset s cell)=
      FP.Structural.Cell.mk (dtype output)
        (finalValues M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor q xs initialZero
          ⟨0,(cell.1,cell.2,())⟩) := by
  rw [Grid.launch_cell]
  have h:=fold_typed M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor xs cell
    (projection cell q.events) (value output initialZero)
  cases output <;> exact h

/-- Composition of already-proved actual source safety/execution and the
independent result theorem. All obligations remain conclusions of shared IO. -/
theorem implements_of_result (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (atomicHalf atomicSingle : F→F→F)
    (half : M.atomicAdd .fp16=some atomicHalf) (single : M.atomicAdd .real=some atomicSingle)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (q : Schedule s c)
    (initialZero : F) (formula : Inputs F s rowScale→Outputs F s output)
    (result : ∀ xs : Inputs F s rowScale,∀ base : State F,∀ cell : MatrixCell s,
      (Grid.launch s c (Grid.action M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor
        (input default s rowScale xs)) q (Grid.zeroCell output initialZero) base).mem "C" (Grid.outputOffset s cell)=
        FP.Structural.Cell.mk (dtype output) (formula xs 0 (cell.1,cell.2,()))) :
    (io M carrier ops zero plans rowScale output s c columnMajor q).ImplementsResetOnCells
      (SignedInputs s rowScale) (initialValues initialZero s output) formula := by
  intro xs signed base binding
  let start:=(layout F s rowScale output columnMajor).patchOutputs xs (initialValues initialZero s output) base
  have startBinding:=patch_inputs s rowScale output columnMajor xs (initialValues initialZero s output) base binding
  have programs:=programs_safe M carrier ops zero plans atomicHalf atomicSingle half single rowScale output s c columnMajor
    q xs start signed startBinding
  refine ⟨startBinding,(layout F s rowScale output columnMajor).patchOutputs_cells xs
    (output_unique F s rowScale output columnMajor xs) _ base,
    (layout F s rowScale output columnMajor).patchOutputs_frame xs _ base,
    schedule_complete M carrier ops zero plans rowScale output s c columnMajor q xs,programs,
    source_locality M carrier ops zero plans atomicHalf atomicSingle half single rowScale output s c columnMajor q xs start signed startBinding,
    source_coverage M carrier ops zero plans atomicHalf atomicSingle half single rowScale output s c columnMajor q xs start signed startBinding,
    finalValues M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor q xs initialZero,?_,?_,?_⟩
  · intro key _
    rcases key with ⟨port,j⟩
    have hp : port=0:=Fin.eq_zero port
    subst port
    obtain ⟨row,column,u⟩:=j
    cases u
    exact trace_fold M carrier ops zero plans atomicHalf atomicSingle half single rowScale output s c columnMajor
      q xs start signed startBinding programs (initialValues initialZero s output) (row,column)
      (projection (row,column) q.events) (value output initialZero)
  · intro port j _
    have hp : port=0:=Fin.eq_zero port
    subst port
    obtain ⟨row,column,u⟩:=j
    cases u
    have h:=(finalValues_launch M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor q xs initialZero base (row,column)).symm.trans
      (result xs base (row,column))
    change mapCell id (((layout F s rowScale output columnMajor).patchOutputs xs _ start).mem "C"
      (Grid.outputOffset s (row,column)))=FP.Structural.Cell.mk (dtype output) (formula xs 0 (row,column,()))
    rw [patch_cell]
    cases output <;> exact h
  · intro r off outside
    rw [(layout F s rowScale output columnMajor).patchOutputs_frame xs _ start r off outside]
    exact (layout F s rowScale output columnMajor).patchOutputs_frame xs _ base r off outside

/-- Complete Real scaled integer matrix product, including original reset,
arbitrary per-address commit order, full source safety, output and frame. -/
theorem real_implements (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (plans : Plans)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (q : Schedule s c) :
    (realIO tree rest plans rowScale output s c columnMajor q).ImplementsResetOnCells
      (SignedInputs s rowScale) (initialValues (0:ℝ) s output) (mathematical s rowScale output) := by
  apply implements_of_result _ _ _ _ _ (·+·) (·+·) rfl rfl
  intro xs base cell
  have h:=Grid.real_complete tree rest plans rowScale output s c columnMajor (input 0 s rowScale xs) q base cell
  cases output <;> exact h

local instance : Inhabited Word32:=⟨FP.Value.zero .fp32⟩

/-- Exact native observation with every original rounding and per-cell atomic
order. It adds no native=Real identity or experimental numerical assumption. -/
theorem native_implements (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (plans : Plans) (rowScale output : Precision) (s : Shape) (c : Config)
    (columnMajor : Bool) (q : Schedule s c) :
    (nativeIO f tree rest plans rowScale output s c columnMajor q).ImplementsResetOnCells
      (SignedInputs s rowScale) (initialValues (FP.Value.zero .fp32) s output)
      (nativeFormula f tree rest plans rowScale output s c columnMajor q) := by
  apply implements_of_result _ _ _ _ _
    (fun x y=>KernelScalarProfile.pack16 (f.atomic16 (KernelScalarProfile.unpack16 x) (KernelScalarProfile.unpack16 y)))
    f.atomic32 rfl rfl
  intro xs base cell
  have h:=Grid.native_complete f tree rest plans rowScale output s c columnMajor
    (input (FP.Value.zero .fp32) s rowScale xs) q base cell
  dsimp only [Grid.nativeAction] at h
  have typed:=finalValues_launch (KernelScalarProfile.native f tree rest) signed32 (KernelNativeModel.operations f).toOps
    plans (fun x y=>KernelScalarProfile.pack16 (f.atomic16 (KernelScalarProfile.unpack16 x) (KernelScalarProfile.unpack16 y)))
    f.atomic32 rowScale output s c columnMajor q xs (FP.Value.zero .fp32) base cell
  have defaultZero : (default : Word32)=FP.Value.zero .fp32 := rfl
  simp only [defaultZero] at typed ⊢
  rw [h] at typed
  change _=FP.Structural.Cell.mk (dtype output)
    ((Grid.nativeWord f tree rest plans rowScale output s c columnMajor
      (input (FP.Value.zero .fp32) s rowScale xs) q cell).read (dtype output))
  rw [h,typed]
  cases output <;> rfl

#axiomsClean programs_safe
#axiomsClean finalValues_launch
#axiomsClean real_implements
#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Gemm
