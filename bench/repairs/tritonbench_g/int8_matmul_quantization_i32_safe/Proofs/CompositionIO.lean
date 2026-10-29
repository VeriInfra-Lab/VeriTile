import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.RawCorrect
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.RawNative
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ProducerIOResults
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.GemmIOBinding

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Composition
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxRecDepth 28000
set_option maxHeartbeats 3000000
variable {F I H : Type} [Inhabited F]

/-- The original arbitrary commit-prefix execution obligation, expressed with
the shared complete GEMM invocation. This is an operational transport helper;
the public tensor IO judgments remain in the reader interfaces. -/
def SourceExecution (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (integerZero : carrier.decode ops.izero=0)
    (plans : Plans) (atomicHalf atomicSingle : F → F → F) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (input : I32Safe.Gemm.Input F)
    (initial : State F) (zero : FP.Structural.Cell F) : Prop :=
  ∀ (before : List (Int8MatmulQuantization.Commit s c)) (e : Int8MatmulQuantization.Commit s c),
    let act := Grid.action M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor input
    let current := Grid.run s c act before (Grid.reset s zero initial)
    ∃ physical : PhysicalCommit s c, ∃ final,
      physical.logical=e ∧
      (Gemm.invocation M carrier ops integerZero plans rowScale output s c columnMajor
        (physical.pid,physical.split)).run current=some final ∧
      final.mem "C" (Grid.outputOffset s e.1)=
        (Grid.advance s c act current e).mem "C" (Grid.outputOffset s e.1)

/-- Transport changes only the execution interface, including at prefixes
that are not prefixes of the selected complete schedule. -/
theorem source_execution_iff (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (integerZero : carrier.decode ops.izero=0)
    (plans : Plans) (atomicHalf atomicSingle : F → F → F) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (input : I32Safe.Gemm.Input F)
    (initial : State F) (zero : FP.Structural.Cell F) :
    SourceExecution M carrier ops integerZero plans atomicHalf atomicSingle rowScale output
      s c columnMajor input initial zero ↔
    KernelInterface.SourceExecution M carrier ops plans atomicHalf atomicSingle rowScale output
      s c columnMajor input initial zero := by
  unfold SourceExecution KernelInterface.SourceExecution
  simp only [Gemm.invocation_run]

omit [Inhabited F] in
/-- The intermediate state is attached by these two fixed storage aliases.
No independent GEMM A/AS state is chosen. -/
theorem attach_inputs (produced : State F) (offset : Nat) :
    (Composed.attach produced).mem "A" offset=produced.mem "q" offset ∧
    (Composed.attach produced).mem "AS" offset=produced.mem "scales" offset := by
  simp [Composed.attach]

omit [Inhabited F] in
theorem attach_other (produced : State F) (region : RegionName) (offset : Nat)
    (notA : region≠"A") (notAS : region≠"AS") :
    (Composed.attach produced).mem region offset=produced.mem region offset := by
  simp only [Composed.attach,if_neg notA,if_neg notAS]

/-- Complete producer-to-GEMM composition in the shared execution interfaces.
The same produced state supplies the fixed A/AS aliases, every arbitrary
GEMM commit prefix, all intermediate values and both exterior frames. -/
theorem composed_real (s : Shape) (c : Config) (columnMajor : Bool) (width : Nat)
    (positive : 0<width) (covers : s.K≤width)
    (selection : I32Safe.Producer.Selection (Composed.geometry s width positive))
    (rest : Algebra ℝ) (plans : Plans) (output : Precision) (values : Nat → Nat → ℝ)
    (B : Nat → Nat → Byte) (bs : Nat → ℝ) (state : State ℝ)
    (reads : KernelProducerTargets.Reads (Composed.geometry s width positive) values state)
    (weights : Composed.WeightReads s c columnMajor B bs state)
    (order : I32Safe.Producer.Schedule (Composed.geometry s width positive))
    (schedule : Int8MatmulQuantization.Schedule s c) :
    ∃ produced,
      runSchedule (order.order.map (Producer.invocation
        (KernelScalarProfile.mathematical selection.tree rest) realInteger8
        (Composed.geometry s width positive))) state=some produced ∧
      (∀ r : Fin s.M, ∀ k : Fin width, k.val<s.K → produced.mem "q" (r.val*s.K+k.val)=
        .mk .int (KernelProducerTargets.mathematicalByte (values r) s.K k).toInt) ∧
      (∀ r : Fin s.M, produced.mem "scales" r=
        .mk .fp16 (KernelProducerTargets.mathematicalScale (values r) s.K)) ∧
      (∀ (region : RegionName) offset, (region≠"q" ∨ s.M*s.K≤offset) →
        (region≠"scales" ∨ s.M≤offset) → produced.mem region offset=state.mem region offset) ∧
      I32Safe.Gemm.Reads (I32Safe.Gemm.parameters s c columnMajor) .half
        (Composed.realInput s values B bs) (Composed.attach produced) ∧
      SourceExecution (KernelScalarProfile.mathematical selection.tree rest)
        mathematicalIntegers KernelIntegerIdeal.operations.toOps rfl plans (·+·) (·+·)
        .half output s c columnMajor (Composed.realInput s values B bs)
        (Composed.attach produced) (Grid.zeroCell output 0) ∧
      (∀ cell : MatrixCell s,
        (Grid.launch s c (Grid.realAction selection.tree rest plans .half output s c columnMajor
          (Composed.realInput s values B bs)) schedule (Grid.zeroCell output 0)
          (Composed.attach produced)).mem "C" (Grid.outputOffset s cell)=
            Grid.zeroCell output
              (target s (I32Safe.Gemm.realInput s (Composed.realInput s values B bs)) cell.1 cell.2)) ∧
      (∀ (region : RegionName) offset, region≠"C" ∨ s.M*s.N≤offset →
        (Grid.launch s c (Grid.realAction selection.tree rest plans .half output s c columnMajor
          (Composed.realInput s values B bs)) schedule (Grid.zeroCell output 0)
          (Composed.attach produced)).mem region offset=(Composed.attach produced).mem region offset) := by
  obtain ⟨produced,ran,data,scales,frame,result⟩ := Public.composedCorrect s c columnMajor
    width positive covers selection rest plans output values B bs state reads weights order schedule
  have bound : I32Safe.Gemm.Reads (I32Safe.Gemm.parameters s c columnMajor) .half
      (Composed.realInput s values B bs) (Composed.attach produced) :=
    Composed.reads_produced s c columnMajor width covers state produced _ _ _ _ data scales weights
      (fun region i hq hs => frame region i (Or.inl hq) (Or.inl hs))
  refine ⟨produced,(Producer.schedule_eq _ _ _ _ _).trans ran,data,scales,frame,bound,
    ?_,result.2.1,result.2.2⟩
  exact (source_execution_iff _ _ _ rfl _ _ _ _ _ _ _ _ _ _ _).mpr result.1

local instance kernelInhabitedWord32 : Inhabited Word32 := ⟨FP.Value.zero .fp32⟩

/-- Native composition retains the original quantized bytes, stored FP16
scales, integer/MMA plans, output casts and arbitrary atomic commit order.
The producer result, fixed aliases and every GEMM prefix use one state. -/
theorem composed_native (f : KernelNativeModel.FloatingInstructions) (s : Shape) (c : Config)
    (columnMajor : Bool) (width : Nat) (positive : 0<width) (covers : s.K≤width)
    (selection : I32Safe.Producer.Selection (Composed.geometry s width positive))
    (rest : Algebra Word32) (plans : Plans) (output : Precision)
    (values : Nat → Nat → Word32) (B : Nat → Nat → Byte) (bs : Nat → Word32)
    (state : State Word32)
    (reads : KernelProducerTargets.Reads (Composed.geometry s width positive) values state)
    (weights : Composed.WeightReads s c columnMajor B bs state)
    (order : I32Safe.Producer.Schedule (Composed.geometry s width positive))
    (schedule : Int8MatmulQuantization.Schedule s c) :
    ∃ produced,
      runSchedule (order.order.map (Producer.invocation
        (KernelScalarProfile.native f selection.tree rest) KernelNativeModel.integer8
        (Composed.geometry s width positive))) state=some produced ∧
      (∀ r : Fin s.M, ∀ k : Fin width, k.val<s.K → produced.mem "q" (r.val*s.K+k.val)=
        .mk .int (KernelProducerTargets.nativeByte f (Composed.geometry s width positive)
          selection (values r) k).toInt) ∧
      (∀ r : Fin s.M, produced.mem "scales" r=.mk .fp16
        (KernelScalarProfile.pack16 (KernelProducerTargets.nativeStoredScale f
          (Composed.geometry s width positive) selection (values r)))) ∧
      (∀ (region : RegionName) offset, (region≠"q" ∨ s.M*s.K≤offset) →
        (region≠"scales" ∨ s.M≤offset) → produced.mem region offset=state.mem region offset) ∧
      I32Safe.Gemm.Reads (I32Safe.Gemm.parameters s c columnMajor) .half
        (Composed.nativeInput f (Composed.geometry s width positive) selection values B bs)
        (Composed.attach produced) ∧
      SourceExecution (KernelScalarProfile.native f selection.tree rest) signed32
        (KernelNativeModel.operations f).toOps rfl plans
        (fun x y => KernelScalarProfile.pack16
          (f.atomic16 (KernelScalarProfile.unpack16 x) (KernelScalarProfile.unpack16 y)))
        f.atomic32 .half output s c columnMajor
        (Composed.nativeInput f (Composed.geometry s width positive) selection values B bs)
        (Composed.attach produced) (Grid.zeroCell output (FP.Value.zero .fp32)) ∧
      (∀ cell : MatrixCell s,
        (Grid.launch s c (Grid.nativeAction f selection.tree rest plans .half output s c columnMajor
          (Composed.nativeInput f (Composed.geometry s width positive) selection values B bs)) schedule
          (Grid.zeroCell output (FP.Value.zero .fp32)) (Composed.attach produced)).mem "C"
          (Grid.outputOffset s cell)=Grid.nativeWord f selection.tree rest plans .half output s c columnMajor
            (Composed.nativeInput f (Composed.geometry s width positive) selection values B bs) schedule cell) ∧
      (∀ (region : RegionName) offset, region≠"C" ∨ s.M*s.N≤offset →
        (Grid.launch s c (Grid.nativeAction f selection.tree rest plans .half output s c columnMajor
          (Composed.nativeInput f (Composed.geometry s width positive) selection values B bs)) schedule
          (Grid.zeroCell output (FP.Value.zero .fp32)) (Composed.attach produced)).mem region offset=
            (Composed.attach produced).mem region offset) := by
  obtain ⟨produced,ran,data,scales,frame,result⟩ := Public.composedFPEquiv f s c columnMajor
    width positive covers selection rest plans output values B bs state reads weights order schedule
  have bound : I32Safe.Gemm.Reads (I32Safe.Gemm.parameters s c columnMajor) .half
      (Composed.nativeInput f (Composed.geometry s width positive) selection values B bs)
      (Composed.attach produced) :=
    Composed.reads_produced s c columnMajor width covers state produced _ _ _ _ data scales weights
      (fun region i hq hs => frame region i (Or.inl hq) (Or.inl hs))
  refine ⟨produced,(Producer.schedule_eq _ _ _ _ _).trans ran,data,scales,frame,bound,
    ?_,result.2.1,result.2.2⟩
  exact (source_execution_iff _ _ _ rfl _ _ _ _ _ _ _ _ _ _ _).mpr result.1

#axiomsClean source_execution_iff
#axiomsClean attach_inputs
#axiomsClean attach_other
#axiomsClean composed_real
#axiomsClean composed_native
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.Composition
