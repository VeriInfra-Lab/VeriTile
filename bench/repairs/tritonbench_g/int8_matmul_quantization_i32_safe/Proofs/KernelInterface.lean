import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.Composition

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.KernelInterface
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
variable {I F H : Type} [Inhabited F]

/-- Every commit prefix is backed by a complete successful quoted source program
at its current memory, with the same independently selected commit effect. -/
def SourceExecution (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (atomicHalf atomicSingle : F → F → F) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (input : Gemm.Input F)
    (initial : State F) (zero : FP.Structural.Cell F) : Prop :=
  ∀ (before : List (Int8MatmulQuantization.Commit s c)) (e : Int8MatmulQuantization.Commit s c),
    let act := Grid.action M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor input
    let current := Grid.run s c act before (Grid.reset s zero initial)
    ∃ physical : PhysicalCommit s c, ∃ final,
      physical.logical=e ∧
      KernelExecution.exec M carrier ops plans (Gemm.source rowScale output (Gemm.parameters s c columnMajor))
        (Grid.atProgram current physical.pid physical.split)=some final ∧
      final.mem "C" (Grid.outputOffset s e.1)=(Grid.advance s c act current e).mem "C" (Grid.outputOffset s e.1)

def RealResult (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (plans : Plans)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (input : Gemm.Input ℝ)
    (q : Schedule s c) (initial : State ℝ) : Prop :=
  SourceExecution (KernelScalarProfile.mathematical tree rest) mathematicalIntegers KernelIntegerIdeal.operations.toOps
    plans (·+·) (·+·) rowScale output s c columnMajor input initial (Grid.zeroCell output 0) ∧
  (∀ cell : MatrixCell s,
    (Grid.launch s c (Grid.realAction tree rest plans rowScale output s c columnMajor input) q
      (Grid.zeroCell output 0) initial).mem "C" (Grid.outputOffset s cell)=
        Grid.zeroCell output (target s (Gemm.realInput s input) cell.1 cell.2)) ∧
  (∀ (region : RegionName) offset, region≠"C" ∨ s.M*s.N≤offset →
    (Grid.launch s c (Grid.realAction tree rest plans rowScale output s c columnMajor input) q
      (Grid.zeroCell output 0) initial).mem region offset=initial.mem region offset)

theorem real (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (plans : Plans)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (input : Gemm.Input ℝ)
    (q : Schedule s c) (initial : State ℝ)
    (reads : Gemm.Reads (Gemm.parameters s c columnMajor) rowScale input initial) :
    RealResult tree rest plans rowScale output s c columnMajor input q initial := by
  refine ⟨?_,Grid.real_complete tree rest plans rowScale output s c columnMajor input q initial,
    Grid.launch_frame _ _ _ _ _ _⟩
  intro before e
  exact Grid.source_at_prefix (KernelScalarProfile.mathematical tree rest) mathematicalIntegers
    KernelIntegerIdeal.operations.toOps plans (·+·) (·+·) rfl rfl rowScale output s c columnMajor input
    initial reads (Grid.zeroCell output 0) before e

local instance : Inhabited Word32 := ⟨FP.Value.zero .fp32⟩
def NativeResult (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (plans : Plans) (rowScale output : Precision) (s : Shape) (c : Config)
    (columnMajor : Bool) (input : Gemm.Input Word32) (q : Schedule s c) (initial : State Word32) : Prop :=
  SourceExecution (KernelScalarProfile.native f tree rest) signed32 (KernelNativeModel.operations f).toOps plans
    (fun x y => KernelScalarProfile.pack16 (f.atomic16 (KernelScalarProfile.unpack16 x) (KernelScalarProfile.unpack16 y)))
    f.atomic32 rowScale output s c columnMajor input initial (Grid.zeroCell output (FP.Value.zero .fp32)) ∧
  (∀ cell : MatrixCell s,
    (Grid.launch s c (Grid.nativeAction f tree rest plans rowScale output s c columnMajor input) q
      (Grid.zeroCell output (FP.Value.zero .fp32)) initial).mem "C" (Grid.outputOffset s cell)=
        Grid.nativeWord f tree rest plans rowScale output s c columnMajor input q cell) ∧
  (∀ (region : RegionName) offset, region≠"C" ∨ s.M*s.N≤offset →
    (Grid.launch s c (Grid.nativeAction f tree rest plans rowScale output s c columnMajor input) q
      (Grid.zeroCell output (FP.Value.zero .fp32)) initial).mem region offset=initial.mem region offset)

theorem native (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (plans : Plans) (rowScale output : Precision) (s : Shape) (c : Config)
    (columnMajor : Bool) (input : Gemm.Input Word32) (q : Schedule s c) (initial : State Word32)
    (reads : Gemm.Reads (Gemm.parameters s c columnMajor) rowScale input initial) :
    NativeResult f tree rest plans rowScale output s c columnMajor input q initial := by
  refine ⟨?_,Grid.native_complete f tree rest plans rowScale output s c columnMajor input q initial,
    Grid.launch_frame _ _ _ _ _ _⟩
  intro before e
  exact Grid.source_at_prefix (KernelScalarProfile.native f tree rest) signed32
    (KernelNativeModel.operations f).toOps plans
    (fun x y => KernelScalarProfile.pack16 (f.atomic16 (KernelScalarProfile.unpack16 x) (KernelScalarProfile.unpack16 y)))
    f.atomic32 rfl rfl rowScale output s c columnMajor input initial reads
    (Grid.zeroCell output (FP.Value.zero .fp32)) before e

#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.KernelInterface
