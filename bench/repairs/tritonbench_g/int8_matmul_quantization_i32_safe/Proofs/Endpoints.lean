import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.Tensor
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.Composition
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectHelperResults

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Tensor
open VeriTile.Triton FP.Structural KernelExecution Direct
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word32 := ⟨FP.Value.zero .fp32⟩

/-- Every event of the native Tensor execution is backed by the complete
readable candidate source, at the actual current output cell. -/
theorem source_native (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (base : Nat → Nat) (request : Request) (laws : GemmLaws base request.original)
    (out : Nat) (caller prepared : Memory Word32 Word16) (template : State Word32)
    (chosen : Host.Invocation request.original.shape) (before : List (Int8MatmulQuantization.Commit request.original.shape chosen.config))
    (e : Int8MatmulQuantization.Commit request.original.shape chosen.config) :
    let r:=request.original
    let b:=r.buffers out
    let input:=originalInput KernelScalarProfile.pack16 r caller
    let current:=Grid.run r.shape chosen.config
      (Grid.nativeAction f tree rest chosen.plans b.aScalePrecision b.outPrecision r.shape chosen.config b.bColumnMajor input)
      before (Grid.reset r.shape (Grid.zeroCell b.outPrecision (FP.Value.zero .fp32))
        (bind KernelScalarProfile.pack16 r out caller prepared template))
    ∃ physical : PhysicalCommit r.shape chosen.config, ∃ final,
      physical.logical=e ∧ KernelExecution.exec (KernelScalarProfile.native f tree rest) signed32
        (KernelNativeModel.operations f).toOps chosen.plans
        (Gemm.source b.aScalePrecision b.outPrecision (Gemm.parameters r.shape chosen.config b.bColumnMajor))
        (Grid.atProgram current physical.pid physical.split)=some final ∧
      final.mem "C" (Grid.outputOffset r.shape e.1)=
        Grid.nativeAction f tree rest chosen.plans b.aScalePrecision b.outPrecision r.shape chosen.config b.bColumnMajor input e
          (current.mem "C" (Grid.outputOffset r.shape e.1)) := by
  have hm : 0<request.original.shape.M := Nat.zero_lt_of_lt e.1.1.isLt
  have hn : 0<request.original.shape.N := Nat.zero_lt_of_lt e.1.2.isLt
  have reads := original_reads KernelScalarProfile.pack16 base request.original laws out caller prepared template hm hn chosen.config
  simpa only [Grid.advance, State.write, ite_true] using
    Grid.source_at_prefix (KernelScalarProfile.native f tree rest) signed32 (KernelNativeModel.operations f).toOps chosen.plans
      (fun x y => KernelScalarProfile.pack16 (f.atomic16 (KernelScalarProfile.unpack16 x) (KernelScalarProfile.unpack16 y)))
      f.atomic32 rfl rfl (request.original.buffers out).aScalePrecision (request.original.buffers out).outPrecision
      request.original.shape chosen.config (request.original.buffers out).bColumnMajor _ _ reads
      (Grid.zeroCell (request.original.buffers out).outPrecision (FP.Value.zero .fp32)) before e

theorem integer_safe (f : KernelNativeModel.FloatingInstructions) (request : Request)
    (chosen : Host.Invocation request.original.shape) (out : Nat) (input : Gemm.Input Word32)
    (cell : MatrixCell request.original.shape) (z : Fin chosen.config.S) (t : Nat) :
    Gemm.accumulator signed32 (KernelNativeModel.operations f).toOps chosen.plans
      (Gemm.parameters request.original.shape chosen.config (request.original.buffers out).bColumnMajor) input cell.1 cell.2 z t =
      (((planCoordinates chosen.config.BK t (chosen.plans chosen.config.BK cell.1 cell.2 z)).map
        (fun q => sourceK chosen.config q.1 z q.2)).map
        (maskedProduct request.original.shape.K (input.A cell.1) (fun k => input.B k cell.2))).sum :=
  accumulator_native_exact f request.original.shape
    (active_bound _ _ _ request.safe (Nat.zero_lt_of_lt cell.1.isLt) (Nat.zero_lt_of_lt cell.2.isLt))
    chosen.config _ chosen.plans input cell.1 cell.2 z t

def nativeRun (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (request : Request) (callerDevice : Nat) (caller : Memory Word32 Word16) (live : Nat → Prop)
    (alloc : GemmAllocation request.original caller live) (template : State Word32)
    (trials : List (Host.Invocation request.original.shape)) (chosen : Host.Invocation request.original.shape) : Result Word32 :=
  run KernelScalarProfile.pack16 request (KernelScalarProfile.native f tree rest) signed32 (KernelNativeModel.operations f).toOps
    (fun x y => KernelScalarProfile.pack16 (f.atomic16 (KernelScalarProfile.unpack16 x) (KernelScalarProfile.unpack16 y)))
    f.atomic32 (Grid.zeroCell (request.original.buffers alloc.output.info.storage).outPrecision (FP.Value.zero .fp32))
    callerDevice caller live alloc template trials chosen

theorem native_complete (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (request : Request) (callerDevice : Nat) (caller : Memory Word32 Word16) (live : Nat → Prop)
    (alloc : GemmAllocation request.original caller live) (template : State Word32)
    (trials : List (Host.Invocation request.original.shape)) (chosen : Host.Invocation request.original.shape) :
    let r:=request.original
    let result:=nativeRun f tree rest request callerDevice caller live alloc template trials chosen
    result.tensor=alloc.output ∧ GemmDescriptor r result.tensor ∧ result.currentDevice=callerDevice ∧
      (∀ cell : MatrixCell r.shape,result.memory (logicalAddress result.tensor (cell.1.val*r.shape.N+cell.2.val))=
        Grid.nativeWord f tree rest chosen.plans (r.buffers alloc.output.info.storage).aScalePrecision
          (r.buffers alloc.output.info.storage).outPrecision r.shape chosen.config (r.buffers alloc.output.info.storage).bColumnMajor
          (originalInput KernelScalarProfile.pack16 r caller) chosen.schedule cell) ∧
      (∀ p,live p.1 → ¬OutputRegion r.shape (r.buffers result.tensor.info.storage) p →
        result.memory p=encode KernelScalarProfile.pack16 (caller p)) := by
  exact complete KernelScalarProfile.pack16 request (KernelScalarProfile.native f tree rest) signed32
    (KernelNativeModel.operations f).toOps _ f.atomic32 _ callerDevice caller live alloc template trials chosen

def realRun (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (request : Request) (callerDevice : Nat)
    (caller : Memory ℝ ℝ) (live : Nat → Prop) (alloc : GemmAllocation request.original caller live) (template : State ℝ)
    (trials : List (Host.Invocation request.original.shape)) (chosen : Host.Invocation request.original.shape) : Result ℝ :=
  run id request (KernelScalarProfile.mathematical tree rest) mathematicalIntegers KernelIntegerIdeal.operations.toOps
    (·+·) (·+·) (Grid.zeroCell (request.original.buffers alloc.output.info.storage).outPrecision 0)
    callerDevice caller live alloc template trials chosen

/-- Independent dense target over the original caller's logical Tensor
inputs. This is a second interpretation, not a native=Real premise. -/
theorem real_complete (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (request : Request) (callerDevice : Nat)
    (caller : Memory ℝ ℝ) (live : Nat → Prop) (alloc : GemmAllocation request.original caller live) (template : State ℝ)
    (trials : List (Host.Invocation request.original.shape)) (chosen : Host.Invocation request.original.shape) :
    let r:=request.original
    let result:=realRun tree rest request callerDevice caller live alloc template trials chosen
    result.tensor=alloc.output ∧ GemmDescriptor r result.tensor ∧ result.currentDevice=callerDevice ∧
      (∀ cell : MatrixCell r.shape,result.memory (logicalAddress result.tensor (cell.1.val*r.shape.N+cell.2.val))=
        Grid.zeroCell (r.buffers alloc.output.info.storage).outPrecision
          (target r.shape (Gemm.realInput r.shape (originalInput id r caller)) cell.1 cell.2)) ∧
      (∀ p,live p.1 → ¬OutputRegion r.shape (r.buffers result.tensor.info.storage) p → result.memory p=encode id (caller p)) := by
  obtain ⟨identity,descriptor,device,values,frame⟩ := complete id request (KernelScalarProfile.mathematical tree rest)
    mathematicalIntegers KernelIntegerIdeal.operations.toOps (·+·) (·+·)
    (Grid.zeroCell (request.original.buffers alloc.output.info.storage).outPrecision 0)
    callerDevice caller live alloc template trials chosen
  refine ⟨identity,descriptor,device,?_,frame⟩
  intro cell
  unfold realRun
  rw [values cell]
  exact (Grid.launch_cell request.original.shape chosen.config
    (Grid.realAction tree rest chosen.plans (request.original.buffers alloc.output.info.storage).aScalePrecision
      (request.original.buffers alloc.output.info.storage).outPrecision request.original.shape chosen.config
      (request.original.buffers alloc.output.info.storage).bColumnMajor (originalInput id request.original caller))
      chosen.schedule _ template cell).symm.trans
    (Grid.real_complete tree rest chosen.plans _ _ _ _ _ _ chosen.schedule template cell)

/-- Byte observations use the shared native word encodings, including the
16-bit width for packed-half structural cells. -/
def nativeByte (cell : FP.Structural.Cell Word32) (i : Nat) : Byte :=
  match cell with
  | .mk .int x => BitVec.ofInt 8 x
  | .mk .real x => x.bits.extractLsb' (8*i) 8
  | .mk .fp16 x => (KernelScalarProfile.unpack16 x).bits.extractLsb' (8*i) 8
  | _ => 0

theorem byte_encoding (cell : Int8MatmulQuantization.Cell Word32 Word16) (i : Nat) :
    nativeByte (encode KernelScalarProfile.pack16 cell) i=cellByte cell i := by
  cases cell with
  | inl byte => simp [nativeByte,encode,cellByte]
  | inr value => cases value <;> simp [nativeByte,encode,cellByte,KernelScalarProfile.half_roundtrip]

/-- The unchanged Torch helper retains its own contract, axes and native
local-operation boundary. The new launch guard is absent from this API. -/
abbrev helperRun := @Direct.runHelper
abbrev helperComplete := @Direct.helper_complete
abbrev helperRealComplete := @Direct.helper_real_complete
abbrev helperNativeComplete := @Direct.helper_native_if_local

#axiomsClean source_native
#axiomsClean integer_safe
#axiomsClean native_complete
#axiomsClean real_complete
#axiomsClean byte_encoding
#axiomsClean helperComplete
#axiomsClean helperRealComplete
#axiomsClean helperNativeComplete
#specNonCircular originalInput avoiding [nativeRun, realRun, run, Host.runtime, KernelExecution.exec]
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Tensor
