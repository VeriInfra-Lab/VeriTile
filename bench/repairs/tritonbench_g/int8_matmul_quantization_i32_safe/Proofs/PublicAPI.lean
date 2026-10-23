import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.Admission

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Public
open VeriTile.Triton FP.Structural Direct KernelExecution
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
variable {I F H : Type} [Inhabited F]

/-- Public physical backing relation: actual prepared memory, no aliased
output/input storage, and pointer identities for original/returned tensors. -/
def Binding (pack : H → F) (base : Nat → Nat) (request : Tensor.Request)
    (caller : Memory F H) (live : Nat → Prop) (alloc : GemmAllocation request.original caller live)
    (template : State F) : Prop :=
  (request.original.buffers alloc.output.info.storage).Separate ∧
  (∀ t∈request.original.views,InputPointerBinding base t) ∧ InputPointerBinding base alloc.output ∧
  Tensor.bind pack request.original alloc.output.info.storage caller alloc.memory template=
    Tensor.bind pack request.original alloc.output.info.storage alloc.memory alloc.memory template

theorem binding (pack : H → F) (base : Nat → Nat) (request : Tensor.Request)
    (caller : Memory F H) (live : Nat → Prop) (alloc : GemmAllocation request.original caller live)
    (admit : Admission.Gemm base request caller live alloc) (template : State F) :
    Binding pack base request caller live alloc template :=
  ⟨Admission.separate base request caller live alloc admit,
   (Admission.input_pointers base request caller live alloc admit).1,
   (Admission.input_pointers base request caller live alloc admit).2,
   Admission.prepared_binding pack base request caller live alloc admit template⟩

/-- The same source-derived result and caller frame as the numerical lemmas,
now used together with physical admission and source simulation below. -/
def Post (pack : H → F) (request : Tensor.Request) (caller : Memory F H) (live : Nat → Prop)
    (alloc : GemmAllocation request.original caller live) (callerDevice : Nat)
    (reference : MatrixCell request.original.shape → FP.Structural.Cell F) (result : Tensor.Result F) : Prop :=
  result.tensor=alloc.output ∧ GemmDescriptor request.original result.tensor ∧ result.currentDevice=callerDevice ∧
  (∀ cell : MatrixCell request.original.shape,
    result.memory (logicalAddress result.tensor (cell.1.val*request.original.shape.N+cell.2.val))=reference cell) ∧
  (∀ p,live p.1 → ¬OutputRegion request.original.shape (request.original.buffers result.tensor.info.storage) p →
    result.memory p=Tensor.encode pack (caller p))

/-- Every modeled commit is a complete readable-source execution on the
actual post-allocation input/output backing at its current old output. -/
def Simulation (pack : H → F) (request : Tensor.Request) (caller : Memory F H) (live : Nat → Prop)
    (alloc : GemmAllocation request.original caller live) (template : State F)
    (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (atomicHalf atomicSingle : F → F → F) (zero : FP.Structural.Cell F)
    (chosen : Host.Invocation request.original.shape) : Prop :=
  let r:=request.original
  let b:=r.buffers alloc.output.info.storage
  let act:=Host.act M carrier ops atomicHalf atomicSingle b.aScalePrecision b.outPrecision r.shape b.bColumnMajor
    (Tensor.originalInput pack r caller) chosen
  ∀ (before : List (Int8MatmulQuantization.Commit r.shape chosen.config)) (e : Int8MatmulQuantization.Commit r.shape chosen.config),
    let current:=Grid.run r.shape chosen.config act before (Grid.reset r.shape zero
      (Tensor.bind pack r alloc.output.info.storage alloc.memory alloc.memory template))
    ∃ physical : PhysicalCommit r.shape chosen.config, ∃ final,
      physical.logical=e ∧ KernelExecution.exec M carrier ops chosen.plans
        (Gemm.source b.aScalePrecision b.outPrecision (Gemm.parameters r.shape chosen.config b.bColumnMajor))
        (Grid.atProgram current physical.pid physical.split)=some final ∧
      final.mem "C" (Grid.outputOffset r.shape e.1)=
        (Grid.advance r.shape chosen.config act current e).mem "C" (Grid.outputOffset r.shape e.1)

theorem simulation (pack : H → F) (base : Nat → Nat) (request : Tensor.Request)
    (caller : Memory F H) (live : Nat → Prop) (alloc : GemmAllocation request.original caller live)
    (admit : Admission.Gemm base request caller live alloc) (template : State F)
    (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (atomicHalf atomicSingle : F → F → F) (half : M.atomicAdd .fp16=some atomicHalf)
    (single : M.atomicAdd .real=some atomicSingle) (zero : FP.Structural.Cell F)
    (chosen : Host.Invocation request.original.shape) :
    Simulation pack request caller live alloc template M carrier ops atomicHalf atomicSingle zero chosen := by
  intro before e
  have hm:=Nat.zero_lt_of_lt e.1.1.isLt
  have hn:=Nat.zero_lt_of_lt e.1.2.isLt
  have reads:=Tensor.original_reads pack base request.original admit.laws alloc.output.info.storage caller alloc.memory template hm hn chosen.config
  rw [Admission.prepared_binding pack base request caller live alloc admit template] at reads
  exact Grid.source_at_prefix M carrier ops chosen.plans atomicHalf atomicSingle half single
    (request.original.buffers alloc.output.info.storage).aScalePrecision
    (request.original.buffers alloc.output.info.storage).outPrecision request.original.shape chosen.config
    (request.original.buffers alloc.output.info.storage).bColumnMajor _ _ reads zero before e

#axiomsClean binding
#axiomsClean simulation
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Public
