import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.Lifecycle
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectPointers
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectGemmReal

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Tensor
open VeriTile.Triton FP.Structural KernelExecution Direct
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false
variable {I F H : Type} [Inhabited F]

/-- Original shared Tensor request plus exactly the new launch guard. -/
structure Request where
  original : Direct.GemmRequest
  safe : LaunchSafe original.shape.M original.shape.N original.shape.K

def encode (pack : H → F) : Int8MatmulQuantization.Cell F H → FP.Structural.Cell F
  | .inl byte => .mk .int byte.toInt
  | .inr (.inl word) => .mk .real word
  | .inr (.inr word) => .mk .fp16 (pack word)

theorem byte_read (pack : H → F) (cell : Int8MatmulQuantization.Cell F H) :
    (encode pack cell).read .int = (read8 (fun _ => cell) (0,0)).toInt := by
  cases cell with
  | inl byte => simp [encode, read8, Int8DequantMatmul.read8]
  | inr value => cases value <;> simp [encode, read8, Int8DequantMatmul.read8, FP.Structural.Cell.read, defaultValue]

/-- Input-only specification reads the original Tensor's logical elements;
it has no source execution, result, allocation or lowered graph field. -/
def originalInput (pack : H → F) (r : Direct.GemmRequest) (m : Memory F H) : Gemm.Input F where
  A := fun row k => read8 m (logicalAddress r.a (row*r.shape.K+k))
  B := fun k n => read8 m (logicalAddress r.b (k*r.shape.N+n))
  aScale := fun row => match (r.buffers 0).aScalePrecision with
    | .half => (encode pack (m (logicalAddress r.aScale row))).read .fp16
    | .single => (encode pack (m (logicalAddress r.aScale row))).read .real
  bScale := fun n => (encode pack (m (logicalAddress r.bScale n))).read .real

/-- Local ABI view of the original storage slices. Existing tensor pointer,
layout and allocation laws determine these addresses. -/
def bind (pack : H → F) (r : Direct.GemmRequest) (out : Nat) (m output : Memory F H)
    (template : State F) : State F :=
  { template with mem := fun region i =>
      if region = "A" then encode pack (m (r.a.info.storage,r.a.info.offset+i))
      else if region = "B" then encode pack (m (r.b.info.storage,r.b.info.offset+i))
      else if region = "AS" then encode pack (m (r.aScale.info.storage,r.aScale.info.offset+i))
      else if region = "BS" then encode pack (m (r.bScale.info.storage,r.bScale.info.offset+i))
      else if region = "C" then encode pack (output (out,offset (r.out.map TensorView.info)+i))
      else template.mem region i }

theorem original_reads (pack : H → F) (base : Nat → Nat) (r : Direct.GemmRequest) (laws : GemmLaws base r)
    (out : Nat) (m output : Memory F H) (template : State F) (hm : 0<r.shape.M) (hn : 0<r.shape.N) (c : Config) :
    Gemm.Reads (Gemm.parameters r.shape c (r.buffers out).bColumnMajor) (r.buffers out).aScalePrecision
      (originalInput pack r m) (bind pack r out m output template) := by
  constructor
  · intro row k hr hk
    have pos : 1 ≤ r.shape.K := by change k<r.shape.K at hk; omega
    have address := gemm_a_address base r laws out row k hr hk
    simp only [Gemm.parameters, bind, ite_true, Nat.mul_one, max_eq_left pos]
    change (encode pack (m (r.a.info.storage,r.a.info.offset+(row*r.shape.K+k)))).read .int = _
    have address' : (r.a.info.storage,r.a.info.offset+(row*r.shape.K+k))=logicalAddress r.a (row*r.shape.K+k) := by
      simpa [GemmRequest.buffers,gemmBuffers,Nat.add_assoc] using address
    rw [address']
    exact byte_read pack _
  · intro k n hk hc
    have posK : 1 ≤ r.shape.K := by change k<r.shape.K at hk; omega
    have posN : 1 ≤ r.shape.N := by omega
    have address := gemm_b_address base r laws out k n hk hc
    have address' : (r.b.info.storage,r.b.info.offset+
        (k*(if (r.buffers out).bColumnMajor then 1 else max r.shape.N 1)+
          n*(if (r.buffers out).bColumnMajor then max r.shape.K 1 else 1)))=
        logicalAddress r.b (k*r.shape.N+n) := by
      cases kind : columnMajor r.b.info <;>
        simpa [Buffers.bIndex, GemmRequest.buffers,gemmBuffers,max_eq_left posK,max_eq_left posN,
          kind,Nat.mul_comm,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc] using address
    simp only [Gemm.parameters, bind, show ("B" : RegionName) ≠ "A" by decide, ite_false, ite_true]
    rw [address']
    exact byte_read pack _
  · intro row hr
    have address := (gemm_scale_addresses base r laws out (⟨row,hr⟩,⟨0,hn⟩)).1
    have address' : (r.aScale.info.storage,r.aScale.info.offset+row)=logicalAddress r.aScale row := by
      simpa [GemmRequest.buffers,gemmBuffers,offset] using address
    cases kind : (r.buffers out).aScalePrecision <;>
      simp [Gemm.parameters, bind, originalInput, Nat.mul_one, address',
        show (r.buffers 0).aScalePrecision=(r.buffers out).aScalePrecision by rfl,kind]
  · intro n hc
    have address := (gemm_scale_addresses base r laws out (⟨0,hm⟩,⟨n,hc⟩)).2
    have address' : (r.bScale.info.storage,r.bScale.info.offset+n)=logicalAddress r.bScale n := by
      simpa [GemmRequest.buffers,gemmBuffers] using address
    simp [Gemm.parameters, bind, originalInput, Nat.mul_one, address']

/-- Allocation cannot change any of the original input storage slices;
this follows from shared separation/allocation APIs, without result premises. -/
theorem allocated_inputs (r : Direct.GemmRequest) (caller : Memory F H) (live : Nat → Prop)
    (inputs : r.live live) (alloc : GemmAllocation r caller live) :
    ∀ t∈r.views, ∀ i, alloc.memory (t.info.storage,i)=caller (t.info.storage,i) := by
  intro t member i
  apply alloc.frame
  left
  cases eq : r.out with
  | none => intro same; dsimp only at same; have h:=inputs t member; rw [same] at h; exact alloc.fresh eq h
  | some output =>
    have identity : alloc.output=output := by simpa [GemmDescriptor,eq] using alloc.descriptor
    have h:=r.contract.output output.info (by simp [eq])
    have mem : t.info∈Direct.inputs r.a.info (some r.aScale.info) r.b.info r.bScale.info Bool.false := by
      simpa [GemmRequest.views,Direct.inputs] using List.mem_map.mpr ⟨t,member,rfl⟩
    simpa [identity] using (h.separated t.info mem).symm

def outputMemory (pack : H → F) (r : Direct.GemmRequest) (out : Nat) (state : State F)
    (prepared : Memory F H) : Address → FP.Structural.Cell F := fun p =>
  if p.1=out ∧ offset (r.out.map TensorView.info)≤p.2 ∧
      p.2<offset (r.out.map TensorView.info)+r.shape.M*r.shape.N
  then state.mem "C" (p.2-offset (r.out.map TensorView.info)) else encode pack (prepared p)

theorem output_readback (pack : H → F) (r : Direct.GemmRequest) (caller : Memory F H) (live : Nat → Prop)
    (alloc : GemmAllocation r caller live) (state : State F) (cell : MatrixCell r.shape) :
    outputMemory pack r alloc.output.info.storage state alloc.memory
      (logicalAddress alloc.output (cell.1.val*r.shape.N+cell.2.val)) = state.mem "C" (Grid.outputOffset r.shape cell) := by
  rw [alloc.binding cell]
  have h:=RowwiseQuantizationTriton.linear_bound r.shape.M r.shape.N cell.1 cell.2 cell.1.isLt cell.2.isLt
  simp [outputMemory,outputAddress,GemmRequest.buffers,gemmBuffers,Grid.outputOffset,Nat.add_assoc,h,
    show offset (r.out.map TensorView.info) ≤ offset (r.out.map TensorView.info)+cell.1.val*r.shape.N+cell.2.val by omega,
    show offset (r.out.map TensorView.info)+cell.1.val*r.shape.N+cell.2.val <
      offset (r.out.map TensorView.info)+r.shape.M*r.shape.N by omega]

theorem output_frame (pack : H → F) (r : Direct.GemmRequest) (caller : Memory F H) (live : Nat → Prop)
    (alloc : GemmAllocation r caller live) (state : State F) (p : Address) (hp : live p.1)
    (outside : ¬OutputRegion r.shape (r.buffers alloc.output.info.storage) p) :
    outputMemory pack r alloc.output.info.storage state alloc.memory p=encode pack (caller p) := by
  have outside' : ¬(p.1=alloc.output.info.storage ∧ offset (r.out.map TensorView.info)≤p.2 ∧
      p.2<offset (r.out.map TensorView.info)+r.shape.M*r.shape.N) := outside
  rw [outputMemory, if_neg outside']
  apply congrArg (encode pack)
  apply alloc.frame
  cases eq : r.out with
  | none => left; intro same; rw [same] at hp; exact alloc.fresh eq hp
  | some out => right; simp [eq]

structure Result (F : Type) where
  tensor : TensorView
  memory : Address → FP.Structural.Cell F
  currentDevice : Nat

def run (pack : H → F) (request : Request) (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (atomicHalf atomicSingle : F → F → F)
    (zero : FP.Structural.Cell F) (callerDevice : Nat) (caller : Memory F H) (live : Nat → Prop)
    (alloc : GemmAllocation request.original caller live) (template : State F)
    (trials : List (Host.Invocation request.original.shape)) (chosen : Host.Invocation request.original.shape) : Result F :=
  let r:=request.original
  let b:=r.buffers alloc.output.info.storage
  let initial:=bind pack r alloc.output.info.storage caller alloc.memory template
  let final:=if r.shape.M=0 ∨ r.shape.N=0 then initial else
    Host.runtime M carrier ops atomicHalf atomicSingle b.aScalePrecision b.outPrecision r.shape b.bColumnMajor
      (originalInput pack r caller) zero trials chosen initial
  ⟨alloc.output,outputMemory pack r alloc.output.info.storage final alloc.memory,callerDevice⟩

/-- Exact returned object, original logical output view, all caller frames,
and the complete source-ordered graph, with no lowered-result assumption. -/
theorem complete (pack : H → F) (request : Request) (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (atomicHalf atomicSingle : F → F → F)
    (zero : FP.Structural.Cell F) (callerDevice : Nat) (caller : Memory F H) (live : Nat → Prop)
    (alloc : GemmAllocation request.original caller live) (template : State F)
    (trials : List (Host.Invocation request.original.shape)) (chosen : Host.Invocation request.original.shape) :
    let r:=request.original
    let result:=run pack request M carrier ops atomicHalf atomicSingle zero callerDevice caller live alloc template trials chosen
    result.tensor=alloc.output ∧ GemmDescriptor r result.tensor ∧ result.currentDevice=callerDevice ∧
      (∀ cell : MatrixCell r.shape, result.memory (logicalAddress result.tensor (cell.1.val*r.shape.N+cell.2.val))=
        (projection cell chosen.schedule.events).foldl
          (fun old z => Host.act M carrier ops atomicHalf atomicSingle (r.buffers alloc.output.info.storage).aScalePrecision
            (r.buffers alloc.output.info.storage).outPrecision r.shape (r.buffers alloc.output.info.storage).bColumnMajor
            (originalInput pack r caller) chosen (cell,z) old) zero) ∧
      (∀ p,live p.1 → ¬OutputRegion r.shape (r.buffers result.tensor.info.storage) p → result.memory p=encode pack (caller p)) := by
  dsimp only
  refine ⟨rfl,alloc.descriptor,rfl,?_,?_⟩
  · intro cell
    have nonempty : ¬(request.original.shape.M=0 ∨ request.original.shape.N=0) := by
      have hm:=cell.1.isLt; have hn:=cell.2.isLt; omega
    simp only [run, if_neg nonempty]
    rw [output_readback, Host.runtime_cell]
  · intro p hp outside
    exact output_frame pack request.original caller live alloc _ p hp outside

#axiomsClean original_reads
#axiomsClean allocated_inputs
#axiomsClean output_readback
#axiomsClean complete
#specNonCircular originalInput avoiding [run, Host.runtime, Gemm.source, KernelExecution.exec]
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Tensor
