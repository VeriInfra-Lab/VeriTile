import bench.tritonbench_g.triton_argmax.Proofs.Result
import bench.tritonbench_g.triton_argmax.Proofs.NativeSafety

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Tensor
open VeriTile.Bench.Repairs.TritonArgmax Source Memory TensorBinding
noncomputable section

/-- Actual returned Tensor objects are those of the unchanged wrapper. -/
def dimensionInvoke (mem : Primitives M) (before : M)
    (o : Checked.Observation) (ax : Nat) (geometry : NativeGraph.DimensionPrograms)
    (g : Native.DimPrograms (geometry.config o ax)) (call : ContiguousCall mem before x callers)
    (a : DimensionAllocation x o ax call.result callers) : Except String (TensorBinding.Result M) :=
  if Checked.check o then .ok ⟨a.output.returned,Native.dimension mem g a.buffers call.state⟩
  else .error "input contract rejected"

theorem dimension_tensor (mem : Primitives M) (before : M)
    (x : Input) (o : Checked.Observation) (observed : o.input=x.tensor.metadata)
    (h : Checked.InputContract o) (dim : Int) (branch : o.dim=some dim)
    (ax : Nat) (normalized : Host.normalizeDim o.input.shape.length dim=some ax)
    (geometry : NativeGraph.DimensionPrograms) (g : Native.DimPrograms (geometry.config o ax))
    (call : ContiguousCall mem before x callers) (prepared : o.prepared=call.result.tensor.metadata)
    (a : DimensionAllocation x o ax call.result callers) (nonempty : 0<Checked.extent o ax) :
    ∃ result,dimensionInvoke mem before o ax geometry g call a=.ok result ∧
      result.returned=a.output.returned ∧
      result.returned.metadata.shape=Host.outputShape x.tensor.metadata.shape (some ax) o.keepdim ∧
      result.returned.dtypeName="torch.int64" ∧
      result.returned.storage=a.output.allocated.storage ∧
      NativeSafety.Dimension o ax call.result result.returned ∧
      (∀ row inner,row<Checked.outer o ax → inner<Checked.inner o ax →
        mem.read 8 result.state (address result.returned.storage 8 (row*Checked.inner o ax+inner))=
          BitVec.ofNat 64 ((reference (Checked.extent o ax) nonempty
            (fun i : Fin (Checked.extent o ax) =>
              Host.logicalWord mem before x.host
                (row*Checked.extent o ax*Checked.inner o ax+i.val*Checked.inner o ax+inner))).val)) ∧
      (∀ caller,caller∈callers → ∀ point,caller.contains point →
        mem.byte result.state point=mem.byte before point) := by
  have hd : Checked.DimensionContract o ax := by simpa only [branch,normalized] using h.2
  have valid : ax<x.tensor.metadata.shape.length := by
    rw [←observed]
    exact Checked.normalized_valid _ _ _ normalized
  let setup := a.setup mem before x o observed ax valid geometry call
  refine ⟨⟨a.output.returned,Native.dimension mem g a.buffers call.state⟩,
    by simp [dimensionInvoke,(Checked.check_iff o).mpr h],rfl,a.shape,a.output.int64,
    a.output.sameStorage,NativeSafety.dimension x o observed h.1 ax valid hd call prepared a,?_,?_⟩
  · intro row inner hr hi
    rw [Native.dimension_source]
    have readback := HostExecution.dimension_returned mem fp32Scalars before x.host setup row inner hr hi
    simp only [setup,DimensionAllocation.setup,HostExecution.dimensionRun,Output.host,
      AllocatedTensor.host,Nat.zero_add,ContiguousCall.packed,dimOutputOffset,
      NativeGraph.DimensionPrograms.config] at readback
    dsimp only [TensorBinding.Result.state,TensorBinding.Result.returned]
    refine readback.trans ?_
    apply congrArg (BitVec.ofNat 64)
    exact dimension_reference (geometry.config o ax) _ row inner hr nonempty
      (fun tile => g.numeric row inner tile ▸ (g.sites row inner tile).numericComplete)
  · intro caller member point inside
    rw [Native.dimension_source]
    exact HostExecution.dimension_all_caller_frame mem fp32Scalars before x.host setup
      callers call.fresh a.fresh caller member point inside

/-- The global caller starts with the wrapper's initialized scratch-index
buffer and retains its requested scalar conversion throughout both stages. -/
def globalInvoke (mem : Primitives M) (before : M) (scratch : Scratch d) (floating : Bool)
    (o : Checked.Observation) (geometry : NativeGraph.GlobalPrograms d o)
    (g : Native.GlobalPrograms geometry.config) (a : GlobalAllocation x o d callers) :
    Except String (TensorBinding.Result M) :=
  if Checked.check o then .ok ⟨a.output.returned,
    Native.global mem scratch floating g a.buffers
      (Host.initializeIndices mem a.buffers geometry.config.midBlock before)⟩
  else .error "input contract rejected"

theorem global_tensor (mem : Primitives M) (before : M) (scratch : Scratch d) (floating : Bool)
    (x : Input) (o : Checked.Observation) (observed : o.input=x.tensor.metadata)
    (h : Checked.InputContract o) (branch : o.dim=none)
    (geometry : NativeGraph.GlobalPrograms d o) (g : Native.GlobalPrograms geometry.config)
    (a : GlobalAllocation x o d callers)
    (positive : 0<geometry.config.block) (midPositive : 0<geometry.config.midBlock)
    (active : ∀ pid,pid<geometry.config.blocks → pid*geometry.config.block<geometry.config.size) :
    ∃ result,globalInvoke mem before scratch floating o geometry g a=.ok result ∧
      result.returned=a.output.returned ∧
      result.returned.metadata.shape=Host.outputShape x.tensor.metadata.shape none o.keepdim ∧
      result.returned.dtypeName="torch.int64" ∧ result.returned=a.output.allocated ∧
      mem.read 8 result.state (address result.returned.storage 8 0)=
        BitVec.ofNat 64 (globalReference scratch geometry.config positive midPositive
          (Host.logicalWord mem before x.host)) ∧
      (∀ caller,caller∈callers → ∀ point,caller.contains point →
        mem.byte result.state point=mem.byte before point) := by
  have hg : Checked.GlobalContract o := by simpa only [branch] using h.2
  let setup := a.setup mem before x o observed hg geometry
  have native := Native.global_source mem scratch floating g a.buffers
    (Host.initializeIndices mem a.buffers geometry.config.midBlock before) positive active
  have readback := HostExecution.global_returned mem scratch.scalars before x.host setup
  refine ⟨⟨a.output.returned,Native.global mem scratch floating g a.buffers
      (Host.initializeIndices mem a.buffers geometry.config.midBlock before)⟩,
    by simp [globalInvoke,(Checked.check_iff o).mpr h],rfl,a.outputShape,a.output.int64,a.directOutput,?_,?_⟩
  · rw [native]
    change mem.read 8 (GlobalMemory.run mem scratch.scalars geometry.config a.buffers
      (Host.initializeIndices mem a.buffers geometry.config.midBlock before))
      (address a.output.returned.storage 8 0)=_ at readback
    simp only [setup,GlobalAllocation.setup] at readback
    rw [readback,global_reference scratch geometry.config positive midPositive _
      (fun pid _ => g.firstNumeric pid ▸ (g.first pid).numericComplete)
      (g.secondNumeric ▸ g.second.numericComplete)]
  · intro caller member point inside
    rw [native]
    exact HostExecution.global_all_caller_frame mem scratch.scalars before x.host setup callers
      (by intro b hb; simp [setup,GlobalAllocation.setup,direct] at hb)
      a.freshMid a.freshIndex a.freshOutput caller member point inside

#axiomsClean dimension_tensor
#axiomsClean global_tensor
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Tensor
