import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.RealKernelResult
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.PhysicalConnection
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.BoundDimension
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.Request
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedNative
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedIntegerMacro
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedDimensionBinding
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.DefaultBinding

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TensorBridge
open VeriTile.Bench.Repairs.TritonArgmax Source Memory TensorBinding
noncomputable section

/-- The unchanged admission supplies all scalar launch-geometry facts. The
requested scratch format remains selected by the actual dtype argument. -/
theorem global_graph_tensor (mem : Primitives M) (before : M) (casts : Request.Casts)
    (x : Input) (o : Checked.Observation) (observed : o.input=x.tensor.metadata)
    (h : Checked.InputContract o) (branch : o.dim=none)
    (kind : Request.Kind) (requested : Request.select o.dtype=some kind)
    (geometry : NativeGraph.GlobalPrograms kind.format o) (g : Native.GlobalPrograms geometry.config)
    (a : GlobalAllocation x o kind.format callers) :
    ∃ result,Tensor.globalInvoke mem before (Request.scratch casts kind) kind.isFloating o geometry g a=.ok result ∧
      result.returned=a.output.returned ∧
      result.returned.metadata.shape=Host.outputShape x.tensor.metadata.shape none o.keepdim ∧
      result.returned.dtypeName="torch.int64" ∧ result.returned=a.output.allocated ∧
      ∃ positive : 0<geometry.config.block,∃ midPositive : 0<geometry.config.midBlock,
      mem.read 8 result.state (address result.returned.storage 8 0)=
        BitVec.ofNat 64 (globalReference (Request.scratch casts kind) geometry.config positive midPositive
          (Host.logicalWord mem before x.host)) ∧
      (∀ caller,caller∈callers → ∀ point,caller.contains point →
        mem.byte result.state point=mem.byte before point) ∧
      o.dtype.adapter=some kind.adapter := by
  have hg : Checked.GlobalContract o := by simpa only [branch] using h.2
  have coverage : Math.GlobalCoverage geometry.config := Checked.global_coverage o hg _ _
    (fun pid _ => (geometry.first pid).covers) geometry.second.covers
  obtain ⟨result,run,returned,shape,dtype,direct,readback,frame⟩ :=
    Tensor.global_tensor mem before (Request.scratch casts kind) kind.isFloating x o observed h branch
      geometry g a coverage.blockPositive (coverage.blocksPositive.trans_le coverage.midFits) coverage.eachActive
  exact ⟨result,run,returned,shape,dtype,direct,coverage.blockPositive,
    coverage.blocksPositive.trans_le coverage.midFits,readback,frame,Request.selected_adapter _ _ requested⟩

theorem dimension_graph_tensor (mem : Primitives M) (before : M)
    (x : Input) (o : Checked.Observation) (observed : o.input=x.tensor.metadata)
    (h : Checked.InputContract o) (dim : Int) (branch : o.dim=some dim)
    (ax : Nat) (normalized : Host.normalizeDim o.input.shape.length dim=some ax)
    (geometry : NativeGraph.DimensionPrograms) (g : Native.DimPrograms (geometry.config o ax))
    (call : ContiguousCall mem before x callers) (prepared : o.prepared=call.result.tensor.metadata)
    (a : DimensionAllocation x o ax call.result callers) :
    ∃ result,Tensor.dimensionInvoke mem before o ax geometry g call a=.ok result ∧
      result.returned=a.output.returned ∧
      result.returned.metadata.shape=Host.outputShape x.tensor.metadata.shape (some ax) o.keepdim ∧
      result.returned.dtypeName="torch.int64" ∧
      result.returned.storage=a.output.allocated.storage ∧
      NativeSafety.Dimension o ax call.result result.returned ∧
      ∃ nonempty : 0<Checked.extent o ax,
      (∀ row inner,row<Checked.outer o ax → inner<Checked.inner o ax →
        mem.read 8 result.state (address result.returned.storage 8 (row*Checked.inner o ax+inner))=
          BitVec.ofNat 64 ((reference (Checked.extent o ax) nonempty
            (fun i : Fin (Checked.extent o ax) => Host.logicalWord mem before x.host
              (row*Checked.extent o ax*Checked.inner o ax+i.val*Checked.inner o ax+inner))).val)) ∧
      (∀ caller,caller∈callers → ∀ point,caller.contains point →
        mem.byte result.state point=mem.byte before point) := by
  have hd : Checked.DimensionContract o ax := by simpa only [branch,normalized] using h.2
  obtain ⟨result,run,returned,shape,dtype,storage,safe,readback,frame⟩ :=
    Tensor.dimension_tensor mem before x o observed h dim branch ax normalized geometry g call prepared a hd.1
  exact ⟨result,run,returned,shape,dtype,storage,safe,hd.1,readback,frame⟩

/-- Public source-backed certificate. Every quoted launch reads the current
physical memory, commits its scalar stores into the same state, and reaches
exactly the returned Tensor memory used by the independent target and frame. -/
theorem global_tensor (mem : Primitives M) (before : M) (casts : Request.Casts)
    (x : Input) (o : Checked.Observation) (observed : o.input=x.tensor.metadata)
    (h : Checked.InputContract o) (branch : o.dim=none)
    (kind : Request.Kind) (requested : Request.select o.dtype=some kind)
    (binding : BoundSource.Binding casts kind)
    (rest : VeriTile.Triton.FP.Structural.Algebra Word) (template : VeriTile.Triton.FP.Structural.State Word)
    (geometry : NativeGraph.GlobalPrograms kind.format o) (g : Native.GlobalPrograms geometry.config)
    (a : GlobalAllocation x o kind.format callers) :
    ∃ result,BoundSource.GlobalRun mem binding rest template geometry.config a.buffers
      (decide (2^31≤o.input.numel))
      (Host.initializeIndices mem a.buffers geometry.config.midBlock before) result.state ∧
      Tensor.globalInvoke mem before (Request.scratch casts kind) kind.isFloating o geometry g a=.ok result ∧
      result.returned=a.output.returned ∧
      result.returned.metadata.shape=Host.outputShape x.tensor.metadata.shape none o.keepdim ∧
      result.returned.dtypeName="torch.int64" ∧ result.returned=a.output.allocated ∧
      ∃ positive : 0<geometry.config.block,∃ midPositive : 0<geometry.config.midBlock,
      mem.read 8 result.state (address result.returned.storage 8 0)=
        BitVec.ofNat 64 (globalReference (Request.scratch casts kind) geometry.config positive midPositive
          (Host.logicalWord mem before x.host)) ∧
      (∀ caller,caller∈callers → ∀ point,caller.contains point →
        mem.byte result.state point=mem.byte before point) ∧
      o.dtype.adapter=some kind.adapter := by
  have hg : Checked.GlobalContract o := by simpa only [branch] using h.2
  have coverage : Math.GlobalCoverage geometry.config := Checked.global_coverage o hg _ _
    (fun pid _ => (geometry.first pid).covers) geometry.second.covers
  let setup := a.setup mem before x o observed hg geometry
  obtain ⟨result,invoke,post⟩ := global_graph_tensor mem before casts x o observed h branch
    kind requested geometry g a
  have state : result.state=Native.global mem (Request.scratch casts kind) kind.isFloating g a.buffers
      (Host.initializeIndices mem a.buffers geometry.config.midBlock before) := by
    have same : (⟨a.output.returned,Native.global mem (Request.scratch casts kind) kind.isFloating g a.buffers
        (Host.initializeIndices mem a.buffers geometry.config.midBlock before)⟩ : TensorBinding.Result M)=result := by
      simpa [Tensor.globalInvoke,(Checked.check_iff o).mpr h] using invoke
    exact congrArg TensorBinding.Result.state same.symm
  refine ⟨result,?_,invoke,post⟩
  rw [state]
  exact BoundSource.global_run mem binding rest template geometry.config g a.buffers setup.layout
    coverage.blockPositive (coverage.blocksPositive.trans_le coverage.midFits) coverage.eachActive _ _

theorem dimension_tensor (mem : Primitives M) (before : M)
    (rest : VeriTile.Triton.FP.Structural.Algebra Word) (template : VeriTile.Triton.FP.Structural.State Word)
    (x : Input) (o : Checked.Observation) (observed : o.input=x.tensor.metadata)
    (h : Checked.InputContract o) (dim : Int) (branch : o.dim=some dim)
    (ax : Nat) (normalized : Host.normalizeDim o.input.shape.length dim=some ax)
    (geometry : NativeGraph.DimensionPrograms) (g : Native.DimPrograms (geometry.config o ax))
    (call : ContiguousCall mem before x callers) (prepared : o.prepared=call.result.tensor.metadata)
    (a : DimensionAllocation x o ax call.result callers) :
    ∃ result,BoundSource.DimensionRun mem rest template (geometry.config o ax) a.buffers
      (decide (2^31≤call.result.tensor.metadata.numel)) call.state result.state ∧
      Tensor.dimensionInvoke mem before o ax geometry g call a=.ok result ∧
      result.returned=a.output.returned ∧
      result.returned.metadata.shape=Host.outputShape x.tensor.metadata.shape (some ax) o.keepdim ∧
      result.returned.dtypeName="torch.int64" ∧
      result.returned.storage=a.output.allocated.storage ∧
      NativeSafety.Dimension o ax call.result result.returned ∧
      ∃ nonempty : 0<Checked.extent o ax,
      (∀ row inner,row<Checked.outer o ax → inner<Checked.inner o ax →
        mem.read 8 result.state (address result.returned.storage 8 (row*Checked.inner o ax+inner))=
          BitVec.ofNat 64 ((reference (Checked.extent o ax) nonempty
            (fun i : Fin (Checked.extent o ax) => Host.logicalWord mem before x.host
              (row*Checked.extent o ax*Checked.inner o ax+i.val*Checked.inner o ax+inner))).val)) ∧
      (∀ caller,caller∈callers → ∀ point,caller.contains point →
        mem.byte result.state point=mem.byte before point) := by
  have hd : Checked.DimensionContract o ax := by simpa only [branch,normalized] using h.2
  have valid : ax<x.tensor.metadata.shape.length := by
    rw [←observed]
    exact Checked.normalized_valid _ _ _ normalized
  let setup := a.setup mem before x o observed ax valid geometry call
  obtain ⟨result,invoke,post⟩ := dimension_graph_tensor mem before x o observed h dim branch ax normalized
    geometry g call prepared a
  have state : result.state=Native.dimension mem g a.buffers call.state := by
    have same : (⟨a.output.returned,Native.dimension mem g a.buffers call.state⟩ : TensorBinding.Result M)=result := by
      simpa [Tensor.dimensionInvoke,(Checked.check_iff o).mpr h] using invoke
    exact congrArg TensorBinding.Result.state same.symm
  refine ⟨result,?_,invoke,post⟩
  rw [state]
  exact BoundSource.dimension_run mem rest template (geometry.config o ax) g a.buffers setup.layout hd.1 _ call.state

#axiomsClean global_tensor
#axiomsClean dimension_tensor
#axiomsClean global_graph_tensor
#axiomsClean dimension_graph_tensor
#specNonCircular globalReference avoiding [Native.global,Tensor.globalInvoke,GlobalMemory.run]
#specNonCircular reference avoiding [Native.dimension,Tensor.dimensionInvoke,DimMemory.run]
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TensorBridge
