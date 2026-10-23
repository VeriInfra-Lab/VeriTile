import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.BoundDimension
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.DefaultBinding

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.NativeKernelResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Memory
noncomputable section

/-- Complete physical execution of both quotations, the independent staged
requested-format target and every byte outside the scratch/output stores. -/
theorem global (mem : Primitives M) (casts : Request.Casts) (kind : Request.Kind)
    (binding : BoundSource.Binding casts kind) (rest : Algebra Word) (template : State Word)
    (c : GlobalConfig) (g : Native.GlobalPrograms c) (p : GlobalMemory.Buffers)
    (layout : GlobalMemory.Layout kind.format c p) (positive : 0<c.block)
    (midPositive : 0<c.midBlock) (active : ∀ pid,pid<c.blocks → pid*c.block<c.size)
    (index64 : Bool) (before : M)
    (initialized : ∀ i,i<c.midBlock → mem.read 8 before (address p.midIndex 8 i)=0) :
    BoundSource.GlobalRun mem binding rest template c p index64 before
      (Native.global mem (Request.scratch casts kind) kind.isFloating g p before) ∧
    mem.read 8 (Native.global mem (Request.scratch casts kind) kind.isFloating g p before)
      (address p.output 8 0)=BitVec.ofNat 64
        (globalReference (Request.scratch casts kind) c positive midPositive (GlobalMemory.input mem p before)) ∧
    (∀ point : Address,
      (∀ pid,pid<c.blocks → Disjoint (address p.midValue kind.format.bytes pid) kind.format.bytes point 1) →
      (∀ pid,pid<c.blocks → Disjoint (address p.midIndex 8 pid) 8 point 1) →
      Disjoint (address p.output 8 0) 8 point 1 →
      mem.byte (Native.global mem (Request.scratch casts kind) kind.isFloating g p before) point=mem.byte before point) := by
  exact ⟨BoundSource.global_run mem binding rest template c g p layout positive midPositive active index64 before,
    Result.global_readback mem (Request.scratch casts kind) kind.isFloating c g p layout before positive midPositive active initialized,
    fun point mid index out => Result.global_frame mem (Request.scratch casts kind) kind.isFloating c g p before positive active point mid index out⟩

/-- Complete dimension-argmax source grid, the first-NaN/first-maximum
input-only index at every output, and the complete exterior byte frame. -/
theorem dimension (mem : Primitives M) (rest : Algebra Word) (template : State Word)
    (c : DimConfig) (g : Native.DimPrograms c) (p : DimMemory.Buffers)
    (layout : DimMemory.Layout c p) (nonempty : 0<c.extent) (index64 : Bool) (before : M) :
    BoundSource.DimensionRun mem rest template c p index64 before (Native.dimension mem g p before) ∧
    (∀ row inner,row<c.outer → inner<c.inner →
      mem.read 8 (Native.dimension mem g p before) (address p.output 8 (dimOutputOffset c row inner))=
        BitVec.ofNat 64 ((reference c.extent nonempty
          (fun i : Fin c.extent => rowInput c (DimMemory.input mem p before) row inner i.val)).val)) ∧
    (∀ point : Address,
      (∀ row inner,row<c.outer → inner<c.inner →
        Disjoint (address p.output 8 (dimOutputOffset c row inner)) 8 point 1) →
      mem.byte (Native.dimension mem g p before) point=mem.byte before point) := by
  exact ⟨BoundSource.dimension_run mem rest template c g p layout nonempty index64 before,
    fun row inner hr hi => Result.dimension_readback mem c g p layout before nonempty row inner hr hi,
    fun point apart => Result.dimension_frame mem c g p layout before point apart⟩

/-- The default FP32 memory execution returns the input-only first-NaN/first-
maximum target. Other requested dtypes use global's staged target. -/
theorem default_readback (mem : Primitives M) (c : GlobalConfig) (coverage : Math.GlobalCoverage c)
    (g : Native.GlobalPrograms c) (p : GlobalMemory.Buffers)
    (layout : GlobalMemory.Layout RealScratch.FloatingKind.fp32.format c p) (s : M)
    (initialized : ∀ i,i<c.midBlock → mem.read 8 s (address p.midIndex 8 i)=0) :
    mem.read 8 (Native.global mem (floatingScratch .fp32 id) Bool.true g p s) (address p.output 8 0)=
      BitVec.ofNat 64 ((reference c.size coverage.sizePositive
        (fun i : Fin c.size => GlobalMemory.input mem p s i.val)).val) := by
  rw [Result.global_readback mem (floatingScratch .fp32 id) Bool.true c g p layout s
    coverage.blockPositive (coverage.blocksPositive.trans_le coverage.midFits) coverage.eachActive initialized,
    Default.staged_reference c coverage]

#axiomsClean global
#axiomsClean dimension
#axiomsClean default_readback
#specNonCircular globalReference avoiding [Native.global,GlobalMemory.run]
#specNonCircular reference avoiding [Native.dimension,DimMemory.run]
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.NativeKernelResult
