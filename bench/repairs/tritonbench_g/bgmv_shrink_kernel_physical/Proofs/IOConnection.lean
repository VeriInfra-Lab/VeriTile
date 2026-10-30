import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.IOOldRun
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.RealTensor
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.Physical

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOConnection
open VeriTile.Triton FP.Structural FPValues NativeGraph
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
noncomputable section

/-- The same old Real runtime result returned by the Tensor bridge is an
endpoint of complete actual source traces in the shared IO semantics. -/
theorem real_trace (h : Call) (u : BlockState) (indices : Nat → Int)
    (valid : ∀ b, b < h.batches → indices b = -1 ∨ 0 ≤ indices b)
    (orders : Orders h) (run : Real.Run h orders u indices) (b : Fin h.batches) (n : Fin h.N) :
    let data := Real.data h u indices
    let io := IO.programIO Real.algebra Real.primitives h indices h.scaling orders id
    let vals := IOProjection.values h data
    io.SourceCellTrace vals (entry h 0 0 data)
      (IOOldRun.initial Real.algebra Real.primitives h orders id data) ⟨0,(b,n,PUnit.unit)⟩
      (io.order vals 0 (b,n,PUnit.unit))
      (ofFloat h.outputType (u.readMem "Y" (outputOffset h b.val n.val)))
      (ofFloat h.outputType (run.output b n)) :=
  IOOldRun.source_run Real.algebra Real.primitives h orders id (Real.data h u indices) valid run b n

/-- The exact old byte observation stores the endpoint of the new source
trace, initialized from the same caller bytes and runtime scale word. -/
theorem native_physical (plans : ScalarTree.Plans) (S : Selection) (I : Instructions)
    (binding : CompilerBinding plans S) (h : Call) (orders : Orders h)
    (a : Native.Binding x w y i s) (before : Native.Raw.State)
    (run : Native.Run plans h orders a before)
    (valid : ∀ b,b<h.batches → (Native.data a before).indices b=-1 ∨ 0≤(Native.data a before).indices b)
    (c : Native.Cell h) (active : (Native.data a before).indices c.1.val≠-1) :
    let data := Native.data a before
    let observe := fun term => (binding.program term).eval I
    let io := IO.programIO algebra (primitives plans) h data.indices data.scaling orders observe
    let vals := IOProjection.values h data
    io.SourceCellTrace vals (entry h 0 0 data)
      (IOOldRun.initial algebra (primitives plans) h orders observe data) ⟨0,(c.1,c.2,PUnit.unit)⟩
      (io.order vals 0 (c.1,c.2,PUnit.unit))
      (ofFloat h.outputType (data.output (outputOffset h c.1.val c.2.val)))
      (ofFloat h.outputType (run.output c.1 c.2)) ∧
    (∀ byte : Fin 4,
      Native.sourceBytes binding I h orders a before run a.output.window.owner
        ((a.output.window.key (Native.address h c)).address+byte.val) =
      Native.Raw.unpack (observe (toFloat h.outputType (IOImplements.results algebra (primitives plans)
        h data.indices data.scaling orders observe (IOProjection.run algebra (primitives plans) h orders data run)
          ⟨0,(c.1,c.2,PUnit.unit)⟩))) byte) := by
  dsimp only
  constructor
  · exact IOOldRun.source_run algebra (primitives plans) h orders
      (fun term => (binding.program term).eval I) (Native.data a before) valid run c.1 c.2
  · intro byte
    trans Native.Raw.unpack (Native.Raw.read (Native.sourceBytes binding I h orders a before run)
      (a.output.window.key (Native.address h c))) byte
    · exact Native.Raw.byte_of_read _ _ byte
    rw [Native.source_read binding I h orders a before run c active]
    change Native.Raw.unpack ((binding.program (run.output c.1 c.2)).eval I) byte =
      Native.Raw.unpack ((binding.program (toFloat h.outputType (ofFloat h.outputType (run.output c.1 c.2)))).eval I) byte
    rw [IOPrograms.to_from_float]

#axiomsClean real_trace
#axiomsClean native_physical
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.IOConnection
