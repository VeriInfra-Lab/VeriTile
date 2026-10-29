import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.IONative

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.FPEquiv
open VeriTile.Triton FP.Structural NativeBackend
noncomputable section

/-- The same complete jobs and regional IO under the original native model:
FP32 input/output, FP16 weights, source dot/conversion/add order and metadata ABI. -/
def sgmvNativeIO (rest : Algebra Raw) (ops : Native.Primitives) (abi : SourceABI)
    (p : KernelArgs) (shape : Shape) (segments : List Segment) : ReadProgramIO Raw Raw where
  inputs := PublicIO.inputChannels p shape segments
  outputs := PublicIO.outputChannels p shape
  scratch := []
  buffers := [p.input,p.weight,p.output,p.starts,p.lengths,p.adapters]
  read := PublicIO.inputRead p shape segments
  inputMask := PublicIO.inputMask p shape segments
  write := PublicIO.outputWrite p shape segments
  outputMask := PublicIO.outputMask p shape segments
  scratchWrite := fun k=>k.elim0
  scratchMask := fun k=>k.elim0
  schedule := fun _=>(jobs shape.columns segments).map
    (PublicIO.invocation (SourceBody.install (algebra rest ops (profile abi))
      (dot ops (weight abi.weight))) abi p)
  encode := id
  observe := fun _ _=>id
  resultRead := PublicIO.outputWrite p shape segments
  resultMask := fun _ _ _=>True

/-- Independent native instruction recurrence. No native-to-Real identity or
new optimization is asserted. Every source-format word is admitted. -/
def sgmvWordFormula (ops : Native.Primitives) (abi : SourceABI) (p : KernelArgs)
    (shape : Shape) (segments : List Segment)
    (xs : ChannelValues Raw (PublicIO.inputChannels p shape segments)) :=
  PublicIO.resultValues p shape (PublicIO.nativeFormula ops abi p shape segments xs)

theorem kernel_native (rest : Algebra Raw) (ops : Native.Primitives) (zeros : ZeroConversions ops)
    (abi : SourceABI) (inputType : abi.input=.single)
    (weightType : abi.weight=.half) (outputType : abi.output=.single)
    (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (separation : SourceGrid.PrivateRegions p)
    (fits : ∀ segment∈segments, SegmentFits shape segment)
    (inner : p.inner=shape.inner) (columns : p.columns=shape.columns) (slice : p.slice=shape.slice)
    (capacity : shape.slice+shape.columns≤shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride) :
    (sgmvNativeIO rest ops abi p shape segments).ImplementsOn
      (PublicIO.nativeDomain p shape segments) (sgmvWordFormula ops abi p shape segments) := by
  exact PublicIO.native_implements rest ops zeros abi inputType weightType outputType p shape segments
    separation fits inner columns slice capacity layout

#axiomsClean kernel_native
#specNonCircular sgmvWordFormula avoiding [runSourceJobs,FP.Structural.run,FP.Structural.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.FPEquiv
