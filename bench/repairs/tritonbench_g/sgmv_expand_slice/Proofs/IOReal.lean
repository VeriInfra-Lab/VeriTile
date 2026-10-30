import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.PublicIO
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.RealWhole

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section

def realFormula (abi : SourceABI) (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (xs : ChannelValues ℝ (inputChannels p shape segments)) : Nat → Nat → ℝ :=
  mathematicalResult shape (inputValues xs) (weightValues xs) abi.addInputs segments (entryValues xs)

theorem real_implements (rest : Algebra ℝ) (abi : SourceABI)
    (hi : abi.input=.single) (hw : abi.weight=.half) (ho : abi.output=.single)
    (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (separation : SourceGrid.PrivateRegions p)
    (fits : ∀ segment∈segments, SegmentFits shape segment)
    (inner : p.inner=shape.inner) (columns : p.columns=shape.columns) (slice : p.slice=shape.slice)
    (capacity : shape.slice+shape.columns≤shape.outputColumns)
    (layout : GridStride.Contiguous shape.outputRows shape.outputColumns p.outputRowStride p.outputColumnStride) :
    (io (SourceBody.install (RealBackend.algebra rest) RealBackend.dot) abi p shape segments).ImplementsOn
      (metadataDomain p shape segments) (fun xs=>resultValues p shape (realFormula abi p shape segments xs)) := by
  apply implements (RealBackend.algebra rest) RealBackend.dot abi hw ho p shape segments separation fits
    inner columns slice capacity layout _ (fun _ h=>h)
  intro xs hd initial binding
  obtain ⟨final,executed,output,frame⟩ := RealBackend.source_result rest abi p shape segments
    (inputValues xs) (weightValues xs) (entryValues xs) initial
    (binding_inputs _ abi hi hw p shape segments xs initial binding inner columns)
    (binding_output _ abi ho p shape segments xs initial binding)
    (binding_metadata _ abi p shape segments xs hd initial binding)
    separation fits slice capacity layout
  exact ⟨final,executed,output⟩

#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
