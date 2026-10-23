import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Correct
import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Misc.NativeCorrect
import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.Tensor

/-! Optional applications of the complete kernel results to returned Tensor
addresses and byte storage, preserving the original admission and allocation scope. -/
namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.TensorBridge
open VeriTile Triton FP.Structural
open Host ByteMemory Tensor
noncomputable section

/-- The same Real source result at the returned tensor's actual pointer,
with arbitrary read-only input aliases and a fresh output allocation. -/
theorem real_returned_tensor (r : Admission.Request) (checked : Admission.check r=Bool.true)
    (a : Tensor.Binding r) (scale : ℝ) (before : Nat → MemCell) (scratch : BlockState) :
    Nonempty (Kernel.GridLaunchedOrdinary (DirectReal.kernel r.shape scale) (grid r.shape)
      (PhysicalDirect.realInput a.layout before scratch)
      (DirectReal.result r.shape scale (PhysicalDirect.realInput a.layout before scratch))) ∧
    (∀ h t v,PhysicalDirect.readReal
      (PhysicalDirect.realResult r.shape a.layout scale before scratch
        (a.tensor.byteBase/4+(h.val*(r.shape.T*r.shape.V)+t.val*r.shape.V+v.val)))=
      Host.formula r.shape (Admission.block_bounds r.shape ((Admission.check_iff r).mp checked).1).1
        (VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real.inputs regions
          (PhysicalDirect.realInput a.layout before scratch)) scale h t v) ∧
    (∀ p,¬(PhysicalDirect.output a.layout).Contains p →
      PhysicalDirect.realResult r.shape a.layout scale before scratch p=before p) ∧
    (∀ input∈(["q","k","v","h","g"] : List RegionName),∀ i,i<capacity r.shape input →
      PhysicalDirect.realResult r.shape a.layout scale before scratch (a.layout.base input+i)=
        before (a.layout.base input+i)) := by
  have result:=PhysicalDirect.real_complete r.shape
    (Admission.block_bounds r.shape ((Admission.check_iff r).mp checked).1).1 a.layout scale before scratch
  have pointer : a.tensor.byteBase/4=a.layout.base "o" := by rw [a.outputPointer]; omega
  simpa only [pointer] using result

theorem native_returned_tensor (r : Admission.Request) (checked : Admission.check r=Bool.true)
    (a : Binding r) (before : Memory) (scratch : State NativeGraph.Expr)
    (I : ScalarGraph.WordPrimitives) :
    FPGrid.run ScalarGraph.algebra r.shape ScalarGraph.scaleTag (FPGrid.programs r.shape)
      (PhysicalDirect.fpInput a.layout (words before) scratch)=some (Tensor.source r a before scratch) ∧
    (∀ nonempty : r.shape.heads*r.shape.T*r.shape.V≠0,
      (Tensor.source r a before scratch).mem=(NativeABI.grid ScalarGraph.algebra r.shape (ScalarGraph.dots r.shape)
        (Admission.accepted r checked nonempty) ScalarGraph.scaleTag (FPGrid.programs r.shape)
        (PhysicalDirect.fpInput a.layout (words before) scratch)).mem) ∧
    (result r a before scratch I).objectId=a.layout.objectId ∧
    (result r a before scratch I).tensor=a.tensor ∧
    (∀ h t v,read (result r a before scratch I).bytes
      (a.tensor.byteBase+4*(h.val*(r.shape.T*r.shape.V)+t.val*r.shape.V+v.val))=
      (ScalarGraph.value r.shape (PhysicalDirect.fpInput a.layout (words before) scratch) r.scale.runtimeWord h t v
        (Admission.block_bounds r.shape ((Admission.check_iff r).mp checked).1).1).eval I.isa) ∧
    (∀ p,p<a.tensor.byteBase ∨ a.tensor.byteBase+4*a.tensor.shape.prod≤p →
      (result r a before scratch I).bytes p=before p) ∧
    (r.shape.heads*r.shape.T*r.shape.V≠0 → ∀ (input : Input) i,
      ((PhysicalDirect.fpInput a.layout (words before) scratch).mem input.region i).read .real=
        .leaf (read before ((input.tensor r).byteBase+4*i))) ∧
    (r.shape.heads*r.shape.T*r.shape.V≠0 → ∀ (input : Input) i,i<capacity r.shape input.region → ∀ j : Fin 4,
      (result r a before scratch I).bytes ((input.tensor r).byteBase+4*i+j.val)=
        before ((input.tensor r).byteBase+4*i+j.val)) :=
  Tensor.complete r checked a before scratch I

theorem empty_tensor (r : Admission.Request) (a : Binding r) (before : Memory)
    (scratch : State NativeGraph.Expr) (I : ScalarGraph.WordPrimitives)
    (h : r.shape.heads*r.shape.T*r.shape.V=0) :
    (result r a before scratch I).bytes=before :=
  Tensor.empty r a before scratch I h

#axiomsClean real_returned_tensor
#axiomsClean native_returned_tensor
#axiomsClean empty_tensor
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.TensorBridge
