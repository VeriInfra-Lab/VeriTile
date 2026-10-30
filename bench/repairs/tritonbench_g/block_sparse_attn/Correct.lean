import bench.repairs.tritonbench_g.block_sparse_attn.Contract
import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.Host
import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.Bounds
import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.Reduction
import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.RealComplete
import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.SourceBinding
namespace VeriTile.Bench.Repairs.BlockSparseAttn
noncomputable section
structure WholeReal (x : Layout) (m : Memory ℝ) (scale : ℝ) : Prop where
  output : ∀ w : Work,w.active x.shape →
    execute x Real.ops (some scale) (Source.work x.shape) (Real.lift m) .o (w.target x.shape)=
      some (Spec.output x.shape x.csr (Real.inputs x m) scale w.batch w.head w.row w.col)
  frame : ∀ r a,¬footprint x.shape r a →
    execute x Real.ops (some scale) (Source.work x.shape) (Real.lift m) r a=some (m r a)

theorem whole_real (a : Host.Arguments) (valid : Host.Contract a) (m : Memory ℝ) (scale : ℝ) :
    WholeReal a.layout m scale := by
  exact ⟨Real.whole_real a.layout m scale _ (Host.complete a valid) valid.alignedD valid.alignedN,
    call_frame a.layout Real.ops (some scale) _ (Host.complete a valid) (Real.lift m)⟩

#axiomsClean whole_real
#specNonCircular Spec.output avoiding [execute,rowValue,blockStep,normalizedStep,Source.work]

namespace CheckedReal
open Physical Wrapper
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory

def source (a : Host.Arguments) (binding : SourceBinding.Binding a.layout Real.ops) (scale : ℝ) (m : Memory ℝ) (w : Work) : ℝ :=
  (SourceBinding.value binding (Real.lift m) (some scale) w).getD 0

def target (a : Host.Arguments) (scale : ℝ) (m : Memory ℝ) (w : Work) : ℝ :=
  Spec.output a.layout.shape a.layout.csr (Real.inputs a.layout m) scale w.batch w.head w.row w.col

theorem source_target (a : Host.Arguments) (binding : SourceBinding.Binding a.layout Real.ops) (header : Host.Header a) (scale : ℝ)
    (m : Memory ℝ) (q : Coord a.layout.shape) : source a binding scale m q.val=target a scale m q.val := by
  unfold source target
  rw [SourceBinding.value_eq,Real.row_real a.layout m scale q.val q.property.2.2.1 header.alignedD header.alignedN]
  rfl

/-- Checked allocation/empty/launch result with dynamic physical stores and
the independent occurrence-multiset causal mathematical formula. -/
theorem checked_call (api : API ℝ Store Byte) (a : Host.Arguments) (out : Host.Tensor)
    (buffers : Buffers) (rowHeader colHeader : MetadataHeader) (rows cols : Nat) (scale : ℝ)
    (binding : SourceBinding.Binding a.layout Real.ops)
    (launch : Bool) (accepted : validate a out buffers rowHeader colHeader rows cols=.ok launch)
    (order : List (Coord a.layout.shape)) (complete : ∀ q,q∈order) (m : Store) :
    ∃ after,SourceBinding.selectedDispatch api 0 a out buffers rowHeader colHeader rows cols binding.toStoreMap (source a binding scale) order m=.ok (out,after) ∧
      ((launch=Bool.false ∧ after=m) ∨ (launch=Bool.true ∧
        (∀ q : Coord a.layout.shape,api.read after (Physical.outputAddress buffers a.layout.shape q.val)=
          target a scale (Physical.inputs api 0 buffers m) q.val) ∧
        (∀ i,Outside buffers a.layout.shape order i → api.byte after i=api.byte m i))) := by
  rw [SourceBinding.dispatch_eq]
  cases launch
  · exact ⟨m,dispatch_empty _ _ _ _ _ _ _ _ _ _ _ _ accepted,Or.inl ⟨rfl,rfl⟩⟩
  · have live:=(validate_launch.mp accepted).2
    refine ⟨_,dispatch_launch _ _ _ _ _ _ _ _ _ _ _ _ accepted,Or.inr ⟨rfl,?_,?_⟩⟩
    · intro q
      rw [Physical.readback api 0 live.physical _ order m q (complete q),source_target a binding live.physical.header]
    · exact Physical.byte_frame api 0 live.physical _ order m

#axiomsClean checked_call
#auditStmt checked_call
#specNonCircular target avoiding [Physical.execute,dispatch,source,rowValue]
end CheckedReal
end
end VeriTile.Bench.Repairs.BlockSparseAttn
