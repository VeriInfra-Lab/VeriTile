import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Correct
import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.RealExecution
import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.Tensor

/-! Complete Tensor applications of the raw source/grid certificates. The existing
admission, finite-byte execution, return identity and device observations remain exact. -/
namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.TensorBridge
open VeriTile.Triton FP.Structural FusedRotaryEmbedding Checked
noncomputable section
set_option maxHeartbeats 4000000

structure Result where
  value : Option Unit
  tensors : Region → Checked.Tensor
  deviceRestored : Nat
  memory : Source.Mem ℝ

def invokeReal (rest : Algebra ℝ) (c : Call) (callerDevice : Nat) (metadata : Physical.Bytes)
    (initial : Memory ℝ) : Option Result :=
  if inputChecks c metadata then
    (Source.grid rest realOps (Source.parameters c.shape) (programs c.shape)
      (Source.lift (host c metadata).config initial)).map (fun final=>⟨none,c.tensor,callerDevice,final⟩)
  else none

/-- Exact-real projection of the complete shared quotation, including Q
rotation, both K-cache layouts, raw V copying, empty return and cell frame. -/
theorem fused_rotary_real (rest : Algebra ℝ) (c : Call) (callerDevice : Nat)
    (metadata : Physical.Bytes) (initial : Memory ℝ) (admitted : inputChecks c metadata=Bool.true) :
    ∃ result,invokeReal rest c callerDevice metadata initial=some result ∧ result.value=none ∧
      result.tensors=c.tensor ∧ result.deviceRestored=callerDevice ∧
      (∀ kind t h d,t<c.shape.tokens → h<c.shape.heads kind → d<c.shape.width →
        (result.memory (Source.region kind.output) (logicalOutput (host c metadata).config kind t h d)).read .real=
          formula c.shape initial kind t h d) ∧
      (∀ r a,¬Footprint (host c metadata).config (Source.decoded r) a →
        result.memory r a=Source.lift (host c metadata).config initial r a) ∧
      (c.shape.tokens=0 → result.memory=Source.lift (host c metadata).config initial) ∧
      (0<c.shape.tokens → 0<c.shape.halfBlock ∧ c.shape.halfBlock≤1048576 ∧
        0<c.shape.fullBlock ∧ c.shape.fullBlock≤1048576) := by
  have valid:=config_valid c metadata (original_checks c metadata admitted)
  obtain ⟨final,executed,values,frame,_⟩:=RealExecution.fused_rotary_real rest (host c metadata).config valid initial
  change Source.grid rest realOps (Source.parameters c.shape) (programs c.shape)
    (Source.lift (host c metadata).config initial)=some final at executed
  refine ⟨⟨none,c.tensor,callerDevice,final⟩,?_,rfl,rfl,rfl,values,frame,?_,arange_bounds c metadata admitted⟩
  · simp [invokeReal,admitted,executed]
  · intro empty
    have noPrograms : programs c.shape=[] := by simp [programs,empty]
    rw [noPrograms,Source.grid] at executed
    exact Option.some.inj executed.symm

/-- Candidate admission retains the original empty behavior and imposes the
frontend tile bound only for live calls. -/
def invokeNative (c : Call) (callerDevice : Nat) (p : Native.Instructions) (initial : Arena.Memory) :
    Option Binding.Result :=
  if inputChecks c (Arena.mirror c initial) then Binding.invoke c callerDevice p initial else none

/-- Complete source execution, independently selected native scalar graph,
finite bytes, actual Tensor logical reads, input observations and frames.
Instruction semantics and compiler profile realization are local boundaries. -/
theorem fused_rotary_native (rest : Algebra NativeSource.Payload) (c : Call)
    (family : TensorObservation.Family c) (callerDevice : Nat) (p : Native.Instructions)
    (initial : Arena.Memory) (admitted : inputChecks c (Arena.mirror c initial)=Bool.true) :
    ∃ final result,
      Source.grid rest (NativeSource.ops p) (Source.parameters c.shape) (programs c.shape)
        (Source.lift (host c (Arena.mirror c initial)).config
          (NativeSource.initial (Physical.load (Arena.mirror c initial))))=some final ∧
      invokeNative c callerDevice p initial=some result ∧
      Binding.NativeResult c callerDevice p p initial result ∧
      (∀ kind t h d,t<c.shape.tokens → h<c.shape.heads kind → d<c.shape.width →
        TensorObservation.logical c family result.memory kind.output (TensorObservation.output_payload kind)
          (logicalOutput (host c (Arena.mirror c initial)).config kind t h d)=
            wordFormula c.shape p (Physical.load (Arena.mirror c initial)) kind t h d) ∧
      (∀ kind t h d,t<c.shape.tokens → h<c.shape.heads kind → d<c.shape.width →
        TensorObservation.logical c family result.memory kind.output (TensorObservation.output_payload kind)
          (logicalOutput (host c (Arena.mirror c initial)).config kind t h d)=
            ((final (Source.region kind.output)
              (logicalOutput (host c (Arena.mirror c initial)).config kind t h d)).read .real kind).value) ∧
      (∀ r (hr : TensorObservation.payloadRegion r) a,a<extent c r →
        TensorObservation.logical c family initial r hr a=Physical.load (Arena.mirror c initial) r a) ∧
      (∀ r (hr : TensorObservation.payloadRegion r) a,a<extent c r →
        (∀ i : Fin 4,Arena.Outside c (Arena.mirror c initial) (Arena.address c r (4*a+i.val))) →
        TensorObservation.logical c family result.memory r hr a=TensorObservation.logical c family initial r hr a) ∧
      (0<c.shape.tokens → 0<c.shape.halfBlock ∧ c.shape.halfBlock≤1048576 ∧
        0<c.shape.fullBlock ∧ c.shape.fullBlock≤1048576) := by
  obtain ⟨final,result,executed,invoked,properties,sourceWords⟩:=
    NativeSource.complete_bytes rest c callerDevice p initial admitted
  refine ⟨final,result,executed,by simp [invokeNative,admitted,invoked],properties,
    TensorObservation.outputs c family initial admitted callerDevice p result properties,?_,
    TensorObservation.inputs c family initial admitted,
    TensorObservation.frame c family initial admitted callerDevice p result properties,
    arange_bounds c _ admitted⟩
  intro kind t h d ht hh hd
  have bound:=Binding.logical_extent c _ (original_checks c _ admitted) kind t h d ht hh hd
  rw [sourceWords,TensorObservation.logical_read c family result.memory _
    (original_checks c _ admitted) _ _ _ bound,TensorObservation.mirror_read c result.memory _ _ bound]

#axiomsClean fused_rotary_real
#axiomsClean fused_rotary_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.TensorBridge
