import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Misc.RawCorrect
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.RawNative
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.TensorView

/-! Complete Tensor applications of the raw kernel interfaces, with the original
admission domain, output coordinates, metadata and byte frames. -/
namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.TensorBridge
open VeriTile.Triton FP.Structural Execution TensorView
open LlamaFF.Contract
noncomputable section

def RealPost (x w : TensorInfo) (view : Output x w) (initial : Words.Memory)
    (final : State ℝ) : Prop :=
  (∀ c : Logical x w,(realResult view final).get c=
    Real.target (Checked.parameters x w) (realEntry (Checked.parameters x w) initial)
      (index x w c).1.val (index x w c).2.val) ∧
  (∀ r o,(r≠("OUT" : RegionName) ∨ ∀ c : Grid.Index (Checked.parameters x w),
    o≠Grid.address (Checked.parameters x w) c) →
    final.mem r o=(realEntry (Checked.parameters x w) initial).mem r o)

/-- The readable shared quotation executes for every admitted shape and both
storage branches. Its actual output Tensor coordinates equal the independent
decoded Real formula, and every other typed cell retains its initial value. -/
theorem correct (x w v r : TensorInfo) (ok : Admitted x w v r)
    (view : Output x w) (initial : Words.Memory) :
    ∃ final,RealGrid.run (Checked.parameters x w) (realEntry (Checked.parameters x w) initial)=some final ∧
      RealPost x w view initial final ∧
      LlamaFF.IndexDomain (shape x w) (LlamaFF.CheckedHost.buffers x w v r view.info.backing) := by
  obtain ⟨final,he,values,frame⟩ := VeriTile.Bench.Repairs.LlamaFFScalarCast.RawCorrect.kernel_correct (Checked.parameters x w)
    (realEntry (Checked.parameters x w) initial) (realEntry_bytes _ _)
  refine ⟨final,he,⟨?_,?_⟩,Checked.indices x w v r ok view.info.backing⟩
  · intro c
    change (final.mem "OUT" (view.physicalIndex c.1.val c.2.1.val c.2.2.val)).read .fp16=_
    rw [coordinates,values (index x w c) trivial,Cell.read_same]
  · intro region offset h
    exact frame region offset (h.imp id (fun h c _ => h c))

theorem kernel_correct (x w v r : TensorInfo) (ok : Admitted x w v r)
    (view : Output x w) (initial : Words.Memory) :
    VeriTile.Spec.Real (∃ final,
      RealGrid.run (Checked.parameters x w) (realEntry (Checked.parameters x w) initial)=some final ∧
      RealPost x w view initial final ∧
      LlamaFF.IndexDomain (shape x w) (LlamaFF.CheckedHost.buffers x w v r view.info.backing)) :=
  correct x w v r ok view initial

def NativePost (a : Words.Instructions) (x w : TensorInfo) (view : Output x w)
    (initial : Words.Memory) (final : State Words.F) : Prop :=
  (∀ c : Logical x w,(nativeResult view initial final).get c=
    ByteGrid.target a (Checked.parameters x w) initial (index x w c)) ∧
  (∀ c : Logical x w,∀ byte : Fin 2,
    (nativeResult view initial final).bytes
      (2*view.physicalIndex c.1.val c.2.1.val c.2.2.val+byte.val)=
      Words.encode16 (ByteGrid.target a (Checked.parameters x w) initial (index x w c)) byte) ∧
  (∀ r b,ByteGrid.Outside (Checked.parameters x w) r b →
    Words.commit initial final r b=initial r b)

/-- All raw half and E5M2 patterns, exact selected K16 carry order, FTZ power
lowering, butterfly reduction, FMA sigmoid gate, half narrowing, actual Tensor
coordinates and the complete relevant byte frame. Numeric parameters are only
raw scalar instructions and one fixed K16 result lane. -/
theorem fp_equiv (x w v r : TensorInfo) (ok : Admitted x w v r)
    (view : Output x w) (a : Words.Instructions) (initial : Words.Memory) :
    ∃ final,ByteGrid.run a (Checked.parameters x w) initial=some final ∧
      NativePost a x w view initial final ∧
      LlamaFF.IndexDomain (shape x w) (LlamaFF.CheckedHost.buffers x w v r view.info.backing) := by
  obtain ⟨final,he,bytes,values,frame⟩ := VeriTile.Bench.Repairs.LlamaFFScalarCast.RawNative.kernel_equiv a (Checked.parameters x w) initial
  refine ⟨final,he,⟨?_,?_,frame⟩,Checked.indices x w v r ok view.info.backing⟩
  · intro c
    change Words.read16 (Words.commit initial final) "OUT"
      (view.physicalIndex c.1.val c.2.1.val c.2.2.val)=_
    rw [coordinates]
    exact values (index x w c)
  · intro c byte
    change Words.commit initial final "OUT"
      (2*view.physicalIndex c.1.val c.2.1.val c.2.2.val+byte.val)=_
    rw [coordinates]
    exact bytes (index x w c) byte

theorem kernel_equiv (x w v r : TensorInfo) (ok : Admitted x w v r)
    (view : Output x w) (a : Words.Instructions) (initial : Words.Memory) :
    ∃ final,ByteGrid.run a (Checked.parameters x w) initial=some final ∧
      NativePost a x w view initial final ∧
      LlamaFF.IndexDomain (shape x w) (LlamaFF.CheckedHost.buffers x w v r view.info.backing) :=
  fp_equiv x w v r ok view a initial

theorem real_metadata (view : Output x w) (final : State ℝ) :
    (realResult view final).view.info.shape=[batch x,sequence x,outputs w] ∧
    (realResult view final).view.info.dtype=.other "float16" ∧
    (realResult view final).view.info.device=x.device :=
  ⟨view.shape,view.dtype,view.device⟩

theorem native_metadata (view : Output x w) (initial : Words.Memory) (final : State Words.F) :
    (nativeResult view initial final).view.info.shape=[batch x,sequence x,outputs w] ∧
    (nativeResult view initial final).view.info.dtype=.other "float16" ∧
    (nativeResult view initial final).view.info.device=x.device :=
  ⟨view.shape,view.dtype,view.device⟩

#axiomsClean correct
#axiomsClean kernel_correct
#axiomsClean fp_equiv
#axiomsClean kernel_equiv
#axiomsClean real_metadata
#axiomsClean native_metadata
#specNonCircular RealPost avoiding [RealGrid.run,Grid.grid,Grid.one,source,FP.Structural.exec]
#specNonCircular NativePost avoiding [ByteGrid.run,Grid.grid,Grid.one,nativeExecution,source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.TensorBridge
