import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Contract
import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Proofs.Kernel
import bench.repairs.tritonbench_g.chunk_delta_fwd.Proofs.RealPhysical
import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Proofs.NativeBinding
import bench.repairs.tritonbench_g.chunk_delta_fwd.Proofs.CheckedHost
import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Correct
import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Misc.NativeCorrect

/-! Optional Python-check correspondence; primary kernel results remain in Correct and FPEquiv. -/
namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.TensorBridge
open VeriTile Triton FP.Structural ChunkDeltaFwd Checked
noncomputable section

theorem python_checks_iff_preconditions (r : Request) : check r=Bool.true ↔ InputContract r := check_iff r

#axiomsClean python_checks_iff_preconditions
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.TensorBridge


namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.TensorBridge.RealApplication
open VeriTile Triton FP.Structural ChunkDeltaFwd Checked
noncomputable section
set_option maxHeartbeats 8000000

/-- Pre-kernel output descriptors supply only shape, dtype, pointer metadata
and separation. No wrapper execution, allocation result or output value is
assumed. The mathematical recurrence is independent of source execution. -/
structure Result (r : Request) (a : Allocations r) (before : RealPhysical.Physical) : Prop where
  source : runGrid Real.algebra Real.dot r.params (grid r.params) (RealPhysical.input r a before)=
    some (RealPhysical.source r a before)
  history : ∀ head,head<r.B*r.H → ∀ chunk,chunk<r.dims.NT → ∀ key,key<r.K → ∀ value,value<r.V →
    RealPhysical.returned r a before (a.history.byteBase+4*(((head*r.dims.NT+chunk)*r.K+key)*r.V+value))=
      RealPhysical.history r a before head chunk key value
  corrected : ∀ head,head<r.B*r.H → ∀ time,time<r.T → ∀ value,value<r.V →
    RealPhysical.returned r a before (a.corrected.byteBase+4*((head*r.T+time)*r.V+value))=
      RealPhysical.corrected r a before head time value
  final : ∀ f,r.final=some f → ∀ head,head<r.B*r.H → ∀ key,key<r.K → ∀ value,value<r.V →
    RealPhysical.returned r a before (f.byteBase+4*((head*r.K+key)*r.V+value))=
      RealPhysical.history r a before head r.dims.NT key value
  exterior : ∀ address,Bytes.Outside a.history address → Bytes.Outside a.corrected address →
    (∀ f∈r.final.toList,Bytes.Outside f address) → RealPhysical.returned r a before address=before address

/-- Complete shared-kernel execution, independent Real tensor results and
exterior frame under the explicit static-dot candidate contract. -/
theorem chunk_delta_kernel_correct (r : Request) (h : InputContract r)
    (a : Allocations r) (before : RealPhysical.Physical) : Spec.Real (Result r a before) := by
  have checked := base_checked r h
  exact ⟨(RealPhysical.source_complete r checked a before).1,
    RealPhysical.history_read r checked a before,RealPhysical.corrected_read r checked a before,
    RealPhysical.final_read r checked a before,RealPhysical.exterior r a before⟩

abbrev arbitrary_grid_order := @Kernel.real_grid

#axiomsClean chunk_delta_kernel_correct
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.TensorBridge.RealApplication


namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.TensorBridge.NativeApplication
open VeriTile Triton FP.Structural ChunkDeltaFwd Checked Checked.NativePointers
noncomputable section
set_option maxHeartbeats 8000000

/-- Current physical bytes are the actual evaluated FP32 source operands.
Input words are identity leaves and include all NaN/Inf/signed-zero payloads. -/
theorem source_load_word (I : Primitives) (r : Request) (a : Allocations r)
    (before : Bytes.Memory) (region : RegionName) (t : Checked.Tensor)
    (bound : descriptor r a region=some t) (i : Nat) :
    (Native.lower (Native.algebra.fp32Load ((input r a before region i).read .real))).eval I.isa=
      Bytes.read before (t.byteBase+4*i) := by
  rw [input_word r a before region t bound i]
  rfl

structure Result (I : Primitives) (r : Request) (h : InputContract r)
    (a : Allocations r) (before : Bytes.Memory) : Prop where
  source : runGrid Native.algebra Native.dot r.params (grid r.params) (input r a before)=
    some (Checked.source r a before)
  inputs : ∀ region t,descriptor r a region=some t → ∀ i,
    (Native.lower (Native.algebra.fp32Load ((input r a before region i).read .real))).eval I.isa=
      Bytes.read before (t.byteBase+4*i)
  pointers : ∀ (x : Coordinates r.dims) site t,descriptor r a (region site)=some t →
    (baseWord r.dims x site).toInt=(base r.dims x site : Int) ∧
    (blockWord r.dims x site).toInt=(blockOffset r.dims x site : Int) ∧
    (word64 t.byteBase+((baseWord r.dims x site).signExtend 64+blockWord r.dims x site)*word64 4).toNat=
      t.byteBase+4*offset r.dims x site ∧
    (t.byteBase+4*offset r.dims x site)%4=0 ∧ t.byteBase+4*offset r.dims x site+4<2^63
  history : ∀ head,head<r.B*r.H → ∀ chunk,chunk<r.dims.NT → ∀ key (hk : key<r.K),∀ value,value<r.V →
    Bytes.read (returnedBytes I r a before) (a.history.byteBase+4*(((head*r.dims.NT+chunk)*r.K+key)*r.V+value))=
      (NativeReference.history r.params (r.shape (base_checked r h))
        (NativeReference.inputs r.params (bits r a before)) head chunk key value hk).eval I.isa
  corrected : ∀ head,head<r.B*r.H → ∀ time,time<r.T → ∀ value,value<r.V →
    Bytes.read (returnedBytes I r a before) (a.corrected.byteBase+4*((head*r.T+time)*r.V+value))=
      (NativeReference.corrected r.params (r.shape (base_checked r h))
        (NativeReference.inputs r.params (bits r a before)) head time value).eval I.isa
  final : ∀ f,r.final=some f → ∀ head,head<r.B*r.H → ∀ key (hk : key<r.K),∀ value,value<r.V →
    Bytes.read (returnedBytes I r a before) (f.byteBase+4*((head*r.K+key)*r.V+value))=
      (NativeReference.history r.params (r.shape (base_checked r h))
        (NativeReference.inputs r.params (bits r a before)) head r.dims.NT key value hk).eval I.isa
  inputFrame : ∀ t∈r.inputs,∀ byte,byte<4*t.count →
    returnedBytes I r a before (t.byteBase+byte)=before (t.byteBase+byte)
  exterior : ∀ address,Bytes.Outside a.history address → Bytes.Outside a.corrected address →
    (∀ f∈r.final.toList,Bytes.Outside f address) → returnedBytes I r a before address=before address

/-- Full shared kernel to independent input-reader native recurrence and
physical FP32 tensor results. Scalar ISA/compiler realization remains an
explicit model boundary; no numerical experiment or finite-input premise. -/
theorem chunk_delta_kernel_native (I : Primitives) (r : Request) (h : InputContract r)
    (a : Allocations r) (pointers : a.check=Bool.true) (before : Bytes.Memory) : Result I r h a before := by
  have checked := base_checked r h
  refine ⟨(source_complete r checked a before).1,source_load_word I r a before,
    HostRun.native_addresses r checked a pointers,?_,?_,?_,?_,?_⟩
  · intro head hh chunk hc key hk value hv
    rw [history_word I r checked a before head hh chunk hc key hk value hv,
      NativeReference.history_binding]
  · intro head hh time ht value hv
    rw [corrected_word I r checked a before head hh time ht value hv,
      NativeReference.corrected_binding]
  · intro f hf head hh key hk value hv
    rw [final_word I r checked a before f hf head hh key hk value hv,
      NativeReference.history_binding]
  · exact caller_frame I r checked a before (Checked.source r a before)
  · exact Checked.exterior I r a before (Checked.source r a before)

#axiomsClean source_load_word
#axiomsClean chunk_delta_kernel_native
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.TensorBridge.NativeApplication
