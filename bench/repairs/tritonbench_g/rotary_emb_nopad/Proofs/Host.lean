import bench.repairs.tritonbench_g.rotary_emb_nopad.Spec
import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.ABI

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Host
open Checked

/-- Python returns None and retains all supplied Tensor objects. -/
structure Returned (α : Type) where
  value : Option Unit
  arguments : Fin 7 → Tensor
  memory : Memory α

noncomputable def mathematical (c : Call) (index : Indices) (m : Memory ℝ) : Returned ℝ :=
  ⟨none,c.tensor,source c index realOps m⟩

noncomputable def realInvoke (c : Call) (metadata : Physical.Bytes) (m : Memory ℝ) : Option (Returned ℝ) :=
  if check c (Physical.indices c.request metadata) then
    some (mathematical c (Physical.indices c.request metadata) m) else none

def selected (p : Native.Instructions) (c : Call) (m : Physical.Bytes) : Returned Physical.Byte :=
  ⟨none,c.tensor,Physical.native p c m⟩

def invoke (p : Native.Instructions) (c : Call) (m : Physical.Bytes) : Option (Returned Physical.Byte) :=
  if check c (Physical.indices c.request m) then some (selected p c m) else none

structure RealClaim (c : Call) (index : Indices) (m : Memory ℝ) (result : Returned ℝ) : Prop where
  none : result.value=Option.none
  objects : result.arguments=c.tensor
  source : result.memory=RotaryEmbNopad.source c index realOps m
  values : ∀ key token head column,token<c.request.tokens → head<c.request.heads key → column<c.request.width →
    result.memory (.other c.request.device) (c.request.destination index key token head column)=
      realColumn c.request m key token head column
  frame : ∀ region address,(∀ w,c.request.Active w → ¬Writes c.request index w region address) →
    result.memory region address=m region address

theorem complete_real (c : Call) (metadata : Physical.Bytes) (m : Memory ℝ)
    (accepted : check c (Physical.indices c.request metadata)=true) :
    ∃ result,realInvoke c metadata m=some result ∧ RealClaim c (Physical.indices c.request metadata) m result := by
  refine ⟨mathematical c (Physical.indices c.request metadata) m,by simp [realInvoke,accepted],?_,?_,?_,?_,?_⟩
  · rfl
  · rfl
  · rfl
  · intro key token head column ht hh hd
    obtain ⟨active,columnEq⟩ := column_owner c.request key token head column ht hh hd
    have read := execute_readback realOps c _ accepted (schedule c.request) m
      (schedule_active c.request) (schedule_nodup c.request) _ (schedule_covers c _ accepted _ active) (second c.request column)
    rw [real_pair] at read
    change (source c _ realOps m) _ _=_
    rw [source_eq_execute c _ accepted]
    rw [destination,columnEq] at read
    exact read
  · intro region address outside
    change source c _ realOps m region address=m region address
    rw [source_eq_execute c _ accepted]
    exact execute_frame realOps c.request _ _ m region address (fun w hw => outside w (schedule_active c.request w hw))

structure NativeClaim (actual target : Native.Instructions) (c : Call) (m : Physical.Bytes)
    (result : Returned Physical.Byte) : Prop where
  none : result.value=Option.none
  objects : result.arguments=c.tensor
  source : Physical.load result.memory=Native.project (RotaryEmbNopad.source c (Physical.indices c.request m)
    (Native.ops actual c.request.cached) (Native.liftMemory (Physical.load m)))
  words : ∀ key token head column,token<c.request.tokens → head<c.request.heads key → column<c.request.width →
    Physical.load result.memory (.other c.request.device)
      (c.request.destination (Physical.indices c.request m) key token head column)=
        Native.expected target c.request (columnWork c.request key token head column) (Physical.load m) (second c.request column)
  bytes : ∀ key token head column,token<c.request.tokens → head<c.request.heads key → column<c.request.width → ∀ byte : Fin 4,
    result.memory (.other c.request.device)
      (4*c.request.destination (Physical.indices c.request m) key token head column+byte.val)=
        Physical.encode (Native.expected target c.request (columnWork c.request key token head column) (Physical.load m) (second c.request column)) byte
  frame : ∀ region address,(∀ w,c.request.Active w → ¬Writes c.request (Physical.indices c.request m) w region (address/4)) →
    result.memory region address=m region address
  metadata : Physical.MetadataAgreement c.request (Physical.indices c.request result.memory) (Physical.indices c.request m)
  cachedK : c.request.cached=true → ∀ offset,offset<wordExtent c.request 1 → ∀ byte : Fin 4,
    result.memory (.other c.request.device) (4*(c.request.kBase+offset)+byte.val)=
      m (.other c.request.device) (4*(c.request.kBase+offset)+byte.val)

theorem complete_native (p : Native.Instructions) (c : Call) (m : Physical.Bytes)
    (accepted : check c (Physical.indices c.request m)=true) :
    ∃ result,invoke p c m=some result ∧ NativeClaim p p c m result := by
  refine ⟨selected p c m,by simp [invoke,accepted],rfl,rfl,Physical.native_source p c m accepted,?_,?_,
    Physical.native_frame p c m accepted,Physical.native_metadata p c m accepted,
    Physical.cached_k_preserved p c m accepted⟩
  · intro key token head column ht hh hd
    obtain ⟨active,columnEq⟩ := column_owner c.request key token head column ht hh hd
    have read := Physical.native_result p c m accepted _ active (second c.request column)
    rw [destination,columnEq] at read
    exact read
  · intro key token head column ht hh hd byte
    obtain ⟨active,columnEq⟩ := column_owner c.request key token head column ht hh hd
    have read := Physical.native_bytes p c m accepted _ active (second c.request column) byte
    rw [destination,columnEq] at read
    exact read

#axiomsClean complete_real
#axiomsClean complete_native
end VeriTile.Bench.Repairs.RotaryEmbNopad.Host
