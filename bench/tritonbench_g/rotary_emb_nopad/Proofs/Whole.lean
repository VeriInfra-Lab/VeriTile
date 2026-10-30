import bench.tritonbench_g.rotary_emb_nopad.Proofs.Vector
import bench.tritonbench_g.rotary_emb_nopad.Proofs.Metadata

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Physical
open Checked Native

/-- Direct source-addressed execution: metadata is decoded from the current
bytes at every pair. Its invariance and the complete vector/grid ordering are
proved below; there is no copied input or staging allocation. -/
def native (p : Instructions) (c : Call) (m : Bytes) : Bytes :=
  dynamicExecute p c.request (schedule c.request) m

theorem native_fixed (p : Instructions) (c : Call) (m : Bytes)
    (accepted : check c (indices c.request m)=true) :
    native p c m=byteExecute p c.request (indices c.request m) (schedule c.request) m :=
  dynamic_eq p c _ accepted _ (schedule_active c.request) m (agreement_refl c.request _)

/-- Constructive complete source correspondence, including both original
statement orders and fresh physical int64 metadata loads. -/
theorem native_source (p : Instructions) (c : Call) (m : Bytes)
    (accepted : check c (indices c.request m)=true) :
    load (native p c m)=project (source c (indices c.request m)
      (ops p c.request.cached) (liftMemory (load m))) := by
  rw [native_fixed p c m accepted,source_eq_execute c _ accepted,execute_project,execute_load]
  rfl

theorem native_result (p : Instructions) (c : Call) (m : Bytes)
    (accepted : check c (indices c.request m)=true) (w : Work) (active : c.request.Active w) (odd : Bool) :
    load (native p c m) (.other c.request.device) (destination c.request (indices c.request m) w odd)=
      expected p c.request w (load m) odd := by
  rw [native_fixed p c m accepted]
  exact readback p c _ accepted _ (schedule_active c.request) (schedule_nodup c.request) m w
    (schedule_covers c _ accepted w active) odd

theorem native_bytes (p : Instructions) (c : Call) (m : Bytes)
    (accepted : check c (indices c.request m)=true) (w : Work) (active : c.request.Active w)
    (odd : Bool) (i : Fin 4) :
    native p c m (.other c.request.device) (4*destination c.request (indices c.request m) w odd+i.val)=
      encode (expected p c.request w (load m) odd) i := by
  rw [native_fixed p c m accepted]
  exact output_bytes p c _ accepted _ (schedule_active c.request) (schedule_nodup c.request) m w
    (schedule_covers c _ accepted w active) odd i

theorem native_frame (p : Instructions) (c : Call) (m : Bytes)
    (accepted : check c (indices c.request m)=true) (region : Region) (address : Nat)
    (outside : ∀ w,c.request.Active w → ¬Writes c.request (indices c.request m) w region (address/4)) :
    native p c m region address=m region address := by
  rw [native_fixed p c m accepted]
  exact frame p c.request _ _ m region address (fun w hw => outside w (schedule_active c.request w hw))

theorem native_metadata (p : Instructions) (c : Call) (m : Bytes)
    (accepted : check c (indices c.request m)=true) :
    MetadataAgreement c.request (indices c.request (native p c m)) (indices c.request m) := by
  rw [native_fixed p c m accepted]
  exact metadata_frame p c _ accepted _ (schedule_active c.request) m

theorem schedule_empty (r : Request) (empty : Checked.empty r=true) : schedule r=[] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro w member
  have active := schedule_active r w member
  simp only [Checked.empty,decide_eq_true_eq] at empty
  rcases empty with zero|zero <;> simp_all [Request.Active]

theorem native_empty (p : Instructions) (c : Call) (m : Bytes)
    (empty : Checked.empty c.request=true) : native p c m=m := by
  rw [native,schedule_empty c.request empty]
  rfl

/-- Every feature column has exactly one active split-half owner. -/
theorem every_column (r : Request) (key : Bool) (token head column : Nat)
    (ht : token<r.tokens) (hh : head<r.heads key) (hd : column<r.width) :
    ∃ w : Work,∃ odd : Bool,r.Active w ∧ w.key=key ∧ w.token=token ∧ w.head=head ∧ r.dim w.pair odd=column := by
  by_cases lower : column<r.half
  · exact ⟨⟨key,token,head,column⟩,false,⟨ht,hh,lower⟩,rfl,rfl,rfl,by simp [Request.dim]⟩
  · refine ⟨⟨key,token,head,column-r.half⟩,true,⟨ht,hh,?_⟩,rfl,rfl,rfl,?_⟩
    · change column-r.half<r.half
      simp only [Request.width] at hd
      omega
    · simp [Request.dim]
      omega

theorem native_readonly (p : Instructions) (c : Call) (m : Bytes)
    (accepted : check c (indices c.request m)=true) (i : Fin 7) (pi : present c.request i=true)
    (readonly : ∀ key : Bool,destinationRole c.request key≠i)
    (offset : Nat) (inside : offset<wordExtent c.request i) (byte : Fin 4) :
    native p c m (.other c.request.device) (4*(base c.request i+offset)+byte.val)=
      m (.other c.request.device) (4*(base c.request i+offset)+byte.val) := by
  apply native_frame p c m accepted
  intro w hw
  have div : (4*(base c.request i+offset)+byte.val)/4=base c.request i+offset := by omega
  rw [div]
  rintro ⟨_,same|same⟩
  · exact readonly_role c _ accepted i pi readonly w hw false offset inside same.symm
  · exact readonly_role c _ accepted i pi readonly w hw true offset inside same.symm

theorem cached_k_preserved (p : Instructions) (c : Call) (m : Bytes)
    (accepted : check c (indices c.request m)=true) (cached : c.request.cached=true)
    (offset : Nat) (inside : offset<wordExtent c.request 1) (byte : Fin 4) :
    native p c m (.other c.request.device) (4*(c.request.kBase+offset)+byte.val)=
      m (.other c.request.device) (4*(c.request.kBase+offset)+byte.val) := by
  exact native_readonly p c m accepted 1 (by simp [present])
    (by intro key; cases key <;> simp [destinationRole,role,cached]) offset inside byte

#axiomsClean native_source
#axiomsClean native_result
#axiomsClean native_bytes
#axiomsClean native_frame
#axiomsClean native_metadata
#axiomsClean native_empty
#axiomsClean every_column
#axiomsClean cached_k_preserved
end VeriTile.Bench.Repairs.RotaryEmbNopad.Physical
