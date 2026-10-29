import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Execute
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.FPReference
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
variable {r : Request} {caller : Set Nat} {before : Memory Word}

def storageBytes : Mode → Nat | .fp32 => 4 | .autocastFP16 => 2
def wordBytes (mode : Mode) (value : Word) : List Nat :=
  List.ofFn (fun i : Fin (storageBytes mode) => value.raw/256^i.val%256)

/-- Exact local-profile target; output uses its compute dtype, caller saved
uses FP32 even in AMP, after the explicit half store and copy conversion. -/
def FPPost {c : Fin 98} (p : LocalProfile) (h : Allocation r caller before) (result : HostResult Word)
    (currentDevice : Nat) : Prop :=
  (∀ m n,m<r.batch.rows → n<r.n →
    result.memory (h.storage 0,outputOffset r.shape m n)=
      (fpArithmetic p).encode r.mode
        (activation (fpArithmetic p) r.activation (callerPre (fpArithmetic p) r c before m n))) ∧
  (∀ m n,m<r.batch.rows → n<r.n → r.hasSaved=true →
    result.memory (r.storage 3,outputOffset r.shape m n)=
      savedValue (fpArithmetic p) r.mode (callerPre (fpArithmetic p) r c before m n)) ∧
  (∀ loc,loc.1∈caller →
    ¬(r.hasSaved=true ∧ loc.1=r.storage 3 ∧ loc.2<r.batch.rows*r.n) →
    result.memory loc=before loc) ∧ HostMetadata r h result currentDevice

specification reference_fp_post (p : LocalProfile) (h : Allocation r caller before) (c : Fin 98)
    (valid : r.Valid) (currentDevice : Nat) (trials : List (Fin 98)) :
    FPPost (c:=c) p h (execute h (fpArithmetic p) c currentDevice trials) currentDevice := by
  rw [execute_eq h (fpArithmetic p) c currentDevice trials valid]
  exact ⟨h.final_output _ c valid,h.final_saved _ c valid,h.final_frame _ c valid,
    h.host_metadata _ c currentDevice⟩

/-- Only the ten scalar/fixed-ISA numeric leaf families and one-event ordinary
machine laws occur. All casts, all K, the full grouped grid and saved copyback
are composed here, rather than assumed as completed operations. -/
theorem native_fp_host_memory (primitive : PrimitiveOps) (p : LocalProfile)
    (leaves : NumericLeaves primitive p) (h : Allocation r caller before)
    (c : Fin 98) (valid : r.Valid) (abi : LocalEventABI Word σ primitive.arithmetic)
    (state : σ) (entry : abi.observe state=h.initial) (trials : List (Fin 98)) :
    abi.observe ((h.autotunedCode trials c).native abi state).2=h.finalMemory (fpArithmetic p) c := by
  rw [h.native_autotuned primitive.arithmetic trials c valid abi state entry]
  exact congrArg (fun a => h.finalMemory a c) leaves.arithmetic_eq

def nativeResult (h : Allocation r caller before) (c : Fin 98)
    (abi : LocalEventABI Word σ a) (state : σ) (currentDevice : Nat) (trials : List (Fin 98)) : HostResult Word :=
  ⟨abi.observe ((h.autotunedCode trials c).native abi state).2,h.returned,h.context,
    if r.hasSaved then some (r.argument 3) else none,currentDevice⟩

specification fp_equiv (primitive : PrimitiveOps) (p : LocalProfile) (leaves : NumericLeaves primitive p)
    (h : Allocation r caller before) (c : Fin 98) (valid : r.Valid)
    (abi : LocalEventABI Word σ primitive.arithmetic) (state : σ)
    (entry : abi.observe state=h.initial) (currentDevice : Nat) (trials : List (Fin 98)) :
    FPPost (c:=c) p h (nativeResult h c abi state currentDevice trials) currentDevice := by
  have equal : nativeResult h c abi state currentDevice trials=execute h (fpArithmetic p) c currentDevice trials := by
    rw [execute_eq h (fpArithmetic p) c currentDevice trials valid]
    unfold nativeResult Allocation.hostResult
    rw [native_fp_host_memory primitive p leaves h c valid abi state entry trials]
  rw [equal]
  exact reference_fp_post p h c valid currentDevice trials

theorem fp_output_bytes (primitive : PrimitiveOps) (p : LocalProfile) (leaves : NumericLeaves primitive p)
    (h : Allocation r caller before) (c : Fin 98) (valid : r.Valid)
    (abi : LocalEventABI Word σ primitive.arithmetic) (state : σ)
    (entry : abi.observe state=h.initial) (m n : Nat) (hm : m<r.batch.rows) (hn : n<r.n)
    (trials : List (Fin 98)) :
    wordBytes r.mode (abi.observe ((h.autotunedCode trials c).native abi state).2 (h.storage 0,outputOffset r.shape m n))=
      wordBytes r.mode ((fpArithmetic p).encode r.mode
        (activation (fpArithmetic p) r.activation (callerPre (fpArithmetic p) r c before m n))) := by
  rw [native_fp_host_memory primitive p leaves h c valid abi state entry trials,h.final_output _ c valid m n hm hn]

theorem fp_saved_bytes (primitive : PrimitiveOps) (p : LocalProfile) (leaves : NumericLeaves primitive p)
    (h : Allocation r caller before) (c : Fin 98) (valid : r.Valid)
    (abi : LocalEventABI Word σ primitive.arithmetic) (state : σ)
    (entry : abi.observe state=h.initial) (m n : Nat) (hm : m<r.batch.rows) (hn : n<r.n)
    (saved : r.hasSaved=true) (trials : List (Fin 98)) :
    wordBytes .fp32 (abi.observe ((h.autotunedCode trials c).native abi state).2 (r.storage 3,outputOffset r.shape m n))=
      wordBytes .fp32 (savedValue (fpArithmetic p) r.mode (callerPre (fpArithmetic p) r c before m n)) := by
  rw [native_fp_host_memory primitive p leaves h c valid abi state entry trials,h.final_saved _ c valid m n hm hn saved]

#axiomsClean reference_fp_post
#axiomsClean native_fp_host_memory
#axiomsClean fp_equiv
#axiomsClean fp_output_bytes
#axiomsClean fp_saved_bytes
end
end VeriTile.Bench.Repairs.TritonLinearActivation
