import bench.repairs.tritonbench_g.rotary_transform.Proofs.HistoricalCorrect
namespace VeriTile.Bench.Repairs.RotaryTransform
open Native Physical Host

theorem selected_pair (call : Call) (callerDevice : Nat) (p : Instructions) (initial : Bytes)
    (hc : (originalConfiguration call.shape call.abi initial).Valid)
    (w : Work) (hw : (originalConfiguration call.shape call.abi initial).Active w) (o : Bool) :
    load (selected call callerDevice p initial).memory call.shape.output
      ((originalConfiguration call.shape call.abi initial).cell w o)=
    pairWord (originalConfiguration call.shape call.abi initial) p w (load initial) o := by
  rw [selected_memory call callerDevice p initial hc]
  exact (native_result _ hc p _ w hw o).trans (prepared_pair _ p call.abi initial w o)

theorem selected_tail (call : Call) (callerDevice : Nat) (p : Instructions) (initial : Bytes)
    (hc : (originalConfiguration call.shape call.abi initial).Valid)
    (n h d : Nat) (hn : n<call.shape.total) (hh : h<call.shape.heads) (hd : d<call.shape.width)
    (tail : 2*call.shape.half≤d) (i : Fin 4) :
    (selected call callerDevice p initial).memory call.shape.output (4*call.shape.address n h d+i.val)=
      initial .input (4*call.shape.address n h d+i.val) := by
  rw [selected_memory call callerDevice p initial hc]
  apply Eq.trans (native_tail _ hc p _ n h d hh hd tail i)
  change prepareBytes call.shape (normalizeFor call.shape call.abi initial) call.shape.output
    (4*call.shape.address n h d+i.val)=initial .input (4*call.shape.address n h d+i.val)
  have bound := address_bound call.shape n h d hn hh hd
  have rem := address_mod call.shape n h d hd
  have div : (4*call.shape.address n h d+i.val)/4=call.shape.address n h d := by omega
  cases hi : call.shape.inplace <;> cases hv : call.shape.varlen <;>
    simp [hv,prepareBytes,copied,Shape.output,hi,div,rem,bound,tail,normalizeFor,Physical.normalize]

/-- Whole returned FP32 payload, including copied suffix and all layout,
conjugation, packed/fixed, offset, and inplace flags. -/
theorem correct_native (call : Call) (callerDevice : Nat) (p : Instructions) (initial : Bytes)
    (hc : (originalConfiguration call.shape call.abi initial).Valid)
    (batch row head col : Nat) (hb : batch<call.shape.batches)
    (hr : row<(originalConfiguration call.shape call.abi initial).length batch)
    (hh : head<call.shape.heads) (hd : col<call.shape.width) :
    load (selected call callerDevice p initial).memory call.shape.output
      (call.shape.address ((originalConfiguration call.shape call.abi initial).start batch+row) head col)=
    nativeFormula (originalConfiguration call.shape call.abi initial) p (load initial) batch row head col := by
  let c := originalConfiguration call.shape call.abi initial
  change load (Host.selected call callerDevice p initial).memory c.shape.output
    (c.shape.address (c.start batch+row) head col)=nativeFormula c p (load initial) batch row head col
  by_cases rot : col<2*c.shape.half
  · obtain ⟨hp,back⟩ := column_inverse c.shape col rot
    let w : Work := ⟨batch,row,head,c.shape.pairOf col⟩
    have active : c.Active w := ⟨hb,hr,hh,hp⟩
    have result := selected_pair call callerDevice p initial hc w active (c.shape.second col)
    change load (Host.selected call callerDevice p initial).memory c.shape.output (c.cell w (c.shape.second col))=
      pairWord c p w (load initial) (c.shape.second col) at result
    rw [native_pair_formula c p (load initial) w active] at result
    simpa only [w,Config.cell,back] using result
  · have tail : 2*call.shape.half≤col := by change 2*c.shape.half≤col;omega
    simp only [nativeFormula,rot,↓reduceIte]
    unfold load
    congr 1
    funext i
    exact selected_tail call callerDevice p initial hc _ _ _ (row_bound c hc batch row hb hr) hh hd tail i

theorem correct_native_bytes (call : Call) (callerDevice : Nat) (p : Instructions) (initial : Bytes)
    (hc : (originalConfiguration call.shape call.abi initial).Valid)
    (batch row head col : Nat) (hb : batch<call.shape.batches)
    (hr : row<(originalConfiguration call.shape call.abi initial).length batch)
    (hh : head<call.shape.heads) (hd : col<call.shape.width) (i : Fin 4) :
    (selected call callerDevice p initial).memory call.shape.output
      (4*call.shape.address ((originalConfiguration call.shape call.abi initial).start batch+row) head col+i.val)=
    encode (nativeFormula (originalConfiguration call.shape call.abi initial) p (load initial) batch row head col) i := by
  rw [←original_bytes (Host.selected call callerDevice p initial).memory call.shape.output
    (call.shape.address ((originalConfiguration call.shape call.abi initial).start batch+row) head col) i,
    correct_native call callerDevice p initial hc batch row head col hb hr hh hd]

/-- Caller frame. The private normalized-CU allocation and the explicit host
suffix copy are accounted for separately from the kernel writes. -/
theorem caller_frame (call : Call) (callerDevice : Nat) (p : Instructions) (initial : Bytes)
    (hc : (originalConfiguration call.shape call.abi initial).Valid) (r : Region) (b : Nat)
    (notPrivate : r≠.cumulative) (copy : ¬copied call.shape r (b/4))
    (outside : ∀ w,(originalConfiguration call.shape call.abi initial).Active w →
      ¬Writes (originalConfiguration call.shape call.abi initial) w r (b/4)) :
    (selected call callerDevice p initial).memory r b=initial r b := by
  rw [selected_memory call callerDevice p initial hc,native_frame _ p _ r b outside]
  cases hv : call.shape.varlen <;> simp [prepareBytes,copy,normalizeFor,hv,Physical.normalize,notPrivate]

theorem readonly_inputs (call : Call) (callerDevice : Nat) (p : Instructions) (initial : Bytes)
    (hc : (originalConfiguration call.shape call.abi initial).Valid) (r : Region) (b : Nat)
    (h : r≠.input ∧ r≠.output ∧ r≠.cumulative) :
    (selected call callerDevice p initial).memory r b=initial r b := by
  apply caller_frame call callerDevice p initial hc r b h.2.2
  · simp [copied,h.2.1]
  · intro w _
    cases hi : call.shape.inplace <;> simp_all [Writes,originalConfiguration,Shape.output]

/-- Only four scalar numerical correspondences remain conditional. -/
theorem fp_equivalent (call : Call) (callerDevice : Nat) (actual target : Instructions)
    (ports : Obligations actual target) (initial : Bytes) :
    selected call callerDevice actual initial=selected call callerDevice target initial := by
  rw [instructions_eq actual target ports]
#axiomsClean correct_native
#axiomsClean correct_native_bytes
#axiomsClean caller_frame
#axiomsClean readonly_inputs
#axiomsClean fp_equivalent
end VeriTile.Bench.Repairs.RotaryTransform
