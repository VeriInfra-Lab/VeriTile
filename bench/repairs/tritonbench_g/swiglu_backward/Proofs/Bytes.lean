import bench.repairs.tritonbench_g.swiglu_backward.Proofs.Native

namespace VeriTile.Bench.Repairs.SwigluBackward.KernelProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward
open Whole
noncomputable section
open Classical

theorem output_aligned (p : Prepared α call entry preexisting) (s : Storage p)
    (h : LaunchContract p s.byteBase) (ch : Channel) (on : enabled call.recompute ch)
    (r q : Nat) (hr : r < p.batchShape.prod) : outputAddress p s ch r q % 4 = 0 := by
  have aligned := h.alignment (Nat.zero_lt_of_lt hr)
  cases ch with
  | dx =>
    have a := aligned .dx (Or.inl (by decide))
    dsimp [Access.view,Prepared.view] at a
    dsimp [outputAddress,Prepared.location,Prepared.view]
    omega
  | dy =>
    have a := aligned .dy (Or.inl (by decide))
    dsimp [Access.view,Prepared.view] at a
    dsimp [outputAddress,Prepared.location,Prepared.view]
    omega
  | out =>
    have a := aligned .out (by simpa [Access.active,enabled] using on)
    dsimp [Access.view,Prepared.view] at a
    dsimp [outputAddress,Prepared.location,Prepared.view]
    omega

def ByteWritten (p : Prepared α call entry preexisting) (s : Storage p) (device address : Nat) : Prop :=
  ∃ start, PhysicalWritten p s device start ∧ start ≤ address ∧ address < start+4

/-- Concrete four-byte writeback. Every byte is read from the actual completed
source state; every byte outside the effective stores uses the entry byte state. -/
def byteCommit [Inhabited α] (encode : α → Nat → β)
    (p : Prepared α call entry preexisting) (s : Storage p) (final : State α)
    (before : Nat → Nat → β) (device address : Nat) : β :=
  if h : ByteWritten p s device address then
    encode (physicalCommit p s final device h.choose) (address-h.choose)
  else before device address

theorem byte_value [Inhabited α] (encode : α → Nat → β)
    (p : Prepared α call entry preexisting) (s : Storage p)
    (h : LaunchContract p s.byteBase) (final : State α) (before : Nat → Nat → β)
    (ch : Channel) (on : enabled call.recompute ch) (r q : Nat)
    (hr : r < p.batchShape.prod) (hq : q < p.ncols) (j : Fin 4) :
    byteCommit encode p s final before call.input.cudaDevice (outputAddress p s ch r q+j.val) =
      encode (physicalCommit p s final call.input.cudaDevice (outputAddress p s ch r q)) j.val := by
  have written : ByteWritten p s call.input.cudaDevice (outputAddress p s ch r q+j.val) :=
    ⟨outputAddress p s ch r q,⟨ch,r,q,on,hr,hq,rfl,rfl⟩,by omega,by omega⟩
  unfold byteCommit
  rw [dif_pos written]
  have witness := written.choose_spec
  obtain ⟨ch',r',q',on',hr',hq',_,address⟩ := witness.1
  have first := output_aligned p s h ch on r q hr
  have second := output_aligned p s h ch' on' r' q' hr'
  have same : written.choose = outputAddress p s ch r q := by
    have bounds := witness.2
    omega
  simp only [same,Nat.add_sub_cancel_left]

theorem byte_frame [Inhabited α] (encode : α → Nat → β)
    (p : Prepared α call entry preexisting) (s : Storage p) (final : State α)
    (before : Nat → Nat → β) (device address : Nat) (outside : ¬ByteWritten p s device address) :
    byteCommit encode p s final before device address = before device address := dif_neg outside

/-- Little-endian byte projection preserves every raw FP32 bit, including
NaN payload bits, infinities, subnormal numbers, and signed zero. -/
def wordByte (word : BitVec 32) (byte : Nat) : BitVec 8 :=
  BitVec.ofNat 8 (word.toNat / 2^(8*byte))

/-- Ordinary little-endian FP32 load semantics at the actual input pointers.
Only entry memory appears; this contains no final-state or arithmetic premise. -/
def ByteInputBinding (M : Observed.NativeModel (BitVec 32))
    (p : Prepared (Observed.Term (BitVec 32)) call (Observed.entryGraph entry) preexisting)
    (s : Storage p) (before : Nat → Nat → BitVec 8) : Prop :=
  ∀ a : Access, (a = .x ∨ a = .y ∨ a = .dout) → ∀ r q,
    r < p.batchShape.prod → q < p.ncols → ∀ j : Fin 4,
    before (s.device (a.view p).allocation) (accessAddress p s a r q+j.val) =
      wordByte (Observed.interpret M (s.preparedRead (s.device (a.view p).allocation)
        (accessAddress p s a r q))) j.val

def LogicalInputBytes
    (p : Prepared (Observed.Term (BitVec 32)) call (Observed.entryGraph entry) preexisting)
    (s : Storage p) (before : Nat → Nat → BitVec 8) : Prop :=
  ∀ r q, r < p.batchShape.prod → q < p.ncols → ∀ j : Fin 4,
    (before (s.device p.xy.matrix.allocation) (accessAddress p s .x r q+j.val) =
      wordByte (call.input.read entry (r*(2*p.ncols)+q)) j.val) ∧
    (before (s.device p.xy.matrix.allocation) (accessAddress p s .y r q+j.val) =
      wordByte (call.input.read entry (r*(2*p.ncols)+(p.ncols+q))) j.val) ∧
    (before (s.device p.dout.matrix.allocation) (accessAddress p s .dout r q+j.val) =
      wordByte (call.upstream.read entry (r*p.ncols+q)) j.val)

theorem logical_input_bytes (M : Observed.NativeModel (BitVec 32))
    (p : Prepared (Observed.Term (BitVec 32)) call (Observed.entryGraph entry) preexisting)
    (s : Storage p) (hc : checks p = Bool.true) (h : LaunchContract p s.byteBase)
    (before : Nat → Nat → BitVec 8) (binding : ByteInputBinding M p s before) :
    LogicalInputBytes p s before := by
  intro r q hr hq j
  have logical := prepared_reads p (checkedDomain p s hc h) r q hr hq
  have x := (initial_read p s hc h .x (Or.inl (by decide)) r q hr hq).symm.trans logical.1
  have y := (initial_read p s hc h .y (Or.inl (by decide)) r q hr hq).symm.trans logical.2.1
  have d := (initial_read p s hc h .dout (Or.inl (by decide)) r q hr hq).symm.trans logical.2.2
  refine ⟨?_,?_,?_⟩
  · apply (binding .x (Or.inl rfl) r q hr hq j).trans
    simpa only [VeriTile.Bench.TritonBenchG.SwigluFwd.Whole.Tensor.read,
      Observed.entryGraph,Observed.interpret] using congrArg (fun t => wordByte (Observed.interpret M t) j.val) x
  · apply (binding .y (Or.inr (Or.inl rfl)) r q hr hq j).trans
    simpa only [VeriTile.Bench.TritonBenchG.SwigluFwd.Whole.Tensor.read,
      Observed.entryGraph,Observed.interpret] using congrArg (fun t => wordByte (Observed.interpret M t) j.val) y
  · apply (binding .dout (Or.inr (Or.inr rfl)) r q hr hq j).trans
    simpa only [VeriTile.Bench.TritonBenchG.SwigluFwd.Whole.Tensor.read,
      Observed.entryGraph,Observed.interpret] using congrArg (fun t => wordByte (Observed.interpret M t) j.val) d

def NativeBytes (M : Observed.NativeModel (BitVec 32))
    (p : Prepared (Observed.Term (BitVec 32)) call (Observed.entryGraph entry) preexisting)
    (s : Storage p) (final : State (Observed.Term (BitVec 32)))
    (before : Nat → Nat → BitVec 8) : Prop :=
  (∀ ch, enabled call.recompute ch → ∀ r q, r < p.batchShape.prod → q < p.ncols → ∀ j : Fin 4,
    byteCommit (fun t j => wordByte (Observed.interpret M t) j) p s final before
      call.input.cudaDevice (outputAddress p s ch r q+j.val) =
      wordByte (Observed.reference M ch
        (call.input.read entry (r*(2*p.ncols)+q))
        (call.input.read entry (r*(2*p.ncols)+(p.ncols+q)))
        (call.upstream.read entry (r*p.ncols+q))) j.val) ∧
  (∀ device address, ¬ByteWritten p s device address →
    byteCommit (fun t j => wordByte (Observed.interpret M t) j) p s final before device address =
      before device address)

theorem native_bytes (M : Observed.NativeModel (BitVec 32))
    (p : Prepared (Observed.Term (BitVec 32)) call (Observed.entryGraph entry) preexisting)
    (s : Storage p) (h : LaunchContract p s.byteBase)
    (final : State (Observed.Term (BitVec 32))) (before : Nat → Nat → BitVec 8)
    (result : NativeResult M p s final) : NativeBytes M p s final before := by
  refine ⟨?_,byte_frame _ p s final before⟩
  intro ch on r q hr hq j
  rw [byte_value _ p s h final before ch on r q hr hq j]
  rw [result.1 ch on r q hr hq]

#axiomsClean native_bytes
#axiomsClean logical_input_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluBackward.KernelProof
