import bench.repairs.tritonbench_g.swiglu_backward.Proofs.Kernel

namespace VeriTile.Bench.Repairs.SwigluBackward.KernelProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
noncomputable section
open Classical
set_option maxHeartbeats 2000000

def accessAddress (p : Prepared α call entry preexisting) (s : Storage p)
    (a : Access) (r q : Nat) : Nat :=
  s.byteBase (a.view p).allocation+4*((a.view p).base+r*(a.view p).rowStride+q)

def sourceRegion : Access → RegionName
  | .x => "X" | .y => "Y" | .dout => "DOUT" | .dx => "DX" | .dy => "DY" | .out => "OUT"

/-- Actual physical initial memory is shared by all storage views. Only
ordinary bounded Tensor storage laws are used; no computed result is assumed. -/
theorem initial_read [Inhabited α] (p : Prepared α call entry preexisting) (s : Storage p)
    (hc : checks p = Bool.true) (h : LaunchContract p s.byteBase)
    (a : Access) (on : a.active call.recompute) (r q : Nat)
    (hr : r < p.batchShape.prod) (hq : q < p.ncols) :
    (p.state.mem (sourceRegion a) (r*(a.view p).rowStride+q)).read .real =
      s.preparedRead (s.device (a.view p).allocation) (accessAddress p s a r q) := by
  have bound := access_bound p (checkedDomain p s hc h) a on r q hr hq
  have physical := s.preparedBinding _ _ bound
  cases a <;> simpa [sourceRegion,Prepared.state,accessAddress,Access.view,Prepared.view,
    Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using physical

theorem original_read (p : Prepared α call entry preexisting) (s : Storage p)
    (t : Tensor) (member : t ∈ [call.input,call.upstream]) (i : Nat) (hi : i < t.shape.prod) :
    t.read entry i = s.entryRead t.cudaDevice (s.byteBase t.allocation+4*t.offset i) := by
  change entry t.allocation (t.offset i) = _
  rw [s.entryBinding _ _ (s.originalBounds t member i hi),s.originalDevice t member]

/-- Both pointer offsets and logical input values are tied to the same
initial state used by the complete source interpreter. -/
def InputBinding [Inhabited α] (p : Prepared α call entry preexisting) (s : Storage p) : Prop :=
  ∀ (config : Config) (r b lane : Nat), r < p.batchShape.prod → b < (p.ncols+config.block-1)/config.block →
    lane < config.block → b*config.block+lane < p.ncols →
    let pointer := fun a : Access => (pointer64 (s.byteBase (a.view p).allocation)
      (a.view p).base (row32 r (a.view p).rowStride) (cols32 b config.block lane)).toNat
    (s.preparedRead (s.device p.xy.matrix.allocation) (pointer .x) =
      call.input.read entry (r*(2*p.ncols)+(b*config.block+lane))) ∧
    (s.preparedRead (s.device p.xy.matrix.allocation) (pointer .y) =
      call.input.read entry (r*(2*p.ncols)+(p.ncols+(b*config.block+lane)))) ∧
    (s.preparedRead (s.device p.dout.matrix.allocation) (pointer .dout) =
      call.upstream.read entry (r*p.ncols+(b*config.block+lane)))

theorem input_binding [Inhabited α] (p : Prepared α call entry preexisting) (s : Storage p)
    (hc : checks p = Bool.true) (h : LaunchContract p s.byteBase) : InputBinding p s := by
  intro config r b lane hr hb hl active
  have index := checked_indices p s hc h
  have logical := prepared_reads p (checkedDomain p s hc h) r (b*config.block+lane) hr active
  have physical (a : Access) (on : a.active call.recompute) :=
    (initial_read p s hc h a on r (b*config.block+lane) hr active).symm
  have pointer (a : Access) (on : a.active call.recompute) :=
    (index a on config r b lane hr hb hl).2 active
  dsimp only
  refine ⟨?_,?_,?_⟩
  · rw [(pointer .x (Or.inl (by decide))).1]
    exact (physical .x (Or.inl (by decide))).trans logical.1
  · rw [(pointer .y (Or.inl (by decide))).1]
    exact (physical .y (Or.inl (by decide))).trans logical.2.1
  · rw [(pointer .dout (Or.inl (by decide))).1]
    exact (physical .dout (Or.inl (by decide))).trans logical.2.2

def outputAddress (p : Prepared α call entry preexisting) (s : Storage p)
    (ch : Channel) (r q : Nat) : Nat :=
  s.byteBase (p.location ch r q).1+4*(p.location ch r q).2

theorem output_address_unique (p : Prepared α call entry preexisting) (s : Storage p)
    (h : LaunchContract p s.byteBase) (ch ch' : Channel) (r q r' q' : Nat)
    (on : enabled call.recompute ch) (on' : enabled call.recompute ch')
    (hr : r < p.batchShape.prod) (hq : q < p.ncols) (hr' : r' < p.batchShape.prod) (hq' : q' < p.ncols)
    (same : outputAddress p s ch r q = outputAddress p s ch' r' q') :
    ch = ch' ∧ r = r' ∧ q = q' := by
  have nonempty : 0 < p.batchShape.prod := Nat.zero_lt_of_lt hr
  have outOn : enabled call.recompute .out → call.recompute = Bool.true := by simp [enabled]
  have separate (on : enabled call.recompute .out) :=
    (h.outputSeparate nonempty (outOn on)).2.2
  cases ch <;> cases ch' <;> simp only [outputAddress,Prepared.location,Prepared.view] at same
  · exact ⟨rfl,gradient_unique p h r q r' q' hr (by omega) hr' (by omega) (by omega)⟩
  · have unique := gradient_unique p h r q r' (p.ncols+q') hr (by omega) hr' (by omega) (by omega)
    omega
  · exact False.elim (separate_bytes _ _ _ _ _ _ (separate on') r' q' r q
      hr' hq' hr (by omega) 0 0 (by decide) (by decide) (by omega))
  · have unique := gradient_unique p h r (p.ncols+q) r' q' hr (by omega) hr' (by omega) (by omega)
    omega
  · have unique := gradient_unique p h r (p.ncols+q) r' (p.ncols+q') hr (by omega) hr' (by omega) (by omega)
    exact ⟨rfl,unique.1,by omega⟩
  · exact False.elim (separate_bytes _ _ _ _ _ _ (separate on') r' q' r (p.ncols+q)
      hr' hq' hr (by omega) 0 0 (by decide) (by decide) (by omega))
  · exact False.elim (separate_bytes _ _ _ _ _ _ (separate on) r q r' q'
      hr hq hr' (by omega) 0 0 (by decide) (by decide) (by omega))
  · exact False.elim (separate_bytes _ _ _ _ _ _ (separate on) r q r' (p.ncols+q')
      hr hq hr' (by omega) 0 0 (by decide) (by decide) (by omega))
  · exact ⟨rfl,SwigluFwd.row_unique _ _ _ r q r' q' (h.outputUnique (outOn on))
      hr hq hr' hq' (by omega)⟩

def PhysicalWritten (p : Prepared α call entry preexisting) (s : Storage p) (device address : Nat) : Prop :=
  ∃ ch r q, enabled call.recompute ch ∧ r < p.batchShape.prod ∧ q < p.ncols ∧
    device = call.input.cudaDevice ∧ address = outputAddress p s ch r q

/-- Physical FP32-word writeback at aligned byte starts. This uses absolute
device/byte keys, so all aliased Tensor views see the same write. The byte
extension in `Bytes` supplies the complete physical byte-memory observation. -/
def physicalCommit [Inhabited α] (p : Prepared α call entry preexisting) (s : Storage p)
    (final : State α) (device address : Nat) : α :=
  if h : PhysicalWritten p s device address then
    let ch := h.choose
    let r := h.choose_spec.choose
    let q := h.choose_spec.choose_spec.choose
    (final.mem (p.layout.output ch) (r*p.layout.stride ch+q)).read .real
  else s.preparedRead device address

theorem physical_value [Inhabited α] (p : Prepared α call entry preexisting) (s : Storage p)
    (h : LaunchContract p s.byteBase) (final : State α) (ch : Channel)
    (on : enabled call.recompute ch) (r q : Nat) (hr : r < p.batchShape.prod) (hq : q < p.ncols) :
    physicalCommit p s final call.input.cudaDevice (outputAddress p s ch r q) =
      (final.mem (p.layout.output ch) (r*p.layout.stride ch+q)).read .real := by
  have written : PhysicalWritten p s call.input.cudaDevice (outputAddress p s ch r q) :=
    ⟨ch,r,q,on,hr,hq,rfl,rfl⟩
  unfold physicalCommit
  rw [dif_pos written]
  have witness := written.choose_spec.choose_spec.choose_spec
  have unique := output_address_unique p s h _ ch _ _ r q witness.1 on
    witness.2.1 witness.2.2.1 hr hq witness.2.2.2.2.symm
  dsimp only
  rw [unique.2.2,unique.2.1,unique.1]

theorem physical_frame [Inhabited α] (p : Prepared α call entry preexisting) (s : Storage p)
    (final : State α) (device address : Nat) (outside : ¬PhysicalWritten p s device address) :
    physicalCommit p s final device address = s.preparedRead device address := by
  exact dif_neg outside

/-- All effective outputs and every exterior aligned physical word.
There is no allocation-ID separation premise. -/
def PhysicalResult [Inhabited α] (f : Channel → α → α → α → α)
    (p : Prepared α call entry preexisting) (s : Storage p) (final : State α) : Prop :=
  (∀ ch, enabled call.recompute ch → ∀ r q, r < p.batchShape.prod → q < p.ncols →
    physicalCommit p s final call.input.cudaDevice (outputAddress p s ch r q) = f ch
      (call.input.read entry (r*(2*p.ncols)+q))
      (call.input.read entry (r*(2*p.ncols)+(p.ncols+q)))
      (call.upstream.read entry (r*p.ncols+q))) ∧
  (∀ device address, address % 4 = 0 → ¬PhysicalWritten p s device address →
    physicalCommit p s final device address = s.preparedRead device address)

theorem physical_result [Inhabited α] (f : Channel → α → α → α → α)
    (p : Prepared α call entry preexisting) (s : Storage p)
    (hc : checks p = Bool.true) (h : LaunchContract p s.byteBase)
    (final : State α) (result : Result f p final) : PhysicalResult f p s final := by
  refine ⟨?_,fun device address _ => physical_frame p s final device address⟩
  intro ch on r q hr hq
  rw [physical_value p s h final ch on r q hr hq]
  rw [←commit_value p (checkedDomain p s hc h) final ch on r q hr hq]
  exact result.1 ch on r q hr hq

#axiomsClean input_binding
#axiomsClean physical_result
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluBackward.KernelProof
