import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.KernelResult

namespace VeriTile.Bench.Repairs.CacheTransform.Single
open VeriTile.Triton FP.Structural KernelBinding KernelResult
noncomputable section
open Classical
set_option maxRecDepth 20000
set_option maxHeartbeats 1000000

inductive Side where
  | cos | sin

def Side.metadata (side : Side) (r : Request) :=
  match side with | .cos => r.cos | .sin => r.sin
def Side.capacity (side : Side) (r : Request) :=
  match side with | .cos => r.cosRows | .sin => r.sinRows
def Side.parameters (side : Side) (r : Request) : Parameters :=
  match side with | .cos => KernelSource.cosine r | .sin => KernelSource.sine r

theorem Side.contiguous (side : Side) (h : InputContract r) :
    (side.metadata r).isContiguous=Bool.true := by
  cases side
  · exact cos_contiguous h
  · exact sin_contiguous h
theorem Side.noNeg (side : Side) (h : InputContract r) :
    (side.metadata r).isNeg=Bool.false := by
  cases side
  · exact cos_noNeg h
  · exact sin_noNeg h
theorem Side.shape (side : Side) (h : InputContract r) :
    (side.metadata r).shape=[side.capacity r,r.width] := by
  cases side
  · exact cos_shape h
  · exact sin_shape h
theorem Side.valid (side : Side) (h : InputContract r) :
    ValidLengths r.mode r.naturalLengths (side.capacity r) := by
  cases side
  · exact (admitted_lengths r h).1
  · exact (admitted_lengths r h).2
theorem Side.extent (side : Side) (h : InputContract r) :
    side.capacity r*r.width*4<2^63 := by
  cases side
  · exact h.2.2.1
  · exact h.2.2.2.1

@[simp] theorem Side.count (side : Side) : (side.parameters r).count=r.count := by cases side <;> rfl
@[simp] theorem Side.rowBytes (side : Side) : (side.parameters r).rowBytes=r.width*4 := by cases side <;> rfl
@[simp] theorem Side.start (side : Side) : (side.parameters r).start=0 := by cases side <;> rfl
@[simp] theorem Side.block (side : Side) : (side.parameters r).block=256 := by cases side <;> rfl

/-- One call's actual byte windows; no other call or temporary table occurs. -/
def initial (side : Side) (r : Request) (cache : Nat → Nat) (rows : Nat → Int)
    (s : State α) : State α :=
  { s with mem := fun region offset =>
      if region=(side.parameters r).cache then .mk .nat (cache offset)
      else if region=(side.parameters r).rows then .mk .int (rows offset)
      else s.mem region offset }

theorem complete [Inhabited α] (M : Algebra α) (side : Side) (r : Request)
    (h : InputContract r) (cache : Nat → Nat) (rows : Nat → Int) (s : State α)
    (table : 0<r.count → ∀ q,q<r.outputRows → rows q=(tableRows r.mode r.naturalLengths q : Int)) :
    let first := initial side r cache rows s
    ∃ t,KernelSource.launch M (side.parameters r) first (blocks r.count)=some t ∧
      (∀ o,o<r.count → t.mem (side.parameters r).output o=.mk .nat
        (byteTarget (side.parameters r) (tableRows r.mode r.naturalLengths) cache o)) ∧
      (∀ (region : RegionName) o,region≠(side.parameters r).output ∨ r.count≤o →
        t.mem region o=first.mem region o) := by
  let first := initial side r cache rows s
  have input : 0<(side.parameters r).count → GatherInput (side.parameters r) first
      r.outputRows (side.capacity r) (tableRows r.mode r.naturalLengths) cache := by
    apply KernelSource.gather_input r h (side.parameters r) (side.capacity r) cache rows first
      (side.count) (side.rowBytes) _ _ _ table _ (side.valid h) (side.extent h)
    · cases side <;> simp [Side.parameters,KernelSource.cosine,KernelSource.sine]
    · cases side <;> simp [Side.parameters,KernelSource.cosine,KernelSource.sine]
    · intro q hq
      cases side <;> simp [first,initial,Side.parameters,KernelSource.cosine,KernelSource.sine]
    · intro o ho; simp [first,initial]
  obtain ⟨values,frame⟩ := gather_correct_total M (side.parameters r) first r.outputRows
    (side.capacity r) (tableRows r.mode r.naturalLengths) cache input
  refine ⟨gather M (side.parameters r) first,?_,?_,?_⟩
  · simpa only [gather,Side.count] using KernelSource.launch_success M r h (side.parameters r)
      side.count side.start side.block first (blocks r.count) (Nat.le_refl _)
  · simpa only [Side.count] using values
  · simpa only [Side.count] using frame

/-- Freshness required only during this gather. In particular a later output
is free to reuse a temporary table from an earlier, already completed call. -/
structure Allocation (side : Side) (x : Cache (side.metadata r)) (h : InputContract r)
    (table : Table r bytes) (z : Output r) : Prop where
  cache : ∀ o,o<4*(side.metadata r).numel →
    ¬z.contains ((x.view (side.contiguous h)).base+o)
  table : 0<r.count → ∀ o,o<8*r.outputRows →
    ¬z.contains (table.metadata.dataPtr.toNat+o)

def realize (z : Output r) (before output : Nat → α) (address : Nat) : α :=
  if z.contains address then output (address-z.view.base) else before address

theorem realize_output (z : Output r) (before output : Nat → α) (o : Nat) (ho : o<r.count) :
    realize z before output (z.view.base+o)=output o := by
  have inside : z.contains (z.view.base+o) := by simp only [Output.contains,inside]; omega
  simp [realize,inside]
theorem realize_frame (z : Output r) (before output : Nat → α) (address : Nat)
    (outside : ¬z.contains address) : realize z before output address=before address := by
  simp [realize,outside]

def realInitial (side : Side) (x : Cache (side.metadata r)) (h : InputContract r)
    (table : Table r bytes) (s : State ℝ) : State ℝ :=
  initial side r (fun byte => (x.view (side.contiguous h)).address (byte/4)) table.read s
def nativeInitial (side : Side) (x : Cache (side.metadata r)) (h : InputContract r)
    (table : Table r bytes) (s : State Unit) : State Unit :=
  initial side r (fun o => (bytes ((x.view (side.contiguous h)).base+o)).toNat) table.read s
def realAfter (side : Side) (z : Output r) (before : Nat → ℝ) (t : State ℝ) : Nat → ℝ :=
  realize z before (fun o => before ((t.mem (side.parameters r).output o).read .nat))
def nativeAfter (side : Side) (z : Output r) (before : Bytes) (t : State Unit) : Bytes :=
  realize z before (fun o => BitVec.ofNat 8 ((t.mem (side.parameters r).output o).read .nat))

theorem real_output (side : Side) (z : Output r) (before : Nat → ℝ) (t : State ℝ) :
    realTensor z (realAfter side z before t)=
      observedTensor t (side.parameters r).output r.outputRows r.width .fp32 (realObserve before) := by
  apply congrArg List.ofFn
  funext row col
  unfold Cache.real
  rw [←z.cache.address z.contiguous _ (by rw [z.size]; exact output_word_bound r row col)]
  change realize z before _ (z.view.base+4*(row.val*r.width+col.val))=_
  rw [realize_output _ _ _ _ (by
    have := output_byte_bound r row col ⟨0,by decide⟩
    simpa only [Nat.add_zero,Nat.mul_comm] using this)]
  simp only [realObserve,Storage.bytes,Nat.add_zero,Nat.mul_comm]

theorem native_output (side : Side) (z : Output r) (before : Bytes) (t : State Unit) :
    wordTensor z (nativeAfter side z before t)=
      observedTensor t (side.parameters r).output r.outputRows r.width .fp32 nativeObserve := by
  apply congrArg List.ofFn
  funext row col
  rw [←z.cache.word_direct z.contiguous z.noNeg _ _
    (by rw [z.size]; exact output_word_bound r row col)]
  unfold read32 nativeObserve
  apply congrArg assemble32
  funext b
  change (realize z before _ (z.view.base+4*(row.val*r.width+col.val)+b.val)).toNat=_
  rw [Nat.add_assoc,realize_output _ _ _ _ (by
    simpa only [Nat.mul_comm] using output_byte_bound r row col b)]
  simp only [BitVec.toNat_ofNat,Storage.bytes,Nat.mul_comm]

#axiomsClean complete
#axiomsClean realize_output
#axiomsClean realize_frame
#axiomsClean real_output
#axiomsClean native_output
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CacheTransform.Single
