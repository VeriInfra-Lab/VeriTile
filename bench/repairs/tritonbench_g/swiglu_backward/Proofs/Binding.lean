import bench.repairs.tritonbench_g.swiglu_backward.Proofs.Contract
import bench.repairs.tritonbench_g.swiglu_fwd.Proofs.Binding

namespace VeriTile.Bench.Repairs.SwigluBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole (Matrix)
noncomputable section
set_option maxHeartbeats 4000000

structure Storage (p : Prepared α call entry preexisting) where
  capacity : Nat → Nat
  originalBounds : ∀ t ∈ [call.input,call.upstream], ∀ k, k < t.shape.prod → t.offset k < capacity t.allocation
  matrixBounds : ∀ m, (m ∈ [p.xy.matrix,p.dout.matrix,p.grad.matrix] ∨
      (call.recompute = Bool.true ∧ m = p.out.matrix)) →
    ∀ r q, r < m.rows → q < m.cols → m.address r q < capacity m.allocation
  byteBase : Nat → Nat
  allocation64 : ∀ a, byteBase a+4*capacity a ≤ 2^64
  device : Nat → Nat
  originalDevice : ∀ t ∈ [call.input,call.upstream], device t.allocation = t.cudaDevice
  matrixDevice : ∀ m, (m ∈ [p.xy.matrix,p.dout.matrix,p.grad.matrix] ∨
      (call.recompute = Bool.true ∧ m = p.out.matrix)) → device m.allocation = m.cudaDevice
  entryRead : Nat → Nat → α
  preparedRead : Nat → Nat → α
  entryBinding : ∀ a k, k < capacity a → entry a k = entryRead (device a) (byteBase a+4*k)
  preparedBinding : ∀ a k, k < capacity a → p.memory a k = preparedRead (device a) (byteBase a+4*k)

theorem separate_bytes (b : Nat → Nat) (rows wa wb : Nat)
    (a a' : VeriTile.Bench.TritonBenchG.SwigluFwd.Whole.Matrix)
    (separate : SpanSeparate b rows wa wb a a')
    (r q r' q' : Nat) (hr : r < rows) (hq : q < wa) (hr' : r' < rows) (hq' : q' < wb)
    (i j : Nat) (hi : i < 4) (hj : j < 4) :
    b a.allocation+4*(a.base+r*a.rowStride+q)+i ≠
      b a'.allocation+4*(a'.base+r'*a'.rowStride+q')+j := by
  have left := SwigluFwd.span_bound a.base rows a.rowStride wa r q hr hq
  have right := SwigluFwd.span_bound a'.base rows a'.rowStride wb r' q' hr' hq'
  simp only [SpanSeparate,matrixEnd,matrixStart] at separate
  rcases separate with before | after <;> omega

theorem separate_locations (b : Nat → Nat) (rows wa wb : Nat)
    (a a' : VeriTile.Bench.TritonBenchG.SwigluFwd.Whole.Matrix)
    (separate : SpanSeparate b rows wa wb a a')
    (r q r' q' : Nat) (hr : r < rows) (hq : q < wa) (hr' : r' < rows) (hq' : q' < wb) :
    (a.allocation,a.base+r*a.rowStride+q) ≠
      (a'.allocation,a'.base+r'*a'.rowStride+q') := by
  intro same
  have address := congrArg (fun loc : Nat × Nat => b loc.1+4*loc.2) same
  exact separate_bytes b rows wa wb a a' separate r q r' q' hr hq hr' hq' 0 0 (by decide) (by decide)
    (by simpa using address)

structure Shapes (p : Prepared α call entry preexisting) : Prop where
  upstream : p.dout.matrix.rows = p.batchShape.prod ∧ p.dout.matrix.cols = p.ncols
  gradient : p.grad.matrix.rows = p.batchShape.prod ∧ p.grad.matrix.cols = 2*p.ncols
  gradientStride : p.grad.matrix.lastStride = 1
  output : call.recompute = Bool.true → p.out.matrix.rows = p.batchShape.prod ∧ p.out.matrix.cols = p.ncols
  outputStride : call.recompute = Bool.true → p.out.matrix.lastStride = 1

theorem checked_shapes (p : Prepared α call entry preexisting) (hc : checks p = Bool.true) : Shapes p := by
  simp only [checks,Bool.and_eq_true,decide_eq_true_eq] at hc
  refine ⟨?_,?_,hc.1.1.2,?_,?_⟩
  · simpa only [p.inputRows] using hc.1.1.1.1
  · rcases Option.eq_none_or_eq_some call.gradient with hg | ⟨t,hg⟩
    · exact p.grad.emptyShape hg
    · simpa [hg,p.inputRows,p.inputCols] using hc.1.1.1.2
  · intro on
    rcases Option.eq_none_or_eq_some call.output with ho | ⟨t,ho⟩
    · exact p.out.emptyShape (by simp [on,ho])
    ·
      have h : (p.out.matrix.rows = p.xy.matrix.rows ∧ p.out.matrix.cols = p.ncols) ∧
          p.out.matrix.lastStride = 1 := by simpa [on,ho] using hc.2
      exact ⟨h.1.1.trans p.inputRows,h.1.2⟩
  · intro on
    have h := hc.2
    simp only [on,ite_true,Bool.and_eq_true,decide_eq_true_eq] at h
    exact h.2

theorem gradient_unique (p : Prepared α call entry preexisting) (h : LaunchContract p b)
    (r q r' q' : Nat) (hr : r < p.batchShape.prod) (hq : q < 2*p.ncols)
    (hr' : r' < p.batchShape.prod) (hq' : q' < 2*p.ncols)
    (same : p.grad.matrix.base+r*p.grad.matrix.rowStride+q =
      p.grad.matrix.base+r'*p.grad.matrix.rowStride+q') : r = r' ∧ q = q' :=
  SwigluFwd.row_unique _ _ _ _ _ _ _ h.gradientUnique hr hq hr' hq' (by omega)

theorem output_unique (p : Prepared α call entry preexisting) (h : LaunchContract p b)
    (ch ch' : Channel) (r q r' q' : Nat) (on : enabled call.recompute ch) (on' : enabled call.recompute ch')
    (hr : r < p.batchShape.prod) (hq : q < p.ncols) (hr' : r' < p.batchShape.prod) (hq' : q' < p.ncols)
    (same : p.location ch r q = p.location ch' r' q') : ch = ch' ∧ r = r' ∧ q = q' := by
  have nonempty : 0 < p.batchShape.prod := Nat.zero_lt_of_lt hr
  have outOn : enabled call.recompute .out → call.recompute = Bool.true := by simp [enabled]
  have separation : call.recompute = Bool.true →
      SpanSeparate b p.batchShape.prod p.ncols (2*p.ncols) p.out.matrix p.grad.matrix :=
    fun on => (h.outputSeparate nonempty on).2.2
  cases ch <;> cases ch' <;> simp only [Prepared.location,Prepared.view,Prod.mk.injEq] at same
  · have unique := gradient_unique p h r q r' q' hr (by omega) hr' (by omega) same.2
    exact ⟨rfl,unique⟩
  · have unique := gradient_unique p h r q r' (p.ncols+q') hr (by omega) hr' (by omega) (by omega)
    omega
  · exact False.elim (separate_locations b _ _ _ _ _ (separation (outOn on'))
      r' q' r q hr' hq' hr (by omega) (Prod.ext same.1.symm same.2.symm))
  · have unique := gradient_unique p h r (p.ncols+q) r' q' hr (by omega) hr' (by omega) (by omega)
    omega
  · have unique := gradient_unique p h r (p.ncols+q) r' (p.ncols+q') hr (by omega) hr' (by omega) (by omega)
    exact ⟨rfl,unique.1,by omega⟩
  · exact False.elim (separate_locations b _ _ _ _ _ (separation (outOn on'))
      r' q' r (p.ncols+q) hr' hq' hr (by omega) (Prod.ext same.1.symm (by omega)))
  · exact False.elim (separate_locations b _ _ _ _ _ (separation (outOn on))
      r q r' q' hr hq hr' (by omega) (Prod.ext same.1 same.2))
  · exact False.elim (separate_locations b _ _ _ _ _ (separation (outOn on))
      r q r' (p.ncols+q') hr hq hr' (by omega) (Prod.ext same.1 (by omega)))
  · have unique := SwigluFwd.row_unique _ _ _ r q r' q' (h.outputUnique (outOn on)) hr hq hr' hq' (by omega)
    exact ⟨rfl,unique⟩

def checkedDomain (p : Prepared α call entry preexisting) (s : Storage p)
    (hc : checks p = Bool.true) (h : LaunchContract p s.byteBase) : Domain p where
  positiveColumns := h.positive
  sourceChecks := hc
  upstreamShape := (checked_shapes p hc).upstream
  gradientShape := (checked_shapes p hc).gradient
  outputShape := (checked_shapes p hc).output
  sameDevice := ⟨h.sameDevice .dout (Or.inl (by decide)), h.sameDevice .dx (Or.inl (by decide)),
    fun on => h.sameDevice .out (Or.inr on)⟩
  capacity := s.capacity
  originalBounds := s.originalBounds
  inputBounds := by
    intro r q hr hq
    exact s.matrixBounds p.xy.matrix (by left; simp) r q (by rwa [p.inputRows]) (by rwa [p.inputCols])
  upstreamBounds := by
    intro r q hr hq
    exact s.matrixBounds p.dout.matrix (by left; simp) r q
      (by rwa [(checked_shapes p hc).upstream.1]) (by rwa [(checked_shapes p hc).upstream.2])
  outputBounds := by
    intro ch on r q hr hq
    have shapes := checked_shapes p hc
    have gradBound : ∀ v, v < 2*p.ncols → p.grad.matrix.base+r*p.grad.matrix.rowStride+v < s.capacity p.grad.matrix.allocation := by
      intro v hv
      have bound := s.matrixBounds p.grad.matrix (by left; simp) r v
        (by rwa [shapes.gradient.1]) (by rwa [shapes.gradient.2])
      simpa [Matrix.address,shapes.gradientStride] using bound
    cases ch with
    | dx => exact gradBound q (by omega)
    | dy => simpa [Prepared.location,Prepared.view,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using gradBound (p.ncols+q) (by omega)
    | out =>
      have enabled : call.recompute = Bool.true := by simpa [enabled] using on
      have bound := s.matrixBounds p.out.matrix (Or.inr ⟨enabled,rfl⟩) r q
        (by rwa [(shapes.output enabled).1]) (by rwa [(shapes.output enabled).2])
      simpa [Prepared.location,Prepared.view,Matrix.address,shapes.outputStride enabled] using bound
  outputUnique := output_unique p h
  inputOwnership := by
    intro ch on r q r' q' hr hq hr' hq' same
    have nonempty : 0 < p.batchShape.prod := Nat.zero_lt_of_lt hr
    have unit := p.xy.unit_offset r' q' (by rwa [p.inputRows]) (by rwa [p.inputCols])
    simp only [Matrix.address,unit] at same
    cases ch with
    | dx =>
      exact separate_locations s.byteBase _ _ _ _ _ (h.gradientSeparate nonempty).1
        r q r' q' hr (by omega) hr' hq' same
    | dy =>
      exact separate_locations s.byteBase _ _ _ _ _ (h.gradientSeparate nonempty).1
        r (p.ncols+q) r' q' hr (by omega) hr' hq' (by
          simpa [Prepared.location,Prepared.view,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using same)
    | out =>
      have enabled : call.recompute = Bool.true := by simpa [enabled] using on
      exact separate_locations s.byteBase _ _ _ _ _ (h.outputSeparate nonempty enabled).1
        r q r' q' hr hq hr' hq' same
  upstreamOwnership := by
    intro ch on r q r' q' hr hq hr' hq' same
    have nonempty : 0 < p.batchShape.prod := Nat.zero_lt_of_lt hr
    have unit := p.dout.unit_offset r' q' (by rwa [(checked_shapes p hc).upstream.1])
      (by rwa [(checked_shapes p hc).upstream.2])
    simp only [Matrix.address,unit] at same
    cases ch with
    | dx =>
      exact separate_locations s.byteBase _ _ _ _ _ (h.gradientSeparate nonempty).2
        r q r' q' hr (by omega) hr' hq' same
    | dy =>
      exact separate_locations s.byteBase _ _ _ _ _ (h.gradientSeparate nonempty).2
        r (p.ncols+q) r' q' hr (by omega) hr' hq' (by
          simpa [Prepared.location,Prepared.view,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using same)
    | out =>
      have enabled : call.recompute = Bool.true := by simpa [enabled] using on
      exact separate_locations s.byteBase _ _ _ _ _ (h.outputSeparate nonempty enabled).2.1
        r q r' q' hr hq hr' hq' same

#axiomsClean checkedDomain
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluBackward
