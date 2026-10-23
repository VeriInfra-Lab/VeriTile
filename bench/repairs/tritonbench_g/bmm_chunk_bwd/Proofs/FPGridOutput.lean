import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.FPOutput
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.FPGrid

namespace VeriTile.Bench.Repairs.BmmChunkBwd.FPGridOutput
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open FPDefault FPReference
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Independent blocked grid recurrence. Each block reads memory left by the
preceding block; the disjoint output boundary makes those input reads invariant.
This definition contains no source or interpreter execution. -/
def reference (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (c : Tuning) :
    List HostFP.Pid → State α → State α
  | [], s => s
  | q::rest,s => reference M D h c rest
      (FPFinish.write M D c.BM c.BN c.BCS (HostContract.parameters h) (HostFP.launch h c q s))

/-- Recurrence locality follows from its explicit load operands. -/
theorem expected_congr (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (s t : State α) (hp : s.pids = t.pids)
    (ha : s.mem p.A = t.mem p.A) (hd : s.mem p.Dout = t.mem p.Dout)
    (hr : s.mem p.Res = t.mem p.Res) :
    expected M D BM BN BCS p s = expected M D BM BN BCS p t := by
  have left (n : Nat) : leftBlock M BM BN BCS p s n = leftBlock M BM BN BCS p t n := by
    funext i
    simp only [leftBlock,row,length,chunk,dBase,head,hp,hd]
  have right (n : Nat) : rightBlock M BN BCS p s n = rightBlock M BN BCS p t n := by
    funext i
    simp only [rightBlock,column,length,chunk,aBase,head,hp,ha]
  have accum (n : Nat) : carry M D BM BN BCS p s n = carry M D BM BN BCS p t n := by
    induction n with
    | zero => rfl
    | succ n ih => simp only [carry,ih,left n,right n]
  funext i
  simp only [expected,accum,length,chunk,offset,head,row,column,hp,hr]

theorem write_mem_congr (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (s t : State α) (hm : s.mem = t.mem) (hp : s.pids = t.pids) :
    (FPFinish.write M D BM BN BCS p s).mem = (FPFinish.write M D BM BN BCS p t).mem := by
  rcases s with ⟨sm,sr,sp,snp,su⟩
  rcases t with ⟨tm,tr,tp,tnp,tu⟩
  dsimp only at hm hp
  subst tm
  subst tp
  unfold FPFinish.write
  apply FPFinish.masked_mem_congr
  · rfl
  · intro i hi
    exact congrArg (Cell.mk .fp32) (congrFun (expected_congr M D BM BN BCS p
      ⟨sm,sr,sp,snp,su⟩ ⟨sm,tr,sp,tnp,tu⟩ rfl rfl rfl rfl) i)

theorem reference_mem_congr (M : Algebra α) (D : DotPrimitive α) (h : HostCall)
    (c : Tuning) (jobs : List HostFP.Pid) (s t : State α) (hm : s.mem = t.mem) :
    (reference M D h c jobs s).mem = (reference M D h c jobs t).mem := by
  induction jobs generalizing s t with
  | nil => exact hm
  | cons q rest ih =>
    apply ih
    exact write_mem_congr M D _ _ _ _ _ _ hm rfl

/-- The complete actual typed grid produces exactly the independent blocked
memory recurrence, including all typed output writes. Arbitrary initial alias
patterns are retained here; logical host copy composition is a separate layer. -/
theorem run_memory (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (c : Tuning)
    (hp : FPGrid.DefaultPrecision h) (jobs : List HostFP.Pid) (s : State α) :
    ∃ final, HostFP.run (backend M D) h c Bool.false jobs s = some final ∧
      final.mem = (reference M D h c jobs s).mem := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons q rest ih =>
    obtain ⟨mid,hm,hmem⟩ := FPOutput.source_memory M D c.BM c.BN c.BCS
      (HostContract.parameters h) (HostFP.launch h c q s)
    rw [← FPGrid.source_default h c hp] at hm
    obtain ⟨final,hf,hfm⟩ := ih mid
    refine ⟨final,?_,?_⟩
    · simp only [HostFP.run,hm,bind,Option.bind_some]
      exact hf
    · exact hfm.trans (reference_mem_congr M D h c rest _ _ hmem)

/-- Exact grid footprint, with no change to inactive holes or any other region. -/
theorem reference_frame (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (c : Tuning)
    (jobs : List HostFP.Pid) (s : State α) (r : RegionName) (a : Nat)
    (miss : r ≠ h.Out ∨ ∀ q ∈ jobs, ∀ i,
      active (HostFP.launch h c q s) c.BM c.BN (HostContract.parameters h) i →
      a ≠ offset (HostFP.launch h c q s) c.BM c.BN (HostContract.parameters h)
        h.outStrides.batch h.outStrides.sequence (h.headStride h.outStrides.head) h.outStrides.feature i) :
    (reference M D h c jobs s).mem r a = s.mem r a := by
  induction jobs generalizing s with
  | nil => rfl
  | cons q rest ih =>
    rw [reference,ih]
    · exact FPOutput.frame M D c.BM c.BN c.BCS (HostContract.parameters h)
        (HostFP.launch h c q s) r a (miss.imp_right (fun hh i hi => hh q (by simp) i hi))
    · rcases miss with hr | hx
      · exact Or.inl hr
      · right
        intro u hu i hi
        exact hx u (List.mem_cons_of_mem _ hu) i hi

/-- Exact output geometry for the direct grid. -/
def LocalInjective (h : HostCall) (c : Tuning) (jobs : List HostFP.Pid) (s : State α) : Prop :=
  ∀ q ∈ jobs, FPOutput.ActiveInjective (HostFP.launch h c q s) c.BM c.BN (HostContract.parameters h)

def SeparateWrites (h : HostCall) (c : Tuning) (jobs : List HostFP.Pid) (s : State α) : Prop :=
  ∀ q ∈ jobs, ∀ q' ∈ jobs, q ≠ q' → ∀ i j,
    active (HostFP.launch h c q s) c.BM c.BN (HostContract.parameters h) i →
    active (HostFP.launch h c q' s) c.BM c.BN (HostContract.parameters h) j →
    offset (HostFP.launch h c q s) c.BM c.BN (HostContract.parameters h)
      h.outStrides.batch h.outStrides.sequence (h.headStride h.outStrides.head) h.outStrides.feature i ≠
    offset (HostFP.launch h c q' s) c.BM c.BN (HostContract.parameters h)
      h.outStrides.batch h.outStrides.sequence (h.headStride h.outStrides.head) h.outStrides.feature j

theorem reference_readback (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (c : Tuning)
    (jobs : List HostFP.Pid) (s : State α) (nd : jobs.Nodup)
    (injective : LocalInjective h c jobs s) (separate : SeparateWrites h c jobs s)
    (inputs : HostGrid.InputsSeparate (HostContract.parameters h))
    (q : HostFP.Pid) (hq : q ∈ jobs) (i : TileIndex [c.BM,c.BN])
    (hi : active (HostFP.launch h c q s) c.BM c.BN (HostContract.parameters h) i) :
    (reference M D h c jobs s).mem h.Out
      (offset (HostFP.launch h c q s) c.BM c.BN (HostContract.parameters h)
        h.outStrides.batch h.outStrides.sequence (h.headStride h.outStrides.head) h.outStrides.feature i) =
      .mk .fp32 (expected M D c.BM c.BN c.BCS (HostContract.parameters h) (HostFP.launch h c q s) i) := by
  induction jobs generalizing s with
  | nil => simp at hq
  | cons u rest ih =>
    let mid := FPFinish.write M D c.BM c.BN c.BCS (HostContract.parameters h) (HostFP.launch h c u s)
    obtain ⟨hu,hn⟩ := List.nodup_cons.mp nd
    by_cases he : q = u
    · subst q
      rw [reference,reference_frame]
      · exact FPOutput.readback M D c.BM c.BN c.BCS (HostContract.parameters h)
          (HostFP.launch h c u s) i hi (injective u (by simp))
      · right
        intro v hv j hj
        exact separate u (by simp) v (by simp [hv]) (fun he => hu (he ▸ hv)) i j hi hj
    · have hqr : q ∈ rest := by simpa [he] using hq
      have hl : LocalInjective h c rest mid := fun v hv => injective v (List.mem_cons_of_mem _ hv)
      have hs : SeparateWrites h c rest mid := fun v hv w hw =>
        separate v (List.mem_cons_of_mem _ hv) w (List.mem_cons_of_mem _ hw)
      have hz := ih mid hn hl hs hqr hi
      change (reference M D h c rest mid).mem h.Out
        (offset (HostFP.launch h c q mid) c.BM c.BN (HostContract.parameters h)
          h.outStrides.batch h.outStrides.sequence (h.headStride h.outStrides.head) h.outStrides.feature i) = _
      rw [hz]
      apply congrArg (Cell.mk .fp32)
      apply congrFun
      apply expected_congr M D c.BM c.BN c.BCS (HostContract.parameters h)
        (HostFP.launch h c q mid) (HostFP.launch h c q s) rfl
      · funext a
        exact FPOutput.frame M D c.BM c.BN c.BCS (HostContract.parameters h)
          (HostFP.launch h c u s) h.A a (Or.inl inputs.1)
      · funext a
        exact FPOutput.frame M D c.BM c.BN c.BCS (HostContract.parameters h)
          (HostFP.launch h c u s) h.Dout a (Or.inl inputs.2.1)
      · funext a
        exact FPOutput.frame M D c.BM c.BN c.BCS (HostContract.parameters h)
          (HostFP.launch h c u s) h.Residual a (Or.inl inputs.2.2)

theorem grid_member (h : HostCall) (c : Tuning) (q : HostFP.Pid)
    (hq : q ∈ HostFP.grid h c) :
    (⟨q.1,q.2.1,q.2.2⟩ : HostGrid.Pid) ∈ HostGrid.programs (HostContract.grid h c) := by
  rcases q with ⟨qt,qb,qc⟩
  apply (HostGrid.mem_programs _ _).mpr
  simpa [HostFP.grid,HostContract.grid,HostGrid.dimensions,HostContract.parameters] using hq

theorem layout_injective (h : HostCall) (c : Tuning) (alloc : FlatAlloc)
    (layout : HostContract.Layout h c alloc) (s : State α) :
    LocalInjective h c (HostFP.grid h c) s := by
  intro q hq a b ha hb he
  exact layout.outputDistinct q.1 q.2.1 q.2.2 a b ha hb he

theorem layout_separate (h : HostCall) (c : Tuning) (alloc : FlatAlloc)
    (layout : HostContract.Layout h c alloc) (s : State α) :
    SeparateWrites h c (HostFP.grid h c) s := by
  intro q hq q' hq' hn i j hi hj
  have ne : (⟨q.1,q.2.1,q.2.2⟩ : HostGrid.Pid) ≠ ⟨q'.1,q'.2.1,q'.2.2⟩ := by
    intro he
    apply hn
    have hh := HostGrid.Pid.mk.inj he
    exact Prod.ext hh.1 (Prod.ext hh.2.1 hh.2.2)
  exact layout.programWrites _ (grid_member h c q hq) _ (grid_member h c q' hq') ne i j hi hj

/-- Independent observable typed output values and an exact full-grid frame. -/
def Result (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (c : Tuning)
    (initial final : State α) : Prop :=
  (∀ q ∈ HostFP.grid h c, ∀ i,
    active (HostFP.launch h c q initial) c.BM c.BN (HostContract.parameters h) i →
    final.mem h.Out (offset (HostFP.launch h c q initial) c.BM c.BN (HostContract.parameters h)
      h.outStrides.batch h.outStrides.sequence (h.headStride h.outStrides.head) h.outStrides.feature i) =
      .mk .fp32 (expected M D c.BM c.BN c.BCS (HostContract.parameters h)
        (HostFP.launch h c q initial) i)) ∧
  (∀ r a, (r ≠ h.Out ∨ ∀ q ∈ HostFP.grid h c, ∀ i,
    active (HostFP.launch h c q initial) c.BM c.BN (HostContract.parameters h) i →
    a ≠ offset (HostFP.launch h c q initial) c.BM c.BN (HostContract.parameters h)
      h.outStrides.batch h.outStrides.sequence (h.headStride h.outStrides.head) h.outStrides.feature i) →
    final.mem r a = initial.mem r a)

theorem complete (M : Algebra α) (D : DotPrimitive α) (h : HostCall) (c : Tuning)
    (hp : FPGrid.DefaultPrecision h) (alloc : FlatAlloc) (layout : HostContract.Layout h c alloc)
    (initial : State α) :
    ∃ final, HostFP.run (backend M D) h c Bool.false (HostFP.grid h c) initial = some final ∧
      Result M D h c initial final := by
  obtain ⟨final,he,hm⟩ := run_memory M D h c hp (HostFP.grid h c) initial
  have hn : (HostFP.grid h c).Nodup :=
    List.nodup_range.product (List.nodup_range.product List.nodup_range)
  refine ⟨final,he,?_,?_⟩
  · intro q hq i hi
    rw [hm]
    exact reference_readback M D h c _ initial hn (layout_injective h c alloc layout initial)
      (layout_separate h c alloc layout initial) layout.inputsSeparate q hq i hi
  · intro r a ha
    rw [hm]
    exact reference_frame M D h c _ initial r a ha

#specNonCircular reference avoiding [HostFP.run, FP.Structural.exec, FPDefault.source]
#axiomsClean run_memory
#axiomsClean reference_frame
#axiomsClean reference_readback
#axiomsClean complete
#specNonCircular Result avoiding [HostFP.run, FP.Structural.exec, FPDefault.source]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.FPGridOutput
