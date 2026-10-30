import bench.tritonbench_g.max_reduction.Proofs.Rewrite

namespace VeriTile.Bench.Repairs.MaxReduction.RawFP
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- The total reduction engine fixes a backend's value/index tree semantics.
It is part of the program type, so the two sources cannot choose different
reduction schedules. Floating values remain opaque (in particular all native
FP32 words are admitted). -/
structure Program (e : Source.Engine A) where
  kernel : ComputeKernel

def Equivalent [Inhabited A] (e : Source.Engine A) (lhs rhs : Program e) : Prop :=
  ∀ s, ∃ a b,
    FP.Structural.exec e.algebra lhs.kernel s = some a ∧
    FP.Structural.exec e.algebra rhs.kernel s = some b ∧ a.mem = b.mem

instance [Inhabited A] (e : Source.Engine A) : Spec.ProgramSyntax (Program e) where
  Statement := FP.GuardedFragment
  Signature := List RegionName × List RegionName
  signature p := (p.kernel.inputs,p.kernel.outputs)
  body p := [⟨[],p.kernel.surfaceBody,.real,none⟩]
  structural := some (fun _ _ => False)
  numerical := fun _ => Equivalent e
  sameContext := fun lhs rhs => lhs = rhs

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def stage1Program (e : Source.Engine A) (v : Bool) (inp out : RegionName) (N B : Nat) : Program e where
  kernel := if v then stage1Commuted inp out N B else stage1 inp out N B

def stage2Program (e : Source.Engine A) (v : Bool) (inp out : RegionName) (N B : Nat) : Program e where
  kernel := if v then stage2Complement inp out N B else stage2 inp out N B

def dimensionProgram (e : Source.Engine A) (v : Bool) (inp ov oi : RegionName)
    (M N K BM BN : Nat) : Program e where
  kernel := if v then dimensionCommuted inp ov oi M N K BM BN else dimension inp ov oi M N K BM BN

/-- The retained first-stage variant commutes only the integer index product.
Both complete sources succeed and agree on every typed memory cell. -/
theorem stage1_index_equiv [Inhabited A] (e : Source.Engine A)
    (inp out : RegionName) (N B : Nat) (hB : 0 < B) :
    stage1Program e Bool.false inp out N B ≡[assumptions]
      stage1Program e Bool.true inp out N B := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro s
  obtain ⟨final,he,_⟩ := Source.stage1_exec e inp out N B s hB
  exact ⟨final,final,he,(Rewrite.stage1_same_run e inp out N B s).symm.trans he,rfl⟩

/-- The retained second-stage variant complements the reversed integer bound. -/
theorem stage2_mask_equiv [Inhabited A] (e : Source.Engine A)
    (inp out : RegionName) (N B : Nat) (hB : 0 < B) :
    stage2Program e Bool.false inp out N B ≡[assumptions]
      stage2Program e Bool.true inp out N B := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro s
  obtain ⟨original,he,hm⟩ := Source.stage2_exec e inp out N B s hB
  obtain ⟨candidate,hc,cm⟩ := Rewrite.stage2_candidate_exec e inp out N B s hB
  exact ⟨original,candidate,he,hc,hm.trans cm.symm⟩

/-- Both FP32 values and signed-index stores retain the exact paired reduction
schedule. No permutation, reassociation, or special-value assumption is used. -/
theorem dimension_index_equiv [Inhabited A] (e : Source.Engine A)
    (inp ov oi : RegionName) (M N K BM BN : Nat) (hBN : 0 < BN) :
    dimensionProgram e Bool.false inp ov oi M N K BM BN ≡[assumptions]
      dimensionProgram e Bool.true inp ov oi M N K BM BN := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro s
  obtain ⟨final,he,_⟩ := Source.dimension_exec e inp ov oi M N K BM BN s hBN
  exact ⟨final,final,he,(Rewrite.dimension_same_run e inp ov oi M N K BM BN s).symm.trans he,rfl⟩

/-- Both complete first-stage sources succeed and have the same full typed
memory. The displayed single-write map also gives every exterior frame cell.
The reduction engine may use any backend tree and all FP32 words. -/
theorem stage1_success [Inhabited A] (e : Source.Engine A) (inp out : RegionName)
    (N B : Nat) (hB : 0 < B) (s : State A) :
    ∃ final,
      FP.Structural.exec e.algebra (stage1 inp out N B) s = some final ∧
      FP.Structural.exec e.algebra (stage1Commuted inp out N B) s = some final ∧
      final.mem = (s.write out (s.pids 0) (.mk .real
        (Source.scalarValue e (fun i => (s.mem inp i).read .real) N B (s.pids 0 * B)))).mem := by
  obtain ⟨final,he,hm⟩ := Source.stage1_exec e inp out N B s hB
  exact ⟨final,he,(Rewrite.stage1_same_run e inp out N B s).symm.trans he,hm⟩

theorem stage2_success [Inhabited A] (e : Source.Engine A) (inp out : RegionName)
    (N B : Nat) (hB : 0 < B) (s : State A) :
    ∃ original candidate,
      FP.Structural.exec e.algebra (stage2 inp out N B) s = some original ∧
      FP.Structural.exec e.algebra (stage2Complement inp out N B) s = some candidate ∧
      original.mem = candidate.mem ∧
      original.mem = (s.write out 0 (.mk .real
        (Source.scalarValue e (fun i => (s.mem inp i).read .real) N B 0))).mem := by
  obtain ⟨original,he,hm⟩ := Source.stage2_exec e inp out N B s hB
  obtain ⟨candidate,hc,cm⟩ := Rewrite.stage2_candidate_exec e inp out N B s hB
  exact ⟨original,candidate,he,hc,hm.trans cm.symm,hm⟩

/-- Successful full value/index executions, including the two masked stores
and every preserved memory cell. Masked padding remains part of each actual
reduction: unlike an elementwise rewrite, it cannot simply be discarded. -/
theorem dimension_success [Inhabited A] (e : Source.Engine A) (inp ov oi : RegionName)
    (M N K BM BN : Nat) (hBN : 0 < BN) (s : State A) :
    ∃ final,
      FP.Structural.exec e.algebra (dimension inp ov oi M N K BM BN) s = some final ∧
      FP.Structural.exec e.algebra (dimensionCommuted inp ov oi M N K BM BN) s = some final ∧
      final.mem = (Source.dimensionStores e inp ov oi M N K BM BN s).mem := by
  obtain ⟨final,he,hm⟩ := Source.dimension_exec e inp ov oi M N K BM BN s hBN
  exact ⟨final,he,(Rewrite.dimension_same_run e inp ov oi M N K BM BN s).symm.trans he,hm⟩

#print_fp_assumptions stage1_index_equiv
#print_fp_assumptions stage2_mask_equiv
#print_fp_assumptions dimension_index_equiv
#axiomsClean stage1_success
#axiomsClean stage2_success
#axiomsClean dimension_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaxReduction.RawFP
