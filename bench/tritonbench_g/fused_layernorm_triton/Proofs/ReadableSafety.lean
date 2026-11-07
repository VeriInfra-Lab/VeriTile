import bench.tritonbench_g.fused_layernorm_triton.Proofs.ReadableWhole
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.ReadableSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem list_safe (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt)
    (each : ∀ st∈code,∀ s : State α,Safety.stepSafe M bounds st s) (s : State α) :
    Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => simp [Safety.traceSafeList]
  | cons head tail ih =>
    apply Safety.traceSafeList_cons_intro (each head (by simp) s)
    intro t _
    exact ih (fun st hst => each st (by simp [hst])) t

theorem tree_safe (M : Algebra α) (bounds : RegionName → Nat)
    (s : State α) {B : Nat} (tree : ReductionTree B) :
    Safety.safeOp M (some .fp32) bounds s (FPStages.treeExpressions tree).mean ∧
    Safety.safeOp M (some .fp32) bounds s (FPStages.treeExpressions tree).m2 ∧
    Safety.safeOp M (some .fp32) bounds s (FPStages.treeExpressions tree).weight := by
  induction tree with
  | leaf i => simp [FPStages.treeExpressions,Safety.safeOp]
  | node l r hl hr =>
    simp [FPStages.treeExpressions,FPStages.combineExpressions,Safety.safeOp,
      hl.1,hl.2.1,hl.2.2,hr.1,hr.2.1,hr.2.2]

theorem tuple_safe (M : Algebra α) (bounds : RegionName → Nat)
    (s : State α) {B : Nat} (tree : ReductionTree B) :
    Safety.traceSafeList M bounds (welfordTupleCode tree) s := by
  apply list_safe
  intro st member t
  simp only [welfordTupleCode,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl | rfl | rfl
  · simpa [Safety.stepSafe_unfold,Safety.safeExpr,Safety.safeCompute,Safety.safeOp]
      using (tree_safe M bounds t tree).1
  · simpa [Safety.stepSafe_unfold,Safety.safeExpr,Safety.safeCompute,Safety.safeOp]
      using (tree_safe M bounds t tree).2.1
  · simpa [Safety.stepSafe_unfold,Safety.safeExpr,Safety.safeCompute,Safety.safeOp]
      using (tree_safe M bounds t tree).2.2

theorem suffix_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s t : State α) (Mean Rstd : RegionName) (D B : Nat)
    (v : WelfordTriple α) (h : WelfordTupleReady B s t v)
    (bounds : RegionName → Nat) (bm : s.pids 0<bounds Mean) (br : s.pids 0<bounds Rstd) :
    Safety.traceSafeList (withEquality M eq32) bounds (welfordSuffix Rstd Mean D) t := by
  simp only [welfordSuffix,SourcePieces.normalization,SourcePieces.statistics,SourcePieces.source,
    Kernels.kernel,Pasted.originalKernel,ComputeKernel.surfaceBody,Source.intRange?,List.drop,
    Option.getD,Option.map,Bool.false_eq_true,ite_false,String.reduceEq,beq_self_eq_true,↓reduceIte]
  simp [
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,bop,store,
    withEquality,Region.cast,ofFloat,toFloat,h.mean,h.m2,h.weight,h.row,
    TileShape.allIndices,List.finRange_succ,Fin.eq_zero,bm,br]


private theorem append_safe (M : Algebra α) (bounds : RegionName → Nat)
    (a b : List ComputeStmt) (s t : State α)
    (ha : Safety.traceSafeList M bounds a s) (ran : run M a s=some t)
    (hb : Safety.traceSafeList M bounds b t) :
    Safety.traceSafeList M bounds (a++b) s := by
  apply (Safety.traceSafeList_append M bounds a b s).mpr
  refine ⟨ha,?_⟩
  intro u hu
  have eq : t=u := Option.some.inj (ran.symm.trans hu)
  simpa only [←eq] using hb

theorem prefix_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) (bounds : RegionName → Nat) :
    Safety.traceSafeList (withEquality M eq32) bounds
      (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).before s := by
  simp only [SourcePieces.statistics,SourcePieces.source,Bool.false_eq_true,ite_false,
    Kernels.kernel,Pasted.originalKernel,ComputeKernel.surfaceBody,Source.intRange?,Option.map,
    Option.getD,String.reduceEq,beq_self_eq_true,↓reduceIte]
  simp [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,numericEq,
    withEquality,Region.cast,ofFloat,toFloat]

theorem chunk_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s t : State α) (X W Bias Y Mean Rstd : RegionName) (S D B k : Nat)
    (tree : Kernels.ReductionTree B)
    (previous : Fin B → WelfordTriple α) (h : WelfordLanesReady s t previous)
    (bounds : RegionName → Nat) (hx : ∀ j,j<D → j+D*s.pids 0<bounds X) :
    Safety.traceSafeList (withEquality M eq32) bounds
      (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).body
      (t.setReg "roffset" .int [] (fun _ => (k*B : Nat))) := by
  simp only [SourcePieces.statistics,SourcePieces.source,Bool.false_eq_true,ite_false,
    Kernels.kernel,Pasted.originalKernel,ComputeKernel.surfaceBody,Source.intRange?,Option.map,
    Option.getD,String.reduceEq,beq_self_eq_true,↓reduceIte]
  simp [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,numericEq,
    withEquality,Region.cast,ofFloat,toFloat,h.mean,h.m2,h.weight,h.row,h.mem,h.indices.base,State.setReg,
    source_column_lt,source_address]
  constructor
  · exact fun j => hx (k*B+j.val)
  · by_cases first : k=0 ∨ B=0 <;>
      simp [first,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr,
        Safety.safeCompute,Safety.safeOp,State.setReg,h.mean,h.m2,h.weight]

theorem statistics_loop_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s p : State α) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) (hB : 0<B)
    (h : WelfordLanesReady s p (ReadableStatistics.zeroLanes M B))
    (bounds : RegionName → Nat) (hx : ∀ j,j<D → j+D*s.pids 0<bounds X) :
    Safety.rangeIntSafe (withEquality M eq32) bounds "roffset" 0 D B
      (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).body p := by
  let P (q : Int) (u : State α) := ∃ (k : Nat) (values : Fin B → WelfordTriple α),
    q=(k*B : Nat) ∧ WelfordLanesReady s u values ∧
      (k=0 → values=ReadableStatistics.zeroLanes M B)
  apply Safety.rangeIntSafe_of_invariant _ _ _ _ _ _ P
  · intro q hq u initial
    rcases initial with ⟨k,values,eq,ready,hzero⟩
    subst q
    exact chunk_safe M eq32 s u X W Bias Y Mean Rstd S D B k tree values ready bounds hx
  · intro q hq u initial v ran
    rcases initial with ⟨k,values,eq,ready,hzero⟩
    subst q
    have effect : ∃ w,run (withEquality M eq32)
        (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).body
        (u.setReg "roffset" .int [] (fun _ => (k*B : Nat)))=some w ∧
        ∃ values : Fin B → WelfordTriple α,WelfordLanesReady s w values := by
      by_cases hk : k=0
      · subst k
        rw [hzero rfl] at ready
        obtain ⟨w,hw,hr⟩ := ReadableStatistics.first_run M eq32 s u X W Bias Y Mean Rstd S D B tree ready
        exact ⟨w,by simpa using hw,_,hr⟩
      · obtain ⟨w,hw,hr⟩ := ReadableStatistics.chunk_run M eq32 s u X W Bias Y Mean Rstd S D B k tree hB (by omega) values ready
        exact ⟨w,hw,_,hr⟩
    obtain ⟨w,hw,values,ready⟩ := effect
    have same : w=v := Option.some.inj (hw.symm.trans ran)
    subst v
    exact ⟨k+1,values,by simp [Nat.add_mul,Nat.cast_add],ready,by simp⟩
  · exact ⟨0,_,by simp,h,by simp⟩

private theorem singleton_safe (M : Algebra α) (bounds : RegionName → Nat)
    (st : ComputeStmt) (s : State α) (safe : Safety.stepSafe M bounds st s) :
    Safety.traceSafeList M bounds [st] s := by
  apply Safety.traceSafeList_cons_intro safe
  intro t _
  simp [Safety.traceSafeList]

theorem statistics_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) (hB : 0<B)
    (bounds : RegionName → Nat) (hx : ∀ j,j<D → j+D*s.pids 0<bounds X)
    (bm : s.pids 0<bounds Mean) (br : s.pids 0<bounds Rstd) :
    Safety.traceSafeList (withEquality M eq32) bounds
      (ReadableWhole.statisticsCode Bool.false X W Bias Y Mean Rstd S D B tree) s := by
  obtain ⟨p,hp,hpr⟩ := ReadableStatistics.prefix_run M eq32 s X W Bias Y Mean Rstd S D B tree
  obtain ⟨q,hq,hqr⟩ := ReadableStatistics.loop_run M eq32 s p X W Bias Y Mean Rstd S D B tree hB hpr
  change rangeInt (withEquality M eq32) "roffset" 0 (Int.ofNat D) B
    (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).body p=some q at hq
  obtain ⟨u,hu,hur⟩ := welford_tuple_run M eq32 s q _ hqr (ReadableStatistics.legacyTree tree)
  let loop := ComputeStmt.forRangeInt "roffset"
    (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).start
    (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).stop
    (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).stride
    (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).body
  have hstart : (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).start = .ref .int [] "zero" := rfl
  have hstop : (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).stop = .castNatToInt (.constNat D) := rfl
  have hstride : (SourcePieces.statistics Bool.false X W Bias Y Mean Rstd S D B tree).stride = .constNat B := rfl
  have loopSafe : Safety.traceSafeList (withEquality M eq32) bounds [loop] p := by
    apply singleton_safe
    simp only [loop,Safety.stepSafe_unfold,hstart,hstop,hstride,Safety.safeOp,evalOp_unfold,
      hpr.indices.zero,Option.bind,bind,pure,true_and,and_true]
    exact statistics_loop_safe M eq32 s p X W Bias Y Mean Rstd S D B tree hB hpr bounds hx
  have loopRun : run (withEquality M eq32) [loop] p=some q := by
    simp only [loop,run,step,hstart,hstop,hstride,evalOp_unfold,hpr.indices.zero,hq,Option.bind,bind,pure]
  have tupleSafe : Safety.traceSafeList (withEquality M eq32) bounds
      [.ifThen (.constBool Bool.true) (welfordTupleCode (ReadableStatistics.legacyTree tree))] q := by
    apply singleton_safe
    simpa [Safety.stepSafe,Safety.safeOp,evalOp_unfold]
      using tuple_safe (withEquality M eq32) bounds q (ReadableStatistics.legacyTree tree)
  have tupleRun : run (withEquality M eq32)
      [.ifThen (.constBool Bool.true) (welfordTupleCode (ReadableStatistics.legacyTree tree))] q=some u := by
    simp [run,step,evalOp_unfold,hu]
  unfold ReadableWhole.statisticsCode
  rw [ReadableStatistics.middle_eq]
  simp only [List.append_assoc]
  exact append_safe _ _ _ _ _ _ (prefix_safe M eq32 s X W Bias Y Mean Rstd S D B tree bounds) hp
    (append_safe _ _ _ _ _ _ loopSafe loopRun
      (append_safe _ _ _ _ _ _ tupleSafe tupleRun (suffix_safe M eq32 s u Mean Rstd D B _ hur bounds bm br)))

theorem projections_safe (M : Algebra α) (bounds : RegionName → Nat)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) :
    Safety.traceSafeList M bounds (SourcePieces.projections X W Bias Y Mean Rstd S D B tree) s := by
  rw [←SourcePieces.projectionBody_eq X W Bias Y Mean Rstd S D B tree]
  apply list_safe
  intro st member t
  simp only [SourcePieces.projectionBody,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl | rfl | rfl <;>
    simp [Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,Safety.safeOp,Safety.safeOp_unfold,ComputeDType.eraseDType]

theorem statistics_safe_for (hoist : Bool) (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) (hB : 0<B)
    (bounds : RegionName → Nat) (hx : ∀ j,j<D → j+D*s.pids 0<bounds X)
    (bm : s.pids 0<bounds Mean) (br : s.pids 0<bounds Rstd) :
    Safety.traceSafeList (withEquality M eq32) bounds
      (ReadableWhole.statisticsCode hoist X W Bias Y Mean Rstd S D B tree) s := by
  cases hoist
  · exact statistics_safe M eq32 s X W Bias Y Mean Rstd S D B tree hB bounds hx bm br
  · obtain ⟨u,hu,hs⟩ := ReadableStatistics.statistics_ready M eq32 s X W Bias Y Mean Rstd S D B tree hB
    have heq : ReadableWhole.statisticsCode Bool.true X W Bias Y Mean Rstd S D B tree =
        ReadableWhole.statisticsCode Bool.false X W Bias Y Mean Rstd S D B tree ++
          SourcePieces.projections X W Bias Y Mean Rstd S D B tree := by rfl
    rw [heq]
    exact append_safe _ _ _ _ _ _
      (statistics_safe M eq32 s X W Bias Y Mean Rstd S D B tree hB bounds hx bm br) hu
      (projections_safe _ bounds u X W Bias Y Mean Rstd S D B tree)

theorem normalize_safe (hoist : Bool) (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B k row : Nat)
    (tree : Kernels.ReductionTree B) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3"=some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10"=some (fun _ => inverse))
    (hx : s.regs .int [1,1] "x0"=some (fun _ => (row : Int)))
    (indexPins : SourceIndexPins B s)
    (pins : hoist=Bool.true → ReadableNormalize.Pins s row mean inverse)
    (bounds : RegionName → Nat)
    (bx : ∀ j,j<D → j+D*row<bounds X) (bw : ∀ j,j<D → j<bounds W)
    (bb : ∀ j,j<D → j<bounds Bias) (byOut : ∀ j,j<D → j+D*row<bounds Y) :
    Safety.traceSafeList (withEquality M eq32) bounds
      (SourcePieces.normalization hoist X W Bias Y Mean Rstd S D B tree).body
      (s.setReg "roffset" .int [] (fun _ => (k*B : Nat))) := by
  rw [←SourcePieces.normalizationBody_eq hoist X W Bias Y Mean Rstd S D B tree]
  cases hoist
  · simp only [SourcePieces.normalizationBody]
    simp [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,numericEq,
    withEquality,Region.cast,ofFloat,toFloat,hm,hr,hx,indexPins.base,State.setReg,store,
      source_column_lt,source_address,source_column_address]
    exact ⟨fun j => bx (k*B+j.val),fun j => bw (k*B+j.val),
      fun j => bb (k*B+j.val),fun j => byOut (k*B+j.val)⟩
  · have hp := pins rfl
    simp only [SourcePieces.normalizationBody]
    simp [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,numericEq,
    withEquality,Region.cast,ofFloat,toFloat,hm,hr,hx,indexPins.base,State.setReg,hp.row,hp.mean,hp.inverse,store,
      source_column_lt,source_address,source_column_address]
    exact ⟨fun j => bx (k*B+j.val),fun j => bw (k*B+j.val),
      fun j => bb (k*B+j.val),fun j => byOut (k*B+j.val)⟩

theorem normalize_loop_safe (hoist : Bool) (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B row : Nat)
    (tree : Kernels.ReductionTree B) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3"=some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10"=some (fun _ => inverse))
    (hx : s.regs .int [1,1] "x0"=some (fun _ => (row : Int)))
    (indexPins : SourceIndexPins B s)
    (pins : hoist=Bool.true → ReadableNormalize.Pins s row mean inverse)
    (bounds : RegionName → Nat)
    (bx : ∀ j,j<D → j+D*row<bounds X) (bw : ∀ j,j<D → j<bounds W)
    (bb : ∀ j,j<D → j<bounds Bias) (byOut : ∀ j,j<D → j+D*row<bounds Y) :
    Safety.rangeIntSafe (withEquality M eq32) bounds "roffset" 0 D B
      (SourcePieces.normalization hoist X W Bias Y Mean Rstd S D B tree).body s := by
  let P (q : Int) (u : State α) := ∃ k,q=(k*B : Nat) ∧ NormalizeRegisters s u row mean inverse ∧
    SourceIndexPins B u ∧ (hoist=Bool.true → ReadableNormalize.Pins u row mean inverse)
  apply Safety.rangeIntSafe_of_invariant _ _ _ _ _ _ P
  · intro q hq u initial
    rcases initial with ⟨k,eq,ready,hindex,hpin⟩
    subst q
    exact normalize_safe hoist M eq32 u X W Bias Y Mean Rstd S D B k row tree mean inverse
      ready.mean ready.rstd ready.row hindex hpin bounds bx bw bb byOut
  · intro q hq u initial v ran
    rcases initial with ⟨k,eq,ready,hindex,hpin⟩
    subst q
    obtain ⟨w,hw,he,hindices,hpins⟩ := ReadableNormalizeLoop.chunk_effect hoist M eq32 u
      X W Bias Y Mean Rstd S D B k row tree mean inverse ready.mean ready.rstd ready.row hindex hpin
    have heq : w=v := Option.some.inj (hw.symm.trans ran)
    subst v
    exact ⟨k+1,by simp [Nat.add_mul,Nat.cast_add],⟨he.registers.pids.trans ready.pids,he.registers.undef.trans ready.undef,
      he.registers.mean,he.registers.rstd,he.registers.row⟩,hindices,hpins⟩
  · exact ⟨0,by simp,⟨rfl,rfl,hm,hr,hx⟩,indexPins,pins⟩

theorem source_safe_for (hoist : Bool) (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) (hB : 0<B) (bounds : RegionName → Nat)
    (bx : ∀ j,j<D → j+D*s.pids 0<bounds X) (bw : ∀ j,j<D → j<bounds W)
    (bb : ∀ j,j<D → j<bounds Bias) (byOut : ∀ j,j<D → j+D*s.pids 0<bounds Y)
    (bm : s.pids 0<bounds Mean) (br : s.pids 0<bounds Rstd) :
    TraceSafe (withEquality M eq32) bounds (SourcePieces.source hoist X W Bias Y Mean Rstd S D B tree) s := by
  obtain ⟨u,hu,hs,hpin⟩ := ReadableWhole.statistics_ready hoist M eq32 s X W Bias Y Mean Rstd S D B tree hB
  unfold TraceSafe
  rw [SourcePieces.complete]
  change Safety.traceSafeList _ _ (ReadableWhole.statisticsCode hoist X W Bias Y Mean Rstd S D B tree ++ _) s
  apply append_safe _ _ _ _ _ _
    (statistics_safe_for hoist M eq32 s X W Bias Y Mean Rstd S D B tree hB bounds bx bm br) hu
  apply singleton_safe
  have hstart : (SourcePieces.normalization hoist X W Bias Y Mean Rstd S D B tree).start = .ref .int [] "zero" := by cases hoist <;> rfl
  have hstop : (SourcePieces.normalization hoist X W Bias Y Mean Rstd S D B tree).stop = .castNatToInt (.constNat D) := by cases hoist <;> rfl
  have hstride : (SourcePieces.normalization hoist X W Bias Y Mean Rstd S D B tree).stride = .constNat B := by cases hoist <;> rfl
  have safeLoop := normalize_loop_safe hoist M eq32 u X W Bias Y Mean Rstd S D B (s.pids 0) tree _ _
      hs.mean hs.rstd hs.row hs.indices hpin bounds bx bw bb byOut
  change Safety.rangeIntSafe (withEquality M eq32) bounds "roffset" 0 (Int.ofNat D) B
    (SourcePieces.normalization hoist X W Bias Y Mean Rstd S D B tree).body u at safeLoop
  simpa only [Safety.stepSafe_unfold,hstart,hstop,hstride,Safety.safeOp,evalOp_unfold,
    hs.indices.zero,Option.bind,bind,pure,true_and,and_true] using safeLoop

#axiomsClean source_safe_for

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.ReadableSafety
