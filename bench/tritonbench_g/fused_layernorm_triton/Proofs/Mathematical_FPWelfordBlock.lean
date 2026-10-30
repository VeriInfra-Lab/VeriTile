import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_FPWelfordTree
namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Exact first body for XBLOCK=1 and 0<rnumel≤RBLOCK. The true
first_iteration branch returns the loaded value, positive zero and one;
it does not perform the general delta/weight update. Mask-only loads,
both casts, the unit row axis and the three post-helper masks remain. -/
def welfordPrefix (X : RegionName) (xnumel rnumel RBLOCK : Nat) : List ComputeStmt := (triton {
  tl.region X = tl.float32
  xoffset = tl.program_id(0) * $(1)
  xindex = xoffset + tl.arange(0, $(1))[:, None]
  xmask = xindex < $(xnumel)
  rbase = tl.arange(0, $(RBLOCK))[None, :]
  x0 = xindex
  tmp3_mean = tl.zeros([$(1), $(RBLOCK)], tl.float32)
  tmp3_m2 = tl.zeros([$(1), $(RBLOCK)], tl.float32)
  tmp3_weight = tl.zeros([$(1), $(RBLOCK)], tl.float32)
  roffset = $((0 : Nat))
  rindex = roffset + rbase
  rmask = rindex < $(rnumel)
  r1 = rindex
  tmp0 = tl.load(X + (r1 + ($(rnumel) * x0)), mask=rmask).to(tl.float32)
  tmp1 = (tmp0).to(tl.float32)
  tmp2, _ = tl.broadcast(tmp1, tmp3_mean)
  tmp3_weight_next = tl.full([$(1), $(RBLOCK)], $((1 : ℝ)), dtype=tl.float32)
  tmp3_mean_next = tmp2
  tmp3_m2_next = tl.zeros([$(1), $(RBLOCK)], tl.float32)
  tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
  tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
  tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
}).surfaceBody

def firstLaneValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X : RegionName) (rnumel RBLOCK : Nat) : Fin RBLOCK → WelfordTriple α := fun i =>
  if i.val<rnumel then
    ⟨M.cast (some .fp32) .real .real
      (M.cast (some .fp32) .real .real ((s.mem X (i.val+rnumel*s.pids 0)).read .real)),
      M.literal (some .fp32) .real 0,M.literal (some .fp32) .real 1⟩
  else ⟨M.literal (some .fp32) .real 0,M.literal (some .fp32) .real 0,M.literal (some .fp32) .real 0⟩

structure WelfordLanesReady {α : Type} [Inhabited α] {N : Nat} (s t : State α)
    (values : Fin N → WelfordTriple α) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  mean : t.regs .real [1,N] "tmp3_mean" = some (fun i => (values i.2.1).mean)
  m2 : t.regs .real [1,N] "tmp3_m2" = some (fun i => (values i.2.1).m2)
  weight : t.regs .real [1,N] "tmp3_weight" = some (fun i => (values i.2.1).weight)
  row : t.regs .nat [1,1] "x0" = some (fun _ => s.pids 0)

theorem welford_prefix_run {α : Type} [Inhabited α] (M : Algebra α) (equal32 : α→α→Bool)
    (s : State α) (X : RegionName) (xnumel rnumel RBLOCK : Nat) :
    ∃ final, run (withEquality M equal32) (welfordPrefix X xnumel rnumel RBLOCK) s = some final ∧
      WelfordLanesReady s final (firstLaneValues M s X rnumel RBLOCK) := by
  simp [welfordPrefix, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt, numericLt,
    withEquality, Region.cast, ofFloat, toFloat]
  constructor <;> simp [State.setReg, firstLaneValues, bop, ofFloat, toFloat, Fin.eq_zero]
  all_goals first | rfl | (funext i; first | (split <;> simp_all [Fin.eq_zero]) | simp [bop])

/-- A tuple reduction's outputs have shape [1], matching reduce(axis=1) of
[1,RBLOCK]. Each scalar expression is the exact chosen helper tree. -/
def welfordTupleCode {N : Nat} (tree : ReductionTree N) : List ComputeStmt :=
  [ .assign .real [1] "tmp3_tmp" (.compute (.alg .fp32 (.full [1] (treeExpressions tree).mean))),
    .assign .real [1] "tmp4_tmp" (.compute (.alg .fp32 (.full [1] (treeExpressions tree).m2))),
    .assign .real [1] "tmp5_tmp" (.compute (.alg .fp32 (.full [1] (treeExpressions tree).weight))) ]

structure WelfordTupleReady {α : Type} [Inhabited α] (s t : State α) (values : WelfordTriple α) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  mean : t.regs .real [1] "tmp3_tmp" = some (fun _ => values.mean)
  m2 : t.regs .real [1] "tmp4_tmp" = some (fun _ => values.m2)
  weight : t.regs .real [1] "tmp5_tmp" = some (fun _ => values.weight)
  row : t.regs .nat [1,1] "x0" = some (fun _ => s.pids 0)

theorem welford_tuple_run {α : Type} [Inhabited α] {N : Nat} (M : Algebra α)
    (equal32 : α→α→Bool) (s t : State α) (values : Fin N→WelfordTriple α)
    (h : WelfordLanesReady s t values) (tree : ReductionTree N) :
    ∃ final, run (withEquality M equal32) (welfordTupleCode tree) t = some final ∧
      WelfordTupleReady s final (treeValue M equal32 values tree) := by
  let v := treeValue M equal32 values tree
  let t1 := t.setReg "tmp3_tmp" .real [1] (fun _ => v.mean)
  let t2 := t1.setReg "tmp4_tmp" .real [1] (fun _ => v.m2)
  have ht := tree_expressions_eval M equal32 t values h.mean h.m2 h.weight tree
  have ht1 := tree_expressions_eval M equal32 t1 values
    (by simpa [t1,State.setReg] using h.mean)
    (by simpa [t1,State.setReg] using h.m2)
    (by simpa [t1,State.setReg] using h.weight) tree
  have ht2 := tree_expressions_eval M equal32 t2 values
    (by simpa [t2,t1,State.setReg] using h.mean)
    (by simpa [t2,t1,State.setReg] using h.m2)
    (by simpa [t2,t1,State.setReg] using h.weight) tree
  have hfull (u : State α) (e : Op .real []) (x : α)
      (he : evalOp (withEquality M equal32) (some .fp32) e u = some (fun _ => x)) :
      evalExpr (withEquality M equal32) (.compute (.alg .fp32 (.full [1] e))) u = some (fun _ => x) := by
    simp only [evalExpr,evalComputeOp]
    rw [evalOp_unfold]
    simp only [ComputeDType.eraseDType, bind, pure, Option.bind]
    rw [he]
  have e0 := hfull t _ _ ht.1
  have e1 := hfull t1 _ _ ht1.2.1
  have e2 := hfull t2 _ _ ht2.2.2
  simp only [ComputeDType.eraseDType, t2, t1, v] at e0 e1 e2
  refine ⟨t2.setReg "tmp5_tmp" .real [1] (fun _ => v.weight), ?_, ?_⟩
  · simp only [welfordTupleCode, run, step, ComputeDType.eraseDType, e0, e1, e2, bind, pure, Option.bind]
    rfl
  · constructor <;> simp [t2,t1,v,State.setReg,h.mem,h.pids,h.undef,h.row]

/-- Original post-tuple statements; initial dummy register declarations only
supply parser type/shape information and are excluded from the emitted list. -/
def welfordSuffix (Rstd Mean : RegionName) (rnumel : Nat) : List ComputeStmt :=
  ((triton {
    tl.region Rstd = tl.float32, Mean = tl.float32
    x0 = tl.program_id(0) + tl.arange(0, $(1))[:, None]
    tmp3_tmp = tl.zeros([$(1)], dtype=tl.float32)
    tmp4_tmp = tl.zeros([$(1)], dtype=tl.float32)
    tmp5_tmp = tl.zeros([$(1)], dtype=tl.float32)
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tl.store(Mean + x0, tmp3)
    tmp6 = $(rnumel)
    tmp7 = tmp4 / tl.toReal(tmp6)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.rsqrt(tmp9)
    tl.debug_barrier()
    tl.store(Rstd + x0, tmp10)
  } : ComputeKernel).surfaceBody).drop 4

def rstdValue {α : Type} (M : Algebra α) (rnumel : Nat) (values : WelfordTriple α) : α :=
  M.unary (some .fp32) .rsqrt (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .div values.m2 (M.fromNat (some .fp32) rnumel))
    (M.literal none .real (1e-05 : ℝ)))

def statisticsMemory {α : Type} (M : Algebra α) (s : State α) (Mean Rstd : RegionName)
    (rnumel : Nat) (values : WelfordTriple α) : State α :=
  (s.write Mean (s.pids 0) (.mk .real values.mean)).write Rstd (s.pids 0)
    (.mk .real (rstdValue M rnumel values))

theorem welford_suffix_run {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α→α→Bool) (s t : State α) (Mean Rstd : RegionName)
    (rnumel : Nat) (values : WelfordTriple α) (h : WelfordTupleReady s t values) :
    ∃ final, run (withEquality M equal32) (welfordSuffix Rstd Mean rnumel) t = some final ∧
      final.mem = (statisticsMemory M s Mean Rstd rnumel values).mem := by
  simp [welfordSuffix, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, Option.bind, ComputeDType.eraseDType, numeric, bop, store,
    withEquality, Region.cast, ofFloat, toFloat,h.mean,h.m2,h.weight,h.row,
    TileShape.allIndices,List.finRange_succ,Fin.eq_zero,State.write,statisticsMemory,rstdValue,h.mem]

/-- Correct helper expansion for the accepted single reduction block. A
complete tree is an explicit numerical execution profile, not an associative
fold quotient and not an unspecified native reduction. -/
def welfordBlock (X Mean Rstd : RegionName) (xnumel rnumel RBLOCK : Nat)
    (tree : ReductionTree RBLOCK) : ComputeKernel :=
  .mk [X] [Mean,Rstd] (welfordPrefix X xnumel rnumel RBLOCK ++ welfordTupleCode tree ++ welfordSuffix Rstd Mean rnumel)

specification welford_block_fp32 {α : Type} [Inhabited α] (M : Algebra α) (equal32 : α→α→Bool)
    (X Mean Rstd : RegionName) (xnumel rnumel RBLOCK : Nat) (tree : ReductionTree RBLOCK)
    (hpositive : 0<rnumel) (hblock : rnumel≤RBLOCK) (htree : tree.Complete) (s : State α) :
    ∃ final, FP.Structural.exec (withEquality M equal32) (welfordBlock X Mean Rstd xnumel rnumel RBLOCK tree) s = some final ∧
      final.mem = (statisticsMemory M s Mean Rstd rnumel
        (treeValue M equal32 (firstLaneValues M s X rnumel RBLOCK) tree)).mem := by
  obtain ⟨p,hp,hpr⟩ := welford_prefix_run M equal32 s X xnumel rnumel RBLOCK
  obtain ⟨u,hu,hur⟩ := welford_tuple_run M equal32 s p _ hpr tree
  obtain ⟨final,hf,hm⟩ := welford_suffix_run M equal32 s u Mean Rstd rnumel _ hur
  refine ⟨final,?_,hm⟩
  simp only [welfordBlock,FP.Structural.exec,ComputeKernel.surfaceBody,run_append,hp,hu,hf,bind,pure,Option.bind]

#axiomsClean welford_prefix_run
#axiomsClean welford_tuple_run
#axiomsClean welford_suffix_run
#axiomsClean welford_block_fp32
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.FPStages
