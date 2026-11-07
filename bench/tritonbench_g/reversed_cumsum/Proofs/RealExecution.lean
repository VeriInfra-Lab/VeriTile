import bench.tritonbench_g.reversed_cumsum.Proofs.RealBridge
import bench.tritonbench_g.reversed_cumsum.Proofs.Mathematical_Formula

namespace VeriTile.Bench.Repairs.ReversedCumsum.RealExecution
open VeriTile.Triton RealProgram RealBridge IndexProgram
open VeriTile.Bench.TritonBenchG.ReversedCumsumScalar
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def sourceInitial (h : Call) (c : Config) (u : BlockState) : BlockState :=
  let u1 := u.setReg "i_s" .int [] (Tile.scalar (u.pids 0:Int))
  let u2 := u1.setReg "i_bh" .int [] (Tile.scalar (u.pids 1:Int))
  let u3 := u2.setReg "o_i" .nat [c.BT] ⟨fun i => i.1.val⟩
  let u4 := u3.setReg "m_s" .real [c.BT,c.BT]
    (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath.upperTriTile c.BT)
  u4.setReg "b_z" .real [BS] (Tile.vec fun _ => some 0)

theorem init_executes (h : Call) (c : Config) (u : BlockState) :
    stepStmts (initCode h c) u = some (sourceInitial h c u) := by
  unfold initCode
  rw [ProgramRewrite.constant_branch]
  simp only [if_true,List.cons_append,List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.castNatToInt (.programId 0)) u = some (Tile.scalar (u.pids 0:Int)) from by simp [evalOp]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.castNatToInt (.programId 1)) _ = some (Tile.scalar (u.pids 1:Int)) from by simp [evalOp]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.arange c.BT) _ = some (⟨fun i : TileIndex [c.BT] => i.1.val⟩ : Tile .nat [c.BT]) from by rw [evalOp]; rfl))]
  let u3 := ((u.setReg "i_s" .int [] (Tile.scalar (u.pids 0:Int))).setReg
    "i_bh" .int [] (Tile.scalar (u.pids 1:Int))).setReg "o_i" .nat [c.BT] ⟨fun i => i.1.val⟩
  have tri := VeriTile.Bench.TritonBenchG.ReversedCumsum.RealInit.triangular_eval
    (layout h c) Bool.false u3 (by simp [u3,layout])
  change evalOp _ u3 = some (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath.upperTriTile c.BT) at tri
  erw [stepStmts.cons_some (stepStmt_assign_eq_some tri)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.full [BS] (.const 0)) _ = some (Tile.vec fun _ => some (0:ℝ)) from by
      simp [evalOp_full,evalOp_const]; rfl))]
  rw [stepStmts.nil]
  rfl

theorem initial_invariant (h : Call) (c : Config) (row : Nat) (f : Geometry.FeatureBlock h)
    (u : BlockState) (positive : 0<c.BT) (pid0 : u.pids 0=f.val) (pid1 : u.pids 1=row) :
    Holds u h c row f.val (u.readMem h.input) (Geometry.chunks h c) (sourceInitial h c u) := by
  have inv := VeriTile.Bench.TritonBenchG.ReversedCumsum.RealInit.invariant (layout h c)
    (zeroUndef u) h.input h.output (u.readMem h.input) (fun _ => rfl) positive
  have launches : VeriTile.Bench.TritonBenchG.ReversedCumsum.RealInit.launch (zeroUndef u)=launch row f.val := by
    simp [VeriTile.Bench.TritonBenchG.ReversedCumsum.RealInit.launch,zeroUndef,launch,pid0,pid1]
  rw [launches] at inv
  apply transfer u h c row f.val (u.readMem h.input) (Geometry.chunks h c) _ _ inv
  · rfl
  all_goals simp [sourceInitial,VeriTile.Bench.TritonBenchG.ReversedCumsum.RealInit.initial,
    zeroUndef,layout,pid0,pid1]

theorem mirror_invariant (initial : BlockState) (h : Call) (c : Config) (row f : Nat)
    (a : Nat → ℝ) (q : Nat) (u : BlockState) (inv : Holds initial h c row f a q u) (j : Int) :
    Holds initial h c row f a q (u.setReg "__rev_i_t" .int [] (Tile.scalar j)) := by
  refine ⟨rfl,inv.count_le,?_,?_,?_,?_,?_,inv.output,inv.frame⟩
  · simpa using inv.feature
  · simpa using inv.head
  · simpa using inv.triangular
  · simpa using inv.accumulated
  · simpa [zeroUndef] using inv.input

theorem iteration (initial : BlockState) (h : Call) (c : Config) (row : Nat)
    (f : Geometry.FeatureBlock h) (a : Nat → ℝ) (distinct : h.input≠h.output)
    (positive : 0<c.BT) (j : Int) (u : BlockState) (hj : j<1)
    (inv : Holds initial h c row f.val a (SignedIndex.remaining j) u) :
    ∃ v, stepStmts (loopBody h c) (u.setReg "__rev_i_t" .int [] (Tile.scalar j))=some v ∧
      Holds initial h c row f.val a (SignedIndex.remaining (j+1)) v := by
  have count : (-j).toNat<Geometry.chunks h c := by
    have hn := inv.count_le
    rw [SignedIndex.current_chunk j hj] at hn
    change (-j).toNat+1≤Geometry.chunks h c at hn
    omega
  have hm : Holds initial h c row f.val a ((-j).toNat+1)
      (u.setReg "__rev_i_t" .int [] (Tile.scalar j)) := by
    rw [←SignedIndex.current_chunk j hj]
    exact mirror_invariant initial h c row f.val a (SignedIndex.remaining j) u inv j
  obtain ⟨v,he,hv⟩ := RealBridge.iteration initial h c row f a distinct positive (-j).toNat
    (u.setReg "__rev_i_t" .int [] (Tile.scalar j)) count hm
  refine ⟨v,?_,?_⟩
  · change stepStmts (.assign .int [] "i_t" indexOp :: RealStep.chunkCode h c)
      (u.setReg "__rev_i_t" .int [] (Tile.scalar j))=some v
    unfold indexOp
    rw [stepStmts.cons_some (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealExecution.mirror_assign j u)]
    simpa only [SignedIndex.int_index_of_nat j hj] using he
  · simpa only [SignedIndex.after_step] using hv

theorem start_eval (h : Call) (c : Config) (u : BlockState) :
    evalOp (reverseStart h c) u=some (Tile.scalar (1-(Geometry.chunks h c:Int))) :=
  VeriTile.Bench.TritonBenchG.ReversedCumsum.RealExecution.start_eval (layout h c) u

theorem statement_eval (h : Call) (c : Config) (u : BlockState) :
    stepStmt (statement h c) u=stepForRangeIntAux "__rev_i_t" (1-(Geometry.chunks h c:Int)) 1 1 (loopBody h c) u := by
  rw [statement,stepForRangeIntAux.forRangeInt_unfold,start_eval]
  simp only [evalOp,Option.bind_some,Option.bind_eq_bind,Tile.scalar]

theorem loop_executes (initial : BlockState) (h : Call) (c : Config) (row : Nat)
    (f : Geometry.FeatureBlock h) (a : Nat → ℝ) (distinct : h.input≠h.output)
    (positive : 0<c.BT) (u : BlockState) (inv : Holds initial h c row f.val a (Geometry.chunks h c) u) :
    ∃ v, stepStmt (statement h c) u=some v ∧ Holds initial h c row f.val a 0 v := by
  obtain ⟨final,v,he,hf,hv⟩ := VeriTile.Bench.TritonBenchG.ReversedCumsum.SignedRange.executes "__rev_i_t" 1 1
    (loopBody h c) (fun j v => Holds initial h c row f.val a (SignedIndex.remaining j) v)
    (by decide) (fun j v hj hv => iteration initial h c row f a distinct positive j v hj hv)
    (1-(Geometry.chunks h c:Int)) u (by simpa only [SignedIndex.at_start] using inv)
  exact ⟨v,(statement_eval h c u).trans he,by simpa only [SignedIndex.at_or_after_end final hf] using hv⟩

def footprint (h : Call) (row f : Nat) : WriteFootprint :=
  VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop.footprint (layout h ⟨1,1⟩) (launch row f) h.output

/-- Independent suffix specification, over the logical input before execution. -/
def suffix (h : Call) (a : Nat → ℝ) (row t s : Nat) : ℝ :=
  VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath.suffixValue
    (fun k => a (Geometry.address h row k s)) h.T t

theorem expected_suffix (h : Call) (c : Config) (row f : Nat) (a : Nat → ℝ)
    (positive : 0<c.BT) (t : Nat) (j : Fin BS) (ht : t<h.T) (hs : f*BS+j.val<h.S) :
    VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop.expected (layout h c) (launch row f) a (fun _ => 0) t j=
      suffix h a row t (f*BS+j.val) := by
  unfold VeriTile.Bench.TritonBenchG.ReversedCumsum.RealLoop.expected suffix
  have val : VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.value (layout h c) (launch row f) a (fun _ => 0) =
      fun k j => if k<h.T ∧ f*BS+j.val<h.S then a (Geometry.address h row k (f*BS+j.val)) else 0 := by
    funext k j
    simp [VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.value,
      VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.active,
      VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.address,layout,launch,Geometry.address]
  simp only [val,hs,and_true]
  exact VeriTile.Bench.TritonBenchG.ReversedCumsum.Formula.zero_padding _ (fun _ => 0) h.T _ t
    (VeriTile.Bench.TritonBenchG.ReversedCumsum.RealInit.chunks_cover (layout h c) positive)
    (Nat.le_of_lt ht) (by intros; rfl)

theorem source_correct (h : Call) (c : Config) (row : Nat) (f : Geometry.FeatureBlock h)
    (initial : BlockState) (distinct : h.input≠h.output) (positive : 0<c.BT)
    (pid0 : initial.pids 0=f.val) (pid1 : initial.pids 1=row) :
    ∃ final, exec (Kernels.sourceKernel h c).eraseDType.toAlgKernel initial=some final ∧
      (∀ (t : Nat) (j : Fin BS), t<h.T → f.val*BS+j.val<h.S →
        final.mem h.output (Geometry.address h row t (f.val*BS+j.val))=
          MemCell.real (suffix h (initial.readMem h.input) row t (f.val*BS+j.val))) ∧
      BlockState.WriteWithin (footprint h row f.val) initial final := by
  have inv := initial_invariant h c row f initial positive pid0 pid1
  obtain ⟨final,he,hv⟩ := loop_executes initial h c row f (initial.readMem h.input) distinct positive _ inv
  refine ⟨final,?_,?_,hv.frame⟩
  · rw [exec,source_body,body,ProgramRewrite.append]
    rw [init_executes]
    change stepStmts [statement h c] _=some final
    rw [stepStmts.cons_some he,stepStmts.nil]
  · intro t j ht hs
    have read := hv.output t j ⟨ht,hs⟩ (by simp)
    rw [expected_suffix h c row f.val (initial.readMem h.input) positive t j ht hs] at read
    simpa [zeroUndef,VeriTile.Bench.TritonBenchG.ReversedCumsum.RealStep.address,layout,launch,Geometry.address] using read

#print axioms source_correct
end
end VeriTile.Bench.Repairs.ReversedCumsum.RealExecution
