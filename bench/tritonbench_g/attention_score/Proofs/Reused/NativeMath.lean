import bench.tritonbench_g.attention_score.Proofs.Reused.NativeIdeal
import bench.tritonbench_g.attention_score.Sources.AlgorithmProjection

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.NativeMath
open VeriTile Triton FP.Structural NativeInstructions NativeStaging NativeIdeal Geometry
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def inputs (decode : BitVec 32 → ℝ) (D : Nat) (p : RealSource.Parameters) (s : State Expr) : RealFormula.Inputs D where
  query := fun row d => (lower ((s.mem p.Q (FPContext.baseQ p s+row*p.queryRow+d.val*p.queryDim)).read .real)).eval (isa decode)
  key := fun col d => (lower ((s.mem p.K (FPContext.baseK p s+d.val*p.keyDim+col*p.keyRow)).read .real)).eval (isa decode)
  normalizer := fun row => (lower ((s.mem p.M (FPContext.baseM p s+row)).read .real)).eval (isa decode)

theorem query_eval (decode : BitVec 32 → ℝ) (B D : Nat) (p : RealSource.Parameters) (s : State Expr)
    (row : Nat) (i : Fin B) (d : Fin D) :
    (NativeFormula.query B D p s row (i,d,())).eval (isa decode)=
      if p.evenM || decide (row+i.val<p.queryContext) then (inputs decode D p s).query (row+i.val) d else 0 := by
  simp [NativeFormula.query,inputs,NativeFormula.zero32]

private theorem key_start (B : Nat) (s : State Expr) (keyRange : s.pids 0*B<2147483648) :
    FPContext.keyStart B s=s.pids 0*B := by
  unfold FPContext.keyStart
  have h := SourceAlgorithm.narrow_nat_exact (s.pids 0*B) keyRange
  simpa only [Int.natCast_mul] using h

theorem key_eval (decode : BitVec 32 → ℝ) (B D : Nat) (p : RealSource.Parameters) (s : State Expr)
    (i : Fin B) (d : Fin D) (keyRange : s.pids 0*B<2147483648) :
    (NativeFormula.key B D p s (d,i,())).eval (isa decode)=
      if p.evenN || decide (s.pids 0*B+i.val<p.keyContext) then (inputs decode D p s).key (s.pids 0*B+i.val) d else 0 := by
  simp [NativeFormula.key,inputs,NativeFormula.zero32,key_start B s keyRange]

theorem normalizer_eval (decode : BitVec 32 → ℝ) (B D : Nat) (p : RealSource.Parameters) (s : State Expr)
    (row : Nat) (i : Fin B) :
    (NativeFormula.normalizer B p s row i).eval (isa decode)=
      if row+i.val<p.queryContext then (inputs decode D p s).normalizer (row+i.val) else 0 := by
  simp [NativeFormula.normalizer,inputs,NativeFormula.zero32,Nat.add_assoc]

theorem raw_eval (decode : BitVec 32 → ℝ) (hz : decode 0=0) (B D : Nat) (p : RealSource.Parameters)
    (s : State Expr) (row : Nat) (i j : Fin B) (keyRange : s.pids 0*B<2147483648) :
    (NativeFormula.raw B D p s row (i,j,())).eval (isa decode)=
      pow2 (RealFormula.score (inputs decode D p s) p.scale p.queryContext p.keyContext p.evenM p.evenN
        (row+i.val) (s.pids 0*B+j.val) -
        (if row+i.val<p.queryContext then (inputs decode D p s).normalizer (row+i.val) else 0)) := by
  simp only [NativeFormula.raw,dot_eval decode hz,query_eval,key_eval decode B D p s _ _ keyRange,normalizer_eval,
    NativeFormula.scale,NativeFormula.zero32,RealFormula.score,zero_add,sub_eq_add_neg,eval_exp,eval_fma,eval_mul,eval_literal,eval_neg]
  simp only [query_eval decode B D p s row i,key_eval decode B D p s j _ keyRange,
    normalizer_eval decode B D p s row i]
  congr 1
  ring

theorem probability_eval (decode : BitVec 32 → ℝ) (hz : decode 0=0) (B D : Nat) (p : RealSource.Parameters)
    (s : State Expr) (row : Nat) (i j : Fin B) (keyRange : s.pids 0*B<2147483648) :
    (NativeFormula.probability B D p s row (i,j,())).eval (isa decode)=
      RealFormula.weight (inputs decode D p s) p.scale p.queryContext p.keyContext
        p.window p.complement p.evenM p.evenN p.offset p.size (row+i.val) (s.pids 0*B+j.val) := by
  have hd : (i.val : Int)-j.val+row-(s.pids 0 : Int)*B+p.offset=
      RealFormula.distance (row+i.val) (s.pids 0*B+j.val) p.offset := by
    unfold RealFormula.distance
    push_cast
    ring
  by_cases hm : p.evenM=Bool.true
  all_goals by_cases hw : p.window=Bool.true
  all_goals by_cases hc : p.complement=Bool.true
  all_goals by_cases hr : row+i.val<p.queryContext
  all_goals simp [NativeFormula.probability,eval_ite,raw_eval decode hz B D p s row i j keyRange,NativeFormula.zero32,
    RealFormula.weight,RealFormula.windowAllows,hd,hm,hw,hc,hr]

theorem contribution_eval (decode : BitVec 32 → ℝ) (hz : decode 0=0) (B D : Nat)
    (hB : Supported B) (p : RealSource.Parameters) (s : State Expr) (row : Nat) (j : Fin B) (keyRange : s.pids 0*B<2147483648) :
    (NativeFormula.contribution B D p s row j).eval (isa decode)=
      ∑ i : Fin B, RealFormula.weight (inputs decode D p s) p.scale p.queryContext p.keyContext
        p.window p.complement p.evenM p.evenN p.offset p.size (row+i.val) (s.pids 0*B+j.val) := by
  have hprofile : ∃ profile : Fin 3, NativeReduction.width profile=B := by
    rcases hB with rfl|rfl|rfl
    · exact ⟨⟨0,by decide⟩,rfl⟩
    · exact ⟨⟨1,by decide⟩,rfl⟩
    · exact ⟨⟨2,by decide⟩,rfl⟩
  obtain ⟨profile,hwidth⟩ := hprofile
  subst B
  rw [NativeFormula.contribution,planFor_width,reduction_eval decode hz]
  apply Finset.sum_congr rfl
  intro i _
  simp only [dif_pos i.isLt]
  exact probability_eval decode hz _ _ _ _ _ _ _ keyRange

#axiomsClean probability_eval
#axiomsClean contribution_eval
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.NativeMath
