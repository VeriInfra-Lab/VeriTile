import bench.repairs.tritonbench_g.max_reduction.Proofs.Real
import bench.repairs.tritonbench_g.max_reduction.Proofs.Address

namespace VeriTile.Bench.Repairs.MaxReduction.RealModel
noncomputable section
open Classical
set_option maxHeartbeats 2000000

def firstStage (input : Nat → ℝ) (count block : Nat) (tree : Tree) (pid : Nat) : WithBot ℝ :=
  valueReduction tree (fun lane => if pid*block+lane<count then (input (pid*block+lane) : WithBot ℝ) else ⊥)

def global (input : Nat → ℝ) (count block mid : Nat) (first : Nat → Tree) (second : Tree) : WithBot ℝ :=
  valueReduction second (fun pid => if pid<mid then firstStage input count block (first pid) pid else ⊥)

theorem global_upper (input : Nat → ℝ) (count block mid width : Nat)
    (hb : 0<block) (hm : mid=Address.count count block) (hw : mid≤width)
    (first : Nat → ReductionPlan block) (second : ReductionPlan width)
    (i : Nat) (hi : i<count) :
    (input i : WithBot ℝ)≤global input count block mid (fun p => (first p).tree) second.tree := by
  obtain ⟨hp,hl,he⟩ := Address.segment_owner count block i hb hi
  have hpl : i/block<mid := hm ▸ hp
  have hll : i%block<block := Nat.mod_lt _ hb
  have hfirst := tree_value_upper (first (i/block)).tree
    (fun lane => if i/block*block+lane<count then (input (i/block*block+lane) : WithBot ℝ) else ⊥)
    (i%block) (((first (i/block)).complete _).mpr hll)
  have he' : i/block*block+i%block=i := he
  have hsecond := tree_value_upper second.tree
    (fun pid => if pid<mid then firstStage input count block (first pid).tree pid else ⊥)
    (i/block) ((second.complete _).mpr (hpl.trans_le hw))
  simp only [hpl,if_pos] at hsecond
  simpa [global,firstStage,he',hi,hpl] using hfirst.trans hsecond

/-- Composition of every stage1 output into the actual stage2. The independent
maximum ranges over the original logical input, not over the intermediate. -/
theorem global_correct (input : Nat → ℝ) (count block mid width : Nat)
    (hn : 0<count) (hb : 0<block) (hm : mid=Address.count count block) (hw : mid≤width)
    (first : Nat → ReductionPlan block) (second : ReductionPlan width) :
    ∃ answer : ℝ,global input count block mid (fun p => (first p).tree) second.tree=(answer : WithBot ℝ) ∧
      IsMaximum input count answer := by
  have hu := global_upper input count block mid width hb hm hw first second
  have hzero := hu 0 hn
  obtain ⟨pid,hpid,hv⟩ := tree_value_attained second.tree
    (fun pid => if pid<mid then firstStage input count block (first pid).tree pid else ⊥)
  have hp : pid<mid := by
    by_contra h
    have he : global input count block mid (fun p => (first p).tree) second.tree=⊥ := by simpa [global,h] using hv
    rw [he] at hzero
    simpa using hzero
  have hg : global input count block mid (fun p => (first p).tree) second.tree=firstStage input count block (first pid).tree pid := by
    simpa [global,hp] using hv
  obtain ⟨lane,hlane,hfirst⟩ := tree_value_attained (first pid).tree
    (fun lane => if pid*block+lane<count then (input (pid*block+lane) : WithBot ℝ) else ⊥)
  have ha : pid*block+lane<count := by
    by_contra h
    have he : firstStage input count block (first pid).tree pid=⊥ := by simpa [firstStage,h] using hfirst
    rw [hg,he] at hzero
    simpa using hzero
  have result : global input count block mid (fun p => (first p).tree) second.tree=(input (pid*block+lane) : WithBot ℝ) := by
    exact hg.trans (by simpa [firstStage,ha] using hfirst)
  refine ⟨input (pid*block+lane),result,⟨pid*block+lane,ha,rfl⟩,?_⟩
  intro i hi
  have bound := hu i hi
  rw [result] at bound
  exact_mod_cast bound

/-- Each full-axis dimension program has the independent maximum and least
attaining index. Segmentation changes only m/k, never this reduction axis. -/
theorem dimension_correct (input : Nat → ℝ) (N K m k width : Nat)
    (hn : 0<N) (cover : N≤width) (plan : ReductionPlan width) :
    ∃ answer index,indexed plan.tree (fun n => input (m*N*K+n*K+k)) N=some (answer,index) ∧
      IsLeastArgmax (fun n => input (m*N*K+n*K+k)) N answer index :=
  indexed_correct _ N width hn cover plan

#axiomsClean global_correct
#axiomsClean dimension_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaxReduction.RealModel
