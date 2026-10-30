import bench.tritonbench_g.parallel_attention.NativeGrid

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000

/-- The parent backward kernel calls DQ, a debug barrier, then DKV, with
one fixed program coordinate. Both helper quotations include their stores. -/
def combinedKernel (shape : Shape) (σ : ℝ) (tile : Tile) : ComputeKernel :=
  ComputeKernel.seq ["q","k","v","do","dz"] ["dq","dk","dv"]
    [queryKernel (dqConfig shape σ tile),
      .mk [] [] [.effectMarker "tl.debug_barrier"],
      backwardKernel (backwardConfig shape σ tile)]

theorem combined_exec {α : Type} [Inhabited α] (M : Algebra α)
    (shape : Shape) (σ : ℝ) (tile : Tile) (s : State α) :
    exec M (combinedKernel shape σ tile) s=
      (exec M (queryKernel (dqConfig shape σ tile)) s).bind
        (exec M (backwardKernel (backwardConfig shape σ tile))) := by
  rw [combinedKernel,exec_seq_cons]
  congr 1
  funext middle
  rw [exec_seq_cons]
  have barrier : FP.Structural.exec M (.mk [] [] [.effectMarker "tl.debug_barrier"]) middle=some middle := by
    simp only [FP.Structural.exec,FP.Structural.run,FP.Structural.step,Option.bind_some]
    rfl
  rw [barrier,Option.bind_some,exec_seq_cons]
  simp [exec_seq_nil]

def backwardWrites (shape : Shape) (p : Program shape 64) (R : RegionName) (off : Nat) : Prop :=
  (queryRectangle shape p.tile).writes 64 shape.BK R off ∨ keyValueWrites shape p R off

theorem backward_write_unique (shape : Shape) (p q : Program shape 64) (R : RegionName) (off : Nat)
    (hp : backwardWrites shape p R off) (hq : backwardWrites shape q R off) : p=q := by
  rcases hp with hp|hp <;> rcases hq with hq|hq
  · exact query_write_unique shape p q R off hp hq
  · rcases hq with hq|hq
    · exact False.elim ((by decide : ("dq" : RegionName)≠"dk") (hp.1.symm.trans hq.1))
    · exact False.elim ((by decide : ("dq" : RegionName)≠"dv") (hp.1.symm.trans hq.1))
  · rcases hp with hp|hp
    · exact False.elim ((by decide : ("dk" : RegionName)≠"dq") (hp.1.symm.trans hq.1))
    · exact False.elim ((by decide : ("dv" : RegionName)≠"dq") (hp.1.symm.trans hq.1))
  · exact keyValue_write_unique shape p q R off hp hq

/-- Local sequential composition. The second helper's target depends only
on preserved inputs; the first helper's stores are outside its footprint. -/
theorem sequential_step
    (first second : State α → Option (State α))
    (wq wk : RegionName → Nat → Prop)
    (tq tk : State α → RegionName → Nat → Cell α)
    (queryRegion : ∀ R off,wq R off → R="dq")
    (otherRegion : ∀ R off,wk R off → R≠"dq")
    (separate : ∀ (R : RegionName) off,R∈(["q","k","v","do","dz"] : List RegionName) → ¬wq R off)
    (stable : ∀ s t,SameInputs s t → ∀ R off,tk s R off=tk t R off)
    (firstStep : ∀ s,∃ final,first s=some final ∧
      (∀ R off,wq R off → final.mem R off=tq s R off) ∧
      (∀ R off,¬wq R off → final.mem R off=s.mem R off))
    (secondStep : ∀ s,∃ final,second s=some final ∧
      (∀ R off,wk R off → final.mem R off=tk s R off) ∧
      (∀ R off,¬wk R off → final.mem R off=s.mem R off)) (s : State α) :
    ∃ final,(first s).bind second=some final ∧
      (∀ R off,wq R off ∨ wk R off → final.mem R off=
        if R="dq" then tq s R off else tk s R off) ∧
      (∀ R off,¬(wq R off ∨ wk R off) → final.mem R off=s.mem R off) := by
  obtain ⟨middle,hfirst,rfirst,ffirst⟩ := firstStep s
  have inputs : SameInputs s middle := by
    intro R hR; funext off
    exact (ffirst R off (separate R off hR)).symm
  obtain ⟨final,hsecond,rsecond,fsecond⟩ := secondStep middle
  refine ⟨final,by rw [hfirst,Option.bind_some,hsecond],?_,?_⟩
  · intro R off writes
    rcases writes with hq|hk
    · have reg := queryRegion R off hq
      rw [if_pos reg,fsecond R off (fun h => otherRegion R off h reg),rfirst R off hq]
    · rw [if_neg (otherRegion R off hk),rsecond R off hk]
      exact (stable s middle inputs R off).symm
  · intro R off outside
    rw [fsecond R off (fun h => outside (Or.inr h)),ffirst R off (fun h => outside (Or.inl h))]

def combinedLaunch {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (σ : ℝ)
    (p : Program shape 64) (s : State α) : Option (State α) :=
  exec (withDot (scalarAlgebra a) (scalarDotPrimitive a)) (combinedKernel shape σ p.tile) (withProgram p s)

def backwardGridTarget {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (σ : ℝ)
    (p : Program shape 64) (s : State α) (R : RegionName) (off : Nat) : Cell α :=
  if R="dq" then queryGridTarget a shape σ p s R off else keyValueGridTarget a shape σ p s R off

theorem backward_grid {α : Type} [Inhabited α] (a : Scalars α) (shape : Shape) (σ : ℝ) (s : State α)
    (keyWidth : 0<shape.BK) (valueWidth : 0<shape.BV) :
    ∃ final,runPrograms (combinedLaunch a shape σ) (programs shape 64) s=some final ∧ SameInputs s final ∧
      (∀ p∈programs shape 64,∀ R off,backwardWrites shape p R off → final.mem R off=backwardGridTarget a shape σ p s R off) ∧
      (∀ R off,(∀ p∈programs shape 64,¬backwardWrites shape p R off) → final.mem R off=s.mem R off) := by
  refine runPrograms_correct _ _ _ ?_ (backward_write_unique shape) ?_ ?_ _ _
  · intro p R off member writes
    rcases writes with ⟨eq,_⟩|⟨eq,_⟩|⟨eq,_⟩ <;> subst R <;>
      simp [queryRectangle,keyRectangle,valueRectangle,outputRectangle] at member
  · intro p before after same R off
    simp only [backwardGridTarget,queryGridTarget,keyValueGridTarget,sameReaders shape before after same]
  · intro p before
    unfold combinedLaunch
    rw [combined_exec]
    apply sequential_step _ _
      ((queryRectangle shape p.tile).writes 64 shape.BK) (keyValueWrites shape p)
      (fun t => queryGridTarget a shape σ p t) (fun t => keyValueGridTarget a shape σ p t)
    · intro R off h; exact h.1
    · intro R off h eq
      subst R
      rcases h with h|h <;> simpa [keyRectangle,queryRectangle,valueRectangle,outputRectangle] using h.1
    · intro R off member h
      have eq := h.1
      change R="dq" at eq
      subst R
      simp at member
    · intro t u same R off
      simp only [keyValueGridTarget,sameReaders shape t u same]
    · intro t
      obtain ⟨final,executed,reads,frame⟩ := query_source a shape σ p.tile t
      refine ⟨final,executed,?_,frame⟩
      intro R off write
      rcases write with ⟨rfl,⟨⟨r,e,⟨⟩⟩,active,rfl⟩⟩
      erw [queryGridTarget,rectangleTarget_at _ _ _ _ active]
      exact reads r e active
    · intro t
      obtain ⟨final,executed,keyReads,valueReads,frame⟩ := keyValue_source a shape σ p.tile t keyWidth valueWidth
      refine ⟨final,executed,?_,frame⟩
      intro R off write
      rcases write with ⟨rfl,⟨⟨r,e,⟨⟩⟩,active,rfl⟩⟩|⟨rfl,⟨⟨r,v,⟨⟩⟩,active,rfl⟩⟩
      · simp only [keyValueGridTarget,keyRectangle,queryRectangle,ite_true]
        erw [rectangleTarget_at _ _ _ _ active]
        exact keyReads r e active
      · simp only [keyValueGridTarget,valueRectangle,outputRectangle,show ¬(("dv" : RegionName)="dk") by decide,ite_false]
        erw [rectangleTarget_at _ _ _ _ active]
        exact valueReads r v active

#axiomsClean combined_exec
#axiomsClean backward_write_unique
#axiomsClean sequential_step
#axiomsClean backward_grid
#specNonCircular backwardGridTarget avoiding [combinedLaunch,combinedKernel,FP.Structural.exec,runPrograms]
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
