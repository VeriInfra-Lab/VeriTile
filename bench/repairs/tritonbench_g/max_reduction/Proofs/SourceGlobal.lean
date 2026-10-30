import bench.repairs.tritonbench_g.max_reduction.Proofs.SourceGrid

namespace VeriTile.Bench.Repairs.MaxReduction.SourceGrid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ApplyPenalty
noncomputable section
open Classical
set_option maxHeartbeats 2000000

def runFirst [Inhabited A] (e : Source.Engine A) (N B : Nat) : List Nat → State A → Option (State A)
  | [],s => some s
  | p::rest,s => do
      let t ← FP.Structural.exec e.algebra (stage1 "input" "middle" N B) (positioned s p 0)
      runFirst e N B rest t

theorem first_program [Inhabited A] (e : Source.Engine A) (N B : Nat) (hb : 0<B)
    (p : Nat) (s : State A) :
    ∃ t,FP.Structural.exec e.algebra (stage1 "input" "middle" N B) (positioned s p 0)=some t ∧
      flatten t=Scatter.write (flatten s) ("middle",p) (.mk .real
        (Source.scalarValue e (fun i => (s.mem "input" i).read .real) N B (p*B))) := by
  obtain ⟨t,he,hm⟩ := Source.stage1_exec e "input" "middle" N B (positioned s p 0) hb
  refine ⟨t,he,?_⟩
  have hf := congrArg (fun mem => fun q : RegionName × Nat => mem q.1 q.2) hm
  change flatten t=flatten ((positioned s p 0).write "middle" _ _) at hf
  rw [hf,write_flat]
  rfl

theorem first_chain [Inhabited A] (e : Source.Engine A) (N B : Nat) (hb : 0<B)
    (order : List Nat) (s : State A) :
    ∃ t,runFirst e N B order s=some t ∧
      flatten t=Scatter.run order (fun p => (("middle" : RegionName),p))
        (fun p => FP.Structural.Cell.mk .real (Source.scalarValue e (fun i => (s.mem "input" i).read .real) N B (p*B)))
        (flatten s) := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons p rest ih =>
    obtain ⟨mid,he,hm⟩ := first_program e N B hb p s
    obtain ⟨t,ht,htm⟩ := ih mid
    have input : (fun i => (mid.mem "input" i).read .real)=(fun i => (s.mem "input" i).read .real) := by
      funext i
      have frame := congrArg (fun f => f ("input",i)) hm
      simp [Scatter.write] at frame
      exact congrArg (FP.Structural.Cell.read .real) frame
    refine ⟨t,by simp only [runFirst,he]; exact ht,?_⟩
    rw [htm,input,hm]
    rfl

def runGlobal [Inhabited A] (first second : Source.Engine A) (N B W : Nat) (s : State A) : Option (State A) := do
  let reached ← runFirst first N B (List.range (Address.count N B)) s
  FP.Structural.exec second.algebra (stage2 "middle" "output" (Address.count N B) W) (positioned reached 0 0)

def globalTarget (first second : Source.Engine A) (input : Nat → A) (N B W : Nat) : A :=
  second.maximum W (fun p => if p.val<Address.count N B then Source.scalarValue first input N B (p.val*B)
    else second.padding)

/-- Actual complete two-kernel AST execution. Stage two reads the reached
intermediate allocation produced by all first-stage programs. -/
theorem global_whole [Inhabited A] (first second : Source.Engine A) (N B W : Nat)
    (hb : 0<B) (hw : 0<W) (s : State A) :
    ∃ t,runGlobal first second N B W s=some t ∧
      t.mem "output" 0=.mk .real (globalTarget first second (fun i => (s.mem "input" i).read .real) N B W) ∧
      (∀ (r : RegionName) (i : Nat), (r≠"middle" ∨ Address.count N B ≤ i) →
        (r≠"output" ∨ i≠0) → t.mem r i=s.mem r i) := by
  obtain ⟨mid,hm,hmid⟩ := first_chain first N B hb (List.range (Address.count N B)) s
  obtain ⟨t,ht,htmem⟩ := Source.stage2_exec second "middle" "output" (Address.count N B) W
    (positioned mid 0 0) hw
  have values : ∀ p,p<Address.count N B → (mid.mem "middle" p).read .real=
      Source.scalarValue first (fun i => (s.mem "input" i).read .real) N B (p*B) := by
    intro p hp
    have hv := Scatter.readback_injective (List.range (Address.count N B)) (fun p => (("middle" : RegionName),p))
      (fun p => FP.Structural.Cell.mk .real
        (Source.scalarValue first (fun i => (s.mem "input" i).read .real) N B (p*B))) (flatten s)
      p (List.mem_range.mpr hp) (by intro i j h; exact (Prod.mk.inj h).2)
    rw [←hmid] at hv
    change mid.mem "middle" p=_ at hv
    rw [hv]
    rfl
  have answer : Source.scalarValue second (fun i => (mid.mem "middle" i).read .real)
      (Address.count N B) W 0=globalTarget first second (fun i => (s.mem "input" i).read .real) N B W := by
    unfold Source.scalarValue globalTarget
    apply congrArg (second.maximum W)
    funext p
    simp only [Nat.zero_add]
    split_ifs with hp
    · exact values p.val hp
    · rfl
  refine ⟨t,by simp only [runGlobal,hm]; exact ht,?_,?_⟩
  · rw [htmem,State.write_same]
    change FP.Structural.Cell.mk .real _=FP.Structural.Cell.mk .real _
    exact congrArg (FP.Structural.Cell.mk .real) answer
  · intro r i hmidmiss houtmiss
    rw [htmem,State.write_other _ _ _ _ _ _ houtmiss]
    change flatten mid (r,i)=flatten s (r,i)
    rw [hmid]
    apply Scatter.frame
    intro p hp same
    have hr := congrArg Prod.fst same
    have hi := congrArg Prod.snd same
    have hp := List.mem_range.mp hp
    rcases hmidmiss with h|h
    · exact h hr
    · omega

#axiomsClean first_chain
#axiomsClean global_whole
end
end VeriTile.Bench.Repairs.MaxReduction.SourceGrid
