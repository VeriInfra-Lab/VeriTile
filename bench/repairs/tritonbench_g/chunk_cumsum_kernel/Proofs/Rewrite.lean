import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.SourcePreservation
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.FPIndexBridge
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Triton.Float.ProfiledSoundness

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.Rewrite
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

abbrev parts := FPIndexBridge.parts

def candidateParts (input output : RegionName) (T BT : Nat) : Source.IntRange :=
  (Source.intRange? (sumLeftKernel .fp32 .fp32 input output T BT).surfaceBody "i_t").getD
    ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩

def additionSite (input output : RegionName) (T BT : Nat) : Site :=
  .addFP32 (Source.prefixBefore (parts input output T BT).body "b_z") []
    (.ref .real [] "b_z") (.ref .real [] "b_zz")

def originalBody (input output : RegionName) (T BT : Nat) : Program where
  kernel := ComputeKernel.mk [input] [output] (parts input output T BT).body
  domain := [additionSite input output T BT]

def candidateBody (input output : RegionName) (T BT : Nat) : Program :=
  { originalBody input output T BT with
    kernel := ComputeKernel.mk [input] [output] (candidateParts input output T BT).body }

/-- The reached body uses the shared strict FP32 addition certificate. -/
@[spec_rule] theorem body_equiv (input output : RegionName) (T BT : Nat) :
    originalBody input output T BT ≡[([] : Spec.Assumptions FP.GuardedFragment)]
      candidateBody input output T BT := by
  equiv_decompose
  fp_prove

/-- Check actual operands in reached source states, except for the final
carry update: that register is never read by a later output. No assumption
on unspecified tail values or on complete execution/results occurs here. -/
def FiniteCarry {α : Type} [Inhabited α] (M : Algebra α) (D : FP.Guarded.Domain α)
    (input output : RegionName) (T BT : Nat) (initial : State α) : Prop :=
  ∀ first, run (engine M) (parts input output T BT).before initial=some first →
    ∀ (q : Nat) u, IntRangeReachable (engine M) "i_t" (((T+BT-1)/BT : Nat) : Int) 1
      (parts input output T BT).body 0 first (q:Int) u → q+1<(T+BT-1)/BT →
      (additionSite input output T BT).Holds M D (u.setReg "i_t" .int [] (fun _ => (q:Int)))

/-- Both displayed final bodies execute. Their different carry results are
private; the subsequent store reads b_o, which was computed before the swap. -/
theorem final_body {α : Type} [Inhabited α] (ops : FPRow.Ops α)
    (input output : RegionName) (T BT pid index : Nat) (s : State α) (ready : FPRow.Ready pid s) :
    ∃ a b,
      run (engine ops.algebra) (parts input output T BT).body
        (s.setReg "i_t" .int [] (fun _ => (index:Int)))=some a ∧
      run (engine ops.algebra) (candidateParts input output T BT).body
        (s.setReg "i_t" .int [] (fun _ => (index:Int)))=some b ∧ a.mem=b.mem := by
  obtain ⟨hp,hz⟩ := ready
  obtain ⟨carry,hz⟩ := Option.isSome_iff_exists.mp hz
  simp (config := {maxSteps := 400000}) [parts,FPIndexBridge.parts,candidateParts,SourcePreservation.original,LegacyKernels.repairedKernel,SourcePreservation.candidate,LegacyKernels.sumLeftKernel,
    ComputeKernel.surfaceBody,Source.intRange?,run,step,FP.Structural.evalExpr,
    evalComputeOp,evalOp_unfold,evalOffsets,FP.Structural.store,Option.bind,engine,Algebra.withDefaultPrecision,
    resolvePrecision,FPRow.Ops.algebra,ComputeDType.eraseDType,numeric,numericLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,hp,hz,TileShape.indexToList]
  apply FPIndexBridge.masked_mem
  rfl

 theorem original_body {α : Type} [Inhabited α] (ops : FPRow.Ops α)
    (input output : RegionName) (T BT pid index : Nat) (s : State α) (ready : FPRow.Ready pid s) :
    ∃ a, run (engine ops.algebra) (parts input output T BT).body
        (s.setReg "i_t" .int [] (fun _ => (index:Int)))=some a ∧ FPRow.Ready pid a := by
  obtain ⟨a,b,ha,_,related,hb⟩ := FPIndexBridge.body ops input output T BT pid index s s
    ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩ ready
  refine ⟨a,ha,related.2.2.2.2.1.trans hb.1,?_⟩
  rw [related.2.2.2.2.2]
  exact hb.2

@[spec_rule] theorem loop {α : Type} [Inhabited α]
    (ops : FPRow.Ops α) (D : FP.Guarded.Domain α)
    (model : FP.Guarded.Models [] ops.algebra D)
    (input output : RegionName) (T BT pid : Nat) (initial first : State α)
    (prelude : run (engine ops.algebra) (parts input output T BT).before initial=some first)
    (finite : FiniteCarry ops.algebra D input output T BT initial)
    (q : Nat) (u : State α) (ready : FPRow.Ready pid u)
    (reached : IntRangeReachable (engine ops.algebra) "i_t" (((T+BT-1)/BT : Nat) : Int) 1
      (parts input output T BT).body 0 first (q:Int) u) :
    ∃ a b,
      rangeInt (engine ops.algebra) "i_t" (q:Int) (((T+BT-1)/BT : Nat) : Int) 1
        (parts input output T BT).body u=some a ∧
      rangeInt (engine ops.algebra) "i_t" (q:Int) (((T+BT-1)/BT : Nat) : Int) 1
        (candidateParts input output T BT).body u=some b ∧ a.mem=b.mem := by
  by_cases hq : q<(T+BT-1)/BT
  · by_cases later : q+1<(T+BT-1)/BT
    · obtain ⟨mid,hm,hr⟩ := original_body ops input output T BT pid q u ready
      have equal := FP.ProfiledRewrite.sound (body_equiv input output T BT)
        α ops.algebra D model (u.setReg "i_t" .int [] (fun _ => (q:Int)))
        (by
          intro site hs
          have eq : site = additionSite input output T BT := by simpa [originalBody] using hs
          subst site
          exact finite first prelude q u reached later)
      have candidate : run (engine ops.algebra) (candidateParts input output T BT).body
          (u.setReg "i_t" .int [] (fun _ => (q:Int)))=some mid := equal.symm.trans hm
      have next : IntRangeReachable (engine ops.algebra) "i_t" (((T+BT-1)/BT : Nat) : Int) 1
          (parts input output T BT).body 0 first ((q+1:Int)) mid := by
        simpa using IntRangeReachable.next reached (by decide : (1:Nat)≠0) (by exact_mod_cast hq) hm
      obtain ⟨a,b,ha,hb,mem⟩ := loop ops D model input output T BT pid initial first prelude finite
        (q+1) mid hr (by simpa using next)
      refine ⟨a,b,?_,?_,mem⟩
      · rw [rangeInt,if_neg (by decide : ¬(1:Nat)=0),if_pos (by exact_mod_cast hq),hm]
        simpa using ha
      · rw [rangeInt,if_neg (by decide : ¬(1:Nat)=0),if_pos (by exact_mod_cast hq),candidate]
        simpa using hb
    · obtain ⟨a,b,ha,hb,mem⟩ := final_body ops input output T BT pid q u ready
      refine ⟨a,b,?_,?_,mem⟩
      · rw [rangeInt,if_neg (by decide : ¬(1:Nat)=0),if_pos (by exact_mod_cast hq),ha]
        change rangeInt (engine ops.algebra) "i_t" ((q:Int)+1) (((T+BT-1)/BT : Nat) : Int) 1
          (parts input output T BT).body a=some a
        rw [rangeInt,if_neg (by decide : ¬(1:Nat)=0),if_neg (by exact_mod_cast later)]
      · rw [rangeInt,if_neg (by decide : ¬(1:Nat)=0),if_pos (by exact_mod_cast hq),hb]
        change rangeInt (engine ops.algebra) "i_t" ((q:Int)+1) (((T+BT-1)/BT : Nat) : Int) 1
          (candidateParts input output T BT).body b=some b
        rw [rangeInt,if_neg (by decide : ¬(1:Nat)=0),if_neg (by exact_mod_cast later)]
  · refine ⟨u,u,?_,?_,rfl⟩
    all_goals rw [rangeInt,if_neg (by decide : ¬(1:Nat)=0),if_neg (by exact_mod_cast hq)]
termination_by (T+BT-1)/BT-q
 decreasing_by omega

theorem candidate_split (input output : RegionName) (T BT : Nat) :
    (sumLeftKernel .fp32 .fp32 input output T BT).surfaceBody =
      (parts input output T BT).before ++
        [.forRangeInt "i_t" (parts input output T BT).start
          (parts input output T BT).stop (parts input output T BT).stride
          (candidateParts input output T BT).body] := rfl

@[spec_rule] theorem successful_pair {α : Type} [Inhabited α]
    (ops : FPRow.Ops α) (D : FP.Guarded.Domain α)
    (model : FP.Guarded.Models [] ops.algebra D)
    (input output : RegionName) (T BT : Nat) (u : State α)
    (finite : FiniteCarry ops.algebra D input output T BT u) :
    ∃ a b, FP.Structural.exec (engine ops.algebra) (repairedKernel .fp32 .fp32 input output T BT) u=some a ∧
      FP.Structural.exec (engine ops.algebra) (sumLeftKernel .fp32 .fp32 input output T BT) u=some b ∧
      a.mem=b.mem ∧ FPRow.Frame output (u.pids 0) T u a ∧ FPRow.Frame output (u.pids 0) T u b := by
  obtain ⟨first,hpre,ready,_⟩ := FPRow.prelude ops .fp32 .fp32 input output T BT u
  have prelude : run (engine ops.algebra) (parts input output T BT).before u=some first := by
    rw [FPIndexBridge.before]; exact hpre
  obtain ⟨a,b,ha,hb,mem⟩ := loop ops D model input output T BT (u.pids 0) u first prelude finite
    0 first ready .initial
  have aexec : FP.Structural.exec (engine ops.algebra) (repairedKernel .fp32 .fp32 input output T BT) u=some a := by
    change run (engine ops.algebra) (repairedKernel .fp32 .fp32 input output T BT).surfaceBody u=some a
    rw [FPIndexBridge.split_source,run_append,prelude]
    simp only [Option.bind_some,run]
    have hs : step (engine ops.algebra)
        (.forRangeInt "i_t" (parts input output T BT).start
          (parts input output T BT).stop (parts input output T BT).stride
          (parts input output T BT).body) first=some a := by
      simp only [step]
      change rangeInt (engine ops.algebra) "i_t" 0 (((T+BT-1)/BT : Nat) : Int) 1
        (parts input output T BT).body first=some a
      exact ha
    rw [hs]
    rfl
  have bexec : FP.Structural.exec (engine ops.algebra) (sumLeftKernel .fp32 .fp32 input output T BT) u=some b := by
    change run (engine ops.algebra) (sumLeftKernel .fp32 .fp32 input output T BT).surfaceBody u=some b
    rw [candidate_split,run_append,prelude]
    simp only [Option.bind_some,run]
    have hs : step (engine ops.algebra)
        (.forRangeInt "i_t" (parts input output T BT).start
          (parts input output T BT).stop (parts input output T BT).stride
          (candidateParts input output T BT).body) first=some b := by
      simp only [step]
      change rangeInt (engine ops.algebra) "i_t" 0 (((T+BT-1)/BT : Nat) : Int) 1
        (candidateParts input output T BT).body first=some b
      exact hb
    rw [hs]
    rfl
  obtain ⟨t,ht,frame⟩ := FPIndexBridge.complete ops input output T BT u
  have same : t=a := Option.some.inj (ht.symm.trans aexec)
  subst t
  exact ⟨a,b,aexec,bexec,mem,frame,fun r off hn => (congrFun (congrFun mem.symm r) off).trans (frame r off hn)⟩

#print_fp_assumptions body_equiv
#axiomsClean final_body
#axiomsClean successful_pair
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.Rewrite
