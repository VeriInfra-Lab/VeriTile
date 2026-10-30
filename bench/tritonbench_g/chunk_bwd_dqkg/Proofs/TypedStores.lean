import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedPost
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedLoop
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

@[simp] theorem masked_writes_regs {ι : Type} (p : ι → Prop) [DecidablePred p]
    (region : ι → RegionName) (offset : ι → Nat) (value : ι → Cell α) (xs : List ι) (s : State α) :
    (xs.foldl (fun t i=>if p i then t.write (region i) (offset i) (value i) else t) s).regs=s.regs := by
  induction xs generalizing s with
  | nil=>rfl
  | cons i xs ih=>simp only [List.foldl_cons,ih];split <;> rfl

def stores (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : OutputReady c s) :
    {t : State α // run (engine (withDot M dot)) ((parts c).after.drop 17) s=some t} := by
  have hdq := Classical.choose_spec ready.dq
  generalize hvdq : Classical.choose ready.dq = vdq at hdq
  have hdk := Classical.choose_spec ready.dk
  generalize hvdk : Classical.choose ready.dk = vdk at hdk
  have hdg := Classical.choose_spec ready.dg
  generalize hvdg : Classical.choose ready.dg = vdg at hdg
  have hpq := Classical.choose_spec ready.pq
  generalize hvpq : Classical.choose ready.pq = vpq at hpq
  have hpk := Classical.choose_spec ready.pk
  generalize hvpk : Classical.choose ready.pk = vpk at hpk
  have hpg := Classical.choose_spec ready.pg
  generalize hvpg : Classical.choose ready.pg = vpg at hpg
  refine ⟨?_,?_⟩
  swap
  simp only [parts,kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
    String.reduceEq,beq_self_eq_true,↓reduceIte,List.take,List.drop]
  simp [run,step.eq_3,evalExpr,evalComputeOp,evalOp_unfold,FP.Structural.store,
    Option.bind,engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,
    ComputeDType.eraseDType,ofFloat,toFloat,State.setReg,hdq,hdk,hdg,hpq,hpk,hpg]
  rfl

def finish (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (ready : Ready c s) : State α :=
  (stores M dot c (post M dot c s ready).val (post_ready M dot c s ready)).val

theorem finish_runs (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (ready : Ready c s) :
    run (engine (withDot M dot)) (parts c).after s=some (finish M dot c s ready) := by
  rw [←List.take_append_drop 17 (parts c).after,run_append]
  simp only [(post M dot c s ready).property,Option.bind_some]
  exact (stores M dot c (post M dot c s ready).val (post_ready M dot c s ready)).property

def finalState (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) : State α :=
  finish M dot c (loopState M dot c s).val (loopState M dot c s).property

theorem source_parts (c : Config) : (kernel c).surfaceBody=(parts c).before++
    [.forRangeDyn "i_v" (parts c).start (parts c).stop (parts c).stride (parts c).body]++(parts c).after := by rfl

theorem source_runs (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) :
    FP.Structural.exec (engine (withDot M dot)) (kernel c) s=some (finalState M dot c s) := by
  change run (engine (withDot M dot)) (kernel c).surfaceBody s=_
  rw [source_parts c]
  simp only [List.append_assoc,run_append,(entry M dot c s).property,Option.bind_some,List.singleton_append]
  rw [run]
  simp only [loop_runs,bind,Option.bind_some,finalState]
  exact finish_runs M dot c _ (loopState M dot c s).property

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
