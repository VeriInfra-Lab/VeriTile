import bench.tritonbench_g.lora_expand_gemv.Proofs.IOSafetySetup
import bench.tritonbench_g.lora_expand_gemv.Proofs.IOSafetyLoop
import bench.tritonbench_g.lora_expand_gemv.Proofs.IOSafetyControl
import bench.tritonbench_g.lora_expand_gemv.Proofs.IOBinding

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOSafety
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false
attribute [local irreducible] Branches.headerExecution Branches.setupExecution

 theorem source_safe (rest : Algebra Word) (plans : ∀ n, ReductionPlan n)
    (g : Geometry) (raw : Nat → Int) (v : Valid g raw) (limits : RegionName → Nat)
    (bounded : IOBinding.Bounds g limits) (s : State Word) (reads : IOGrid.Reads g raw s.mem)
    (hb : s.pids 1<g.B) (hn : s.pids 0<g.N) :
    TraceSafe (Branches.engine rest .ieeeValue plans) limits
      (Branches.source .singleHalf .single g.parameters) s := by
  have hl := length_one g (by omega : 0<g.N)
  have hi : Branches.selected g.parameters s=logicalIndex g raw (s.pids 1) := reads.index _ hb
  change Safety.traceSafeList (Branches.engine rest .ieeeValue plans) limits _ s
  rw [Branches.sourceShape,Safety.traceSafeList_append]
  refine ⟨header_safe rest plans g limits s (bounded.index _ hb),?_⟩
  intro hd hdRun
  have hdEq : hd=(Branches.headerExecution rest .ieeeValue plans .singleHalf .single g.parameters s).val :=
    Option.some.inj (hdRun.symm.trans (Branches.headerExecution rest .ieeeValue plans .singleHalf .single g.parameters s).property)
  subst hd
  rw [Safety.traceSafeList_cons]
  refine ⟨?_,by intro t _; simp⟩
  rw [Safety.stepSafe_unfold]
  constructor
  · simp [Branches.guard,Branches.source,
      VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.source,ComputeKernel.surfaceBody,
      Safety.safeOp_unfold]
  · have hg : evalOp (Branches.engine rest .ieeeValue plans) none
        (Branches.guard .singleHalf .single g.parameters)
        (Branches.headerExecution rest .ieeeValue plans .singleHalf .single g.parameters s).val =
        some (fun _ => decide (Branches.selected g.parameters s≠-1)) :=
      Branches.guardEval rest .ieeeValue plans .singleHalf .single g.parameters s
    rw [hg]
    by_cases active : logicalIndex g raw (s.pids 1)=-1
    · simp [hi,active]
    · simp only [hi,active,ne_eq,not_false_eq_true,decide_true,ite_true]
      apply (active_safe_iff rest plans g limits _).mpr
      rw [Safety.traceSafeList_append]
      refine ⟨setup_safe rest plans g limits s ?_,?_⟩
      · intro k hk
        apply bounded.x _ _ hb
        rcases hk with even|hk
        · have := v.even even; have := k.isLt; omega
        · exact hk
      · intro prepared hp
        have hpEq : prepared=(Branches.setupExecution rest .ieeeValue plans .singleHalf .single g.parameters s).val :=
          Option.some.inj (hp.symm.trans (Branches.setupExecution rest .ieeeValue plans .singleHalf .single g.parameters s).property)
        subst prepared
        have inv := Branches.setupInvariant rest .ieeeValue plans .singleHalf .single g.parameters s
        have idBounds := (v.ids _ hb).resolve_left active
        have idNat : (logicalIndex g raw (s.pids 1)).toNat<g.L :=
          (Int.toNat_lt idBounds.1).2 idBounds.2
        have hw : ∀ (m : Fin 256) (k : Fin g.block),m.val<Branches.length g.parameters → k.val<g.K →
            Branches.bOffset g.parameters s m.val k.val<limits "W" := by
          intro m k hm hk
          have hm0 : m.val=0 := by omega
          have hbase : ((g.l0 : Int)*logicalIndex g raw (s.pids 1)+(s.pids 0 : Int)*1*(g.wn : Int)).toNat=
              (logicalIndex g raw (s.pids 1)).toNat*g.l0+s.pids 0*g.wn := by
            rw [Int.toNat_add (mul_nonneg (by positivity) idBounds.1) (by positivity),
              Int.toNat_mul (by positivity) idBounds.1]
            simp [Int.toNat_mul,Nat.mul_comm]
          unfold Branches.bOffset Branches.weightBase
          rw [hl,hi]
          change ((g.l0 : Int)*logicalIndex g raw (s.pids 1)+(s.pids 0 : Int)*1*(g.wn : Int)).toNat+
            m.val*g.wn+k.val*g.wk<limits "W"
          rw [hbase,hm0]
          simpa [woff] using bounded.w _ _ k.val idNat hn hk
        have hy : ∀ m : Fin 256,m.val<Branches.length g.parameters →
            Branches.outputOffset g.parameters s m.val<limits "Y" := by
          intro m hm
          have hm0 : m.val=0 := by omega
          unfold Branches.outputOffset Branches.outputBase
          rw [hl,hm0]
          simpa [Geometry.parameters,yoff] using bounded.y _ _ hb hn
        exact loop_safe rest plans g limits s _ inv hl hw hy

 theorem schedule_safe (rest : Algebra Word) (plans : ∀ n, ReductionPlan n)
    (g : Geometry) (raw : Nat → Int) (v : Valid g raw) (limits : RegionName → Nat)
    (bounded : IOBinding.Bounds g limits) (initial : Memory) (todo seen : List Nat) (s : State Word)
    (h : IOGrid.Snapshot plans g raw initial seen s.mem)
    (inside : ∀ q∈todo,q<g.count) (distinct : todo.Nodup)
    (disjoint : ∀ q∈todo,q∉seen) :
    ScheduleSafe limits (todo.map (LoraIO.launch rest plans g)) s := by
  induction todo generalizing seen s with
  | nil => trivial
  | cons q qs ih =>
    have hq := inside q (by simp)
    have hc := coordinates g q hq
    constructor
    · exact source_safe rest plans g raw v limits bounded
        ((LoraIO.launch rest plans g q).initial s) h.1 hc.2 hc.1
    · intro next hr
      obtain ⟨actual,he,hm⟩ := IOGrid.one_program rest plans g raw v initial seen s h q hq (disjoint q (by simp))
      have : next=actual := Option.some.inj (hr.symm.trans he)
      subst next
      have hn : IOGrid.Snapshot plans g raw initial (q::seen) actual.mem := by
        rw [hm]; exact IOGrid.snapshot_put plans g raw v initial seen s.mem h q hq
      have hd := List.nodup_cons.mp distinct
      apply ih (q::seen) actual hn (fun p hp => inside p (by simp [hp])) hd.2
      intro p hp
      have hpq : p≠q := by intro eq; exact hd.1 (eq ▸ hp)
      simp [hpq,disjoint p (by simp [hp])]

#axiomsClean source_safe
#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOSafety
