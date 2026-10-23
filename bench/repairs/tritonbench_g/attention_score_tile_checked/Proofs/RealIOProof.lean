import bench.repairs.tritonbench_g.attention_score_tile_checked.RealIO

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.RealInterface.Proof
open VeriTile Triton Reused Geometry Physical
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccess
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000

theorem capacities (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (s : BlockState) (xs : Inputs ℝ c) (bounds : RegionBounds)
    (hb : (programIO c offset size scale window complement).InBounds bounds s xs) :
    ∀ r a, a < DenseSafety.bounds c r → a < bounds r := by
  intro r a ha
  by_cases hq : r="q"
  · subst r
    exact hb.1 (0 : Fin 4) ⟨a,by simpa [DenseSafety.bounds] using ha⟩ (Or.inl (by decide))
  by_cases hk : r="k"
  · subst r
    exact hb.1 (1 : Fin 4) ⟨a,by simpa [DenseSafety.bounds] using ha⟩ (Or.inl (by decide))
  by_cases hm : r="m"
  · subst r
    exact hb.1 (2 : Fin 4) ⟨a,by simpa [DenseSafety.bounds] using ha⟩ (Or.inl (by decide))
  by_cases ho : r="out"
  · subst r
    exact hb.2.1 (0 : Fin 1) ⟨a,by simpa [DenseSafety.bounds] using ha⟩ trivial
  simp [DenseSafety.bounds,hq,hk,hm,ho] at ha

theorem source_safe (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (bounds : RegionBounds) (cap : ∀ r a, a < DenseSafety.bounds c r → a < bounds r)
    (s : BlockState) (job : WholeGrid.Job c) :
    (AlgorithmStage.single (source c offset size scale window complement) (launch c job)).Safe bounds s := by
  change (DenseSource.kernel c offset size scale window complement).eraseDType.toAlgKernel.TraceSafe
    bounds (DenseSource.programState c s job.2.2.val job.1.val job.2.1.val)
  apply RealSource.safe bounds (DenseSource.programState c s job.2.2.val job.1.val job.2.1.val) c.B c.D
    (parameters c offset size scale window complement) Bool.false (supported_positive c.B c.block)
  rw [DenseSource.config_eq]
  change RealSafety.Bounds _ _ c.N "out"
    (RealSource.outputBase (parameters c offset size scale window complement) (job.1.val*c.H+job.2.1.val))
  rw [DenseSource.output_base]
  have old := DenseSafety.all_bounds c job.1 job.2.1 job.2.2 offset size scale window complement
  constructor
  · intro block hblock lane active
    exact cap _ _ (old.query block hblock lane active)
  · intro lane active
    exact cap _ _ (old.key lane active)
  · intro row hr
    exact cap _ _ (old.normalizer row hr)
  · intro lane hl
    exact cap _ _ (old.output lane hl)

theorem execution (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (order : List (WholeGrid.Job c)) (s t : BlockState)
    (run : WholeGrid.run c offset size scale window complement order s=some t) :
    ∃ trace : AlgorithmExecution (order.map fun job =>
        .single (source c offset size scale window complement) (launch c job)) s t,
      ∀ bounds, (∀ r a, a < DenseSafety.bounds c r → a < bounds r) → trace.Safe bounds := by
  induction order generalizing s with
  | nil =>
    have ht : s=t := Option.some.inj run
    subst t
    exact ⟨.nil s,fun _ _ => trivial⟩
  | cons job rest ih =>
    cases step : WholeGrid.invoke c offset size scale window complement job s with
    | none => simp [WholeGrid.run,step] at run
    | some middle =>
      have tail : WholeGrid.run c offset size scale window complement rest middle=some t := by
        simpa [WholeGrid.run,step] using run
      obtain ⟨trace,safe⟩ := ih middle tail
      have head : (AlgorithmStage.single (source c offset size scale window complement)
          (launch c job)).Runs s middle := by
        rw [AlgorithmStage.single_runs]
        exact step
      exact ⟨.cons head trace,fun bounds cap =>
        ⟨source_safe c offset size scale window complement bounds cap s job,safe bounds cap⟩⟩

theorem target_binding (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (s : BlockState) (xs : Inputs ℝ c)
    (binding : (programIO c offset size scale window complement).InputBinding s xs)
    (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) :
    RealReturned.target c s offset size scale window complement z h j =
      RealPhysical.logicalTarget c (logicalInputs c xs) offset size scale window complement z h j := by
  unfold RealReturned.target LogicalFormula.column RealPhysical.logicalTarget
  apply Finset.sum_congr rfl
  intro i _
  have hq (d : Fin c.D) : (DenseSource.inputs c s z.val h.val).query i.val d =
      (logicalInputs c xs).q z h i d := by
    simpa [DenseSource.inputs,RealLoads.inputs,logicalInputs,programIO,arity,region,
      ChanTy.read,Nat.mul_one,Nat.add_assoc] using
      binding (0 : Fin 4) ⟨_,query_bound c z h i d⟩ (Or.inl (by decide))
  have hk (d : Fin c.D) : (DenseSource.inputs c s z.val h.val).key j.val d =
      (logicalInputs c xs).k z ⟨h.val/(c.H/c.HK),grouped_head c h⟩ j d := by
    simpa [DenseSource.inputs,RealLoads.inputs,logicalInputs,programIO,arity,region,
      ChanTy.read,Shape.keyBase,Nat.mul_one,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
      binding (1 : Fin 4) ⟨_,key_all_heads_bound c z ⟨h.val/(c.H/c.HK),grouped_head c h⟩ j d⟩
        (Or.inl (by decide))
  have hm : (DenseSource.inputs c s z.val h.val).normalizer i.val =
      (logicalInputs c xs).m z h ⟨i.val,lt_of_lt_of_le i.isLt c.normalizerCapacity⟩ := by
    simpa [DenseSource.inputs,RealLoads.inputs,logicalInputs,programIO,arity,region,
      ChanTy.read,Shape.normalizerBase,Nat.add_assoc] using
      binding (2 : Fin 4) ⟨_,normalizer_bound c z h i⟩
        (Or.inl (by decide))
  simp only [LogicalFormula.weight,hq,hk,hm]

def castExecution {a b : List AlgorithmStage} {s t : BlockState} (h : a=b)
    (trace : AlgorithmExecution b s t) : AlgorithmExecution a s t := h ▸ trace

theorem castExecution_safe {a b : List AlgorithmStage} {s t : BlockState} (h : a=b)
    (trace : AlgorithmExecution b s t) (bounds : RegionBounds) :
    (castExecution h trace).Safe bounds ↔ trace.Safe bounds := by
  cases h
  rfl

theorem implements (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool) :
    (programIO c offset size scale window complement).ImplementsOn (initialOutput c)
      (formula c offset size scale window complement) := by
  intro xs domain s binding
  by_cases he : empty c
  · have stages : (programIO c offset size scale window complement).stages xs=[] := by
      simp [programIO,he]
    refine ⟨s,castExecution stages (AlgorithmExecution.nil s),?_,?_,?_⟩
    · intro bounds hb
      exact (castExecution_safe stages (.nil s) bounds).mpr trivial
    · intro o j active
      have hz : s.readMem "out" j.val=0 := (binding (3 : Fin 4) j (Or.inr he)).trans (domain he j)
      change s.readMem "out" j.val=formula c offset size scale window complement xs o j
      rw [hz]
      rcases he with hn | ho
      · letI : IsEmpty (Fin c.N) := ⟨fun i => by have hi := i.isLt; omega⟩
        simp [formula,RealPhysical.logicalTarget]
      · exact Fin.elim0 (j.cast ho)
    · intro r a hout hscr
      rfl
  · obtain ⟨t,run,same,values⟩ := RealReturned.grid_tensor c s offset size scale window complement
    obtain ⟨trace,safe⟩ := execution c offset size scale window complement (WholeGrid.jobs c) s t run
    have stages : (programIO c offset size scale window complement).stages xs=
        (WholeGrid.jobs c).map (fun job => .single (source c offset size scale window complement) (launch c job)) := by
      simp [programIO,he]
    refine ⟨t,castExecution stages trace,?_,?_,?_⟩
    · intro bounds hb
      exact (castExecution_safe stages trace bounds).mpr
        (safe bounds (capacities c offset size scale window complement s xs bounds hb))
    · intro o j active
      change t.readMem "out" j.val=formula c offset size scale window complement xs o j
      rw [←output_address c j,values]
      exact target_binding c offset size scale window complement s xs binding _ _ _
    · intro r a hout hscr
      apply RealReturned.run_frame c offset size scale window complement (WholeGrid.jobs c) s t run r a
      by_cases hr : r="out"
      · right
        by_contra ha
        have hl : a<c.outputSize := Nat.lt_of_not_ge ha
        have h := hout (0 : Fin 1) ⟨a,hl⟩ trivial
        exact h.elim (fun h => h hr) (fun h => h rfl)
      · exact Or.inl hr

#axiomsClean implements
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.RealInterface.Proof
