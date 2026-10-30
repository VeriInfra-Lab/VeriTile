import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.GraphSafety
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.Grid

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.SourceGraph
open VeriTile.Triton FP.Structural BmmOptimized Source Execution
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem range_safe (p : Scalar α) (rest : Algebra α) (cfg : Parameters)
    (bounds : RegionName → Nat) (initial current : State (Node α)) (k : Nat)
    (h : Invariant (scalar p) cfg initial k current)
    (ha : ∀ q,q < iterations cfg → ∀ i : AIndex cfg,
      (cfg.divM=Bool.true ∨ row cfg initial i.1<cfg.M) ∧
        (cfg.divK=Bool.true ∨ inner cfg q i.2.1<cfg.K) →
      aOffset cfg initial q i<bounds cfg.A)
    (hb : ∀ q,q < iterations cfg → ∀ i : BIndex cfg,
      (cfg.divK=Bool.true ∨ inner cfg q i.1<cfg.K) ∧
        (cfg.divN=Bool.true ∨ column cfg initial i.2.1<cfg.N) →
      bOffset cfg initial q i<bounds cfg.B) :
    Safety.rangeSafe (engine p rest) bounds "_" k (iterations cfg) 1 (body cfg) current := by
  by_cases active : k < iterations cfg
  · rw [Safety.rangeSafe_unfold,if_neg (by decide),if_pos active]
    have hs:=loopRegister (scalar p) cfg initial current k h
    refine ⟨body_safe p rest cfg bounds initial _ k hs (ha k active) (hb k active),?_⟩
    rw [←body_equal,bodyExecution (scalar p) (base p rest) cfg initial _ k hs]
    exact range_safe p rest cfg bounds initial _ (k+1)
      (bodyInvariant (scalar p) cfg initial _ k hs) ha hb
  · rw [Safety.rangeSafe_unfold,if_neg (by decide),if_neg active]
    trivial
termination_by iterations cfg-k
decreasing_by omega

theorem source_safe (p : Scalar α) (rest : Algebra α) (cfg : Parameters)
    (bounds : RegionName → Nat) (initial : State (Node α))
    (ha : ∀ q,q < iterations cfg → ∀ i : AIndex cfg,
      (cfg.divM=Bool.true ∨ row cfg initial i.1<cfg.M) ∧
        (cfg.divK=Bool.true ∨ inner cfg q i.2.1<cfg.K) →
      aOffset cfg initial q i<bounds cfg.A)
    (hb : ∀ q,q < iterations cfg → ∀ i : BIndex cfg,
      (cfg.divK=Bool.true ∨ inner cfg q i.1<cfg.K) ∧
        (cfg.divN=Bool.true ∨ column cfg initial i.2.1<cfg.N) →
      bOffset cfg initial q i<bounds cfg.B)
    (ho : ∀ i : CIndex cfg,active cfg initial i → oOffset cfg initial i<bounds cfg.O) :
    FP.Structural.TraceSafe (engine p rest) bounds (kernel cfg) initial := by
  unfold FP.Structural.TraceSafe
  rw [source_decomposition,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prelude_safe p rest cfg bounds initial,?_⟩
  intro start ran
  have equal : start=preludeResult (scalar p) cfg initial :=
    Option.some.inj (ran.symm.trans ((prelude_equal p rest cfg initial).symm.trans
      (preludeExecution (scalar p) (base p rest) cfg initial)))
  subst start
  have hi:=preludeInvariant (scalar p) cfg initial
  obtain ⟨middle,loop,hmid⟩:=loopExecution (scalar p) (base p rest) cfg initial _ 0 (by omega) hi
  rw [List.singleton_append,Safety.traceSafeList_unfold]
  refine ⟨?_,?_⟩
  · simpa [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,hi.iterations]
      using range_safe p rest cfg bounds initial _ 0 hi ha hb
  · simp only [FP.Structural.step,evalOp_unfold,hi.iterations,bind,Option.bind]
    rw [←range_equal,loop]
    exact epilogue_safe p rest cfg bounds initial middle hmid ho

private theorem optional_mask (n tile coordinate : Nat) :
    (decide (n%tile=0)=Bool.true ∨ coordinate<n) ↔ axisMask n tile coordinate := by
  unfold axisMask
  split_ifs <;> simp_all

theorem a_window (s : Shape) (c : Config) (pid : Program) (hp : pid∈programs s c)
    (initial : State α) (q : Nat) (hq : q < iterations (parameters s c))
    (i : AIndex (parameters s c))
    (mask : ((parameters s c).divM=Bool.true ∨
        row (parameters s c) (Bridge.start s c pid initial) i.1<s.m) ∧
      ((parameters s c).divK=Bool.true ∨ inner (parameters s c) q i.2.1<s.k)) :
    aOffset (parameters s c) (Bridge.start s c pid initial) q i<s.batch*s.m*s.k := by
  let ci : TileIndex [c.tm,c.tn] := (i.1,⟨0,(config_bounds c).2.2.1⟩,PUnit.unit)
  have masks:=source_masks s c pid hp (Bridge.laneOf ci) (Bridge.lane_mem c ci)
    q i.2.1.val hq i.2.1.isLt
  have active : (cellOf s c pid (Bridge.laneOf ci)).row<s.m ∧ q*32+i.2.1.val<s.k := by
    apply masks.1.mp
    simpa only [parameters,optional_mask] using mask
  have batch:=(source_lane_tiles s c pid hp (Bridge.laneOf ci) (Bridge.lane_mem c ci)).2.2
  have bound:=active_a_bound s _ _ _ batch active.1 active.2
  change s.a pid.2.2 (cellOf s c pid (Bridge.laneOf ci)).row (q*32+i.2.1.val)<_ at bound
  rw [←Bridge.a_offset s c pid initial ci q i.2.1] at bound
  exact bound

theorem b_window (s : Shape) (c : Config) (pid : Program) (hp : pid∈programs s c)
    (initial : State α) (q : Nat) (hq : q < iterations (parameters s c))
    (i : BIndex (parameters s c))
    (mask : ((parameters s c).divK=Bool.true ∨ inner (parameters s c) q i.1<s.k) ∧
      ((parameters s c).divN=Bool.true ∨
        column (parameters s c) (Bridge.start s c pid initial) i.2.1<s.n)) :
    bOffset (parameters s c) (Bridge.start s c pid initial) q i<s.batch*s.k*s.n := by
  let ci : TileIndex [c.tm,c.tn] := (⟨0,(config_bounds c).1⟩,i.2.1,PUnit.unit)
  have masks:=source_masks s c pid hp (Bridge.laneOf ci) (Bridge.lane_mem c ci)
    q i.1.val hq i.1.isLt
  have active : q*32+i.1.val<s.k ∧ (cellOf s c pid (Bridge.laneOf ci)).column<s.n := by
    apply masks.2.1.mp
    simpa only [parameters,optional_mask] using mask
  have batch:=(source_lane_tiles s c pid hp (Bridge.laneOf ci) (Bridge.lane_mem c ci)).2.2
  have bound:=active_b_bound s _ _ _ batch active.1 active.2
  change s.b pid.2.2 (q*32+i.1.val) (cellOf s c pid (Bridge.laneOf ci)).column<_ at bound
  rw [←Bridge.b_offset s c pid initial ci q i.1] at bound
  exact bound

theorem selected_safe (s : Shape) (c : Config) (p : Scalar α) (rest : Algebra α)
    (pid : Program) (hp : pid∈programs s c) (bounds : RegionName → Nat)
    (ha : ∀ i,i<s.batch*s.m*s.k → i<bounds "A")
    (hb : ∀ i,i<s.batch*s.k*s.n → i<bounds "B")
    (ho : ∀ i,i<s.size → i<bounds "O") (initial : State (Node α)) :
    TraceSafe (engine p rest) bounds (kernel (parameters s c)) (Bridge.start s c pid initial) := by
  apply source_safe
  · intro q hq i hm
    exact ha _ (a_window s c pid hp initial q hq i hm)
  · intro q hq i hm
    exact hb _ (b_window s c pid hp initial q hq i hm)
  · intro i hm
    have active:=(Bridge.active_exact s c pid hp initial i).mp hm
    rw [Bridge.o_offset]
    exact ho _ (address_bound s _ active)

#axiomsClean source_safe
#axiomsClean selected_safe
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.SourceGraph
