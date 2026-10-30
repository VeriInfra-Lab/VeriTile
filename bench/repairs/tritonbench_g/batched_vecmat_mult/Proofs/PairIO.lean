import bench.repairs.tritonbench_g.batched_vecmat_mult.IO
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.PairNumerics
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.PairSafety
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.IOSafety
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.PairIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open SourceGraph (Expr Word Memory Parameters PlanSelector engine treeValue sumValues
  row col outputOffset aOffset bOffset blockSum Outside)
open VecmatIO
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 30000

def MemoryEqual (cfg : FP.Config) (left right : Memory) : Prop :=
  ∀ r off, mapCell (observe cfg) (left r off)=mapCell (observe cfg) (right r off)

theorem expected_memory (cfg : FP.Config) (plans : PlanSelector) (p : Parameters)
    (s t : State Expr) (hp : s.pids=t.pids)
    (ha : s.mem p.A=t.mem p.A) (hb : s.mem p.B=t.mem p.B)
    (same : MemoryEqual cfg s.mem t.mem) :
    MemoryEqual cfg (SourceGraph.expectedMemory plans p s) (BlockFirst.expectedMemory plans p t) := by
  have offsets (i : TileIndex [p.BM,p.BN]) : outputOffset p s i=outputOffset p t i := by
    simp only [outputOffset,row,col,hp]
  have aux : ∀ (xs : List (TileIndex [p.BM,p.BN])) (l r : Memory), MemoryEqual cfg l r →
      MemoryEqual cfg
        (xs.foldl (fun mem i => fun reg off =>
          if reg=p.output ∧ off=outputOffset p s i then
            .mk .real (SourceGraph.accumulated plans p s (p.K/p.BK) i) else mem reg off) l)
        (xs.foldl (fun mem i => fun reg off =>
          if reg=p.output ∧ off=outputOffset p t i then
            .mk .real (BlockFirst.accumulated plans p t (p.K/p.BK) i) else mem reg off) r) := by
    intro xs
    induction xs with
    | nil => intro l r same; exact same
    | cons i xs ih =>
      intro l r same
      simp only [List.foldl_cons]
      apply ih
      intro reg off
      rw [offsets]
      dsimp only
      split
      · change Cell.mk .real (Expr.word _) = Cell.mk .real (Expr.word _)
        rw [BlockFirst.accumulated_evaluate cfg plans p s t hp ha hb]
      · exact same reg off
  exact aux _ _ _ same

def Exterior (p : Parameters) (r : RegionName) (off : Nat) : Prop :=
  ∀ (m : Fin p.M) (n : Fin p.N), r≠p.output ∨ off≠m.val*p.N+n.val

theorem local_exterior (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) (pid : Nat) (valid : pid<SourceGraph.gridCount p)
    (s : State Expr) (r : RegionName) (off : Nat) (outside : Exterior p r off) :
    Outside p ((VecmatIO.launch rest profile p pid).initial s) r off := by
  by_cases hr : r=p.output
  · right
    intro i
    obtain ⟨hm,hn⟩ := SourceGraph.coordinate_bounds p pid valid
    exact (outside
      ⟨_,SourceGraph.tiled_coordinate_lt p.M p.BM ⟨_,hm⟩ i.1⟩
      ⟨_,SourceGraph.tiled_coordinate_lt p.N p.BN ⟨_,hn⟩ i.2.1⟩).resolve_left (not_not.mpr hr)
  · exact Or.inl hr

/-- Pair all complete current-memory invocations. Both runs preserve raw
exterior cells; only their newly computed output graphs require observation. -/
theorem jobs (cfg : FP.Config) (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) (separate : SourceGraph.InputsSeparate p)
    (ps : List Nat) (valid : ∀ pid∈ps,pid<SourceGraph.gridCount p)
    (s t : State Expr) (ha : s.mem p.A=t.mem p.A) (hb : s.mem p.B=t.mem p.B)
    (same : MemoryEqual cfg s.mem t.mem) :
    ∃ left right,
      runSchedule (ps.map (VecmatIO.launch rest profile p)) s=some left ∧
      runSchedule (ps.map (launch rest profile p)) t=some right ∧
      MemoryEqual cfg left.mem right.mem ∧
      (∀ r off, Exterior p r off → left.mem r off=s.mem r off) ∧
      (∀ r off, Exterior p r off → right.mem r off=t.mem r off) := by
  induction ps generalizing s t with
  | nil => exact ⟨s,t,rfl,rfl,same,fun _ _ _ => rfl,fun _ _ _ => rfl⟩
  | cons pid ps ih =>
    let ls := (VecmatIO.launch rest profile p pid).initial s
    let rs := (launch rest profile p pid).initial t
    obtain ⟨l,lr,lm,lf⟩ := SourceGraph.complete_correct rest (KernelProfile.toPlans profile) p ls
    obtain ⟨r,rr,rm,rf⟩ := BlockFirst.complete_correct rest (KernelProfile.toPlans profile) p rs
    have eq : MemoryEqual cfg l.mem r.mem := by
      rw [lm,rm]
      exact expected_memory cfg _ p ls rs rfl ha hb same
    have pa : l.mem p.A=r.mem p.A := by
      funext off
      rw [lf p.A off (Or.inl separate.1),rf p.A off (Or.inl separate.1)]
      exact congrFun ha off
    have pb : l.mem p.B=r.mem p.B := by
      funext off
      rw [lf p.B off (Or.inl separate.2),rf p.B off (Or.inl separate.2)]
      exact congrFun hb off
    obtain ⟨left,right,lrun,rrun,eq,lframe,rframe⟩ :=
      ih (fun q hq => valid q (List.mem_cons_of_mem _ hq)) l r pa pb eq
    refine ⟨left,right,?_,?_,eq,?_,?_⟩
    · change ((VecmatIO.launch rest profile p pid).run s >>= _) = _
      change (exec (engine rest (KernelProfile.toPlans profile)) (SourceGraph.source p) ls >>= _) = _
      rw [lr]
      exact lrun
    · change ((launch rest profile p pid).run t >>= _) = _
      change (exec (engine rest (KernelProfile.toPlans profile)) (BlockFirst.source p) rs >>= _) = _
      rw [rr]
      exact rrun
    · intro reg off outside
      rw [lframe reg off outside]
      exact lf reg off (local_exterior rest profile p pid (valid pid (by simp)) s reg off outside)
    · intro reg off outside
      rw [rframe reg off outside]
      exact rf reg off (local_exterior rest profile p pid (valid pid (by simp)) t reg off outside)

theorem launch_safe (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) (bounds : RegionName → Nat) (windows : IOSafety.Windows p bounds)
    (pid : Nat) (valid : pid<SourceGraph.gridCount p) (s : State Expr) :
    (launch rest profile p pid).Safe bounds s := by
  obtain ⟨hm,hn⟩ := SourceGraph.coordinate_bounds p pid valid
  have row (i : Fin p.BM) := SourceGraph.tiled_coordinate_lt p.M p.BM ⟨_,hm⟩ i
  have col (j : Fin p.BN) := SourceGraph.tiled_coordinate_lt p.N p.BN ⟨_,hn⟩ j
  apply BlockFirst.source_safe rest (KernelProfile.toPlans profile) p bounds
  · intro c hc i k
    exact windows.1 ⟨_,row i⟩ ⟨_,SourceGraph.tiled_coordinate_lt p.K p.BK ⟨c,hc⟩ k⟩
  · intro c hc i j k
    exact windows.2.1 ⟨_,row i⟩ ⟨_,col j⟩ ⟨_,SourceGraph.tiled_coordinate_lt p.K p.BK ⟨c,hc⟩ k⟩
  · intro i
    exact windows.2.2 ⟨_,row i.1⟩ ⟨_,col i.2.1⟩

theorem jobs_safe (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) (bounds : RegionName → Nat) (windows : IOSafety.Windows p bounds)
    (ps : List Nat) (valid : ∀ pid∈ps,pid<SourceGraph.gridCount p) (s : State Expr) :
    ScheduleSafe bounds (ps.map (launch rest profile p)) s := by
  induction ps generalizing s with
  | nil => trivial
  | cons pid ps ih =>
    exact ⟨launch_safe rest profile p bounds windows pid (valid pid (by simp)) s,
      fun t _ => ih (fun q hq => valid q (by simp [hq])) t⟩

/-- Full actual floor-grid pair on the original unrestricted expression-input
binding, observing its native contracted FP32 words. Includes empty axes and
K=0; every CTA executes and both exterior memory frames are retained. -/
theorem equivalent (cfg : FP.Config) (rest : Algebra Expr) (profile : KernelProfile.Selector)
    (p : Parameters) (separate : SourceGraph.InputsSeparate p) :
    (io cfg rest profile p).Equivalent (candidate cfg rest profile p) := by
  intro vals _ bounds hb initial _
  have windows : IOSafety.Windows p bounds :=
    ⟨fun m k => hb.2.1 (0 : Fin 2) (m,k,()) trivial,
      fun m n k => hb.2.1 (1 : Fin 2) (m,n,k,()) trivial,
      fun m n => hb.2.2.1 (0 : Fin 1) (m,n,()) trivial⟩
  obtain ⟨left,right,lrun,rrun,eq,lf,rf⟩ := jobs cfg rest profile p separate
    (List.range (SourceGraph.gridCount p)) (fun _ h => List.mem_range.mp h)
    initial initial rfl rfl (fun _ _ => rfl)
  refine ⟨IOSafety.grid rest profile p bounds windows initial,
    jobs_safe rest profile p bounds windows _ (fun _ h => List.mem_range.mp h) initial,
    left,right,lrun,rrun,?_,?_,?_⟩
  · intro o j _
    change Fin 1 at o
    have ho : o=(0 : Fin 1) := Subsingleton.elim _ _
    subst o
    exact eq p.output (j.1.val*p.N+j.2.1.val)
  · intro r off outside
    exact lf r off (fun m n => outside.1 (0 : Fin 1) (m,n,()) trivial)
  · intro r off outside
    exact rf r off (fun m n => outside.1 (0 : Fin 1) (m,n,()) trivial)

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.PairIO
