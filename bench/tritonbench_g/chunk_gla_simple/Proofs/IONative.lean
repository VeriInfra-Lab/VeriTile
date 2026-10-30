import bench.tritonbench_g.chunk_gla_simple.Proofs.IONativeBindings
import bench.tritonbench_g.chunk_gla_simple.Proofs.IONativeSafety

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PublicIO
open VeriTile Triton FP.Structural Host
open NativeGraph (Expr)
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

@[simp] theorem native_initial (c : Shape) (q : GridIndex (grid c)) (s : State Expr) :
    (nativeLaunch c q).initial s = FPGrid.launch c s q := rfl

theorem native_position (c : Shape) (q : GridIndex (grid c)) (s : State Expr) :
    IONativeSafety.position (FPGrid.launch c s q) = launchIndex c q := by
  simp [IONativeSafety.position,FPGrid.launch,HostFP.atProgram,GridIndex.toPids,
    launchIndex,grid,Shape.grid,Grid.rank]

theorem native_run (c : Shape) (jobs : List (GridIndex (grid c))) (s : State Expr) :
    runSchedule (jobs.map (nativeLaunch c)) s =
      FPGrid.run ScalarGraph.algebra c ScalarGraph.scaleTag jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons q rest ih =>
    simp only [List.map_cons,runSchedule,Invocation.run,native_initial,FPGrid.run]
    cases he : FP.Structural.exec (FP.ProfiledRewrite.engine ScalarGraph.algebra)
      (kernel c .fp32 .fp32 .fp32 .fp32 ScalarGraph.scaleTag) (FPGrid.launch c s q)
    · simp only [nativeLaunch,he,bind,Option.bind_none]
    · simp only [nativeLaunch,he,bind,Option.bind_some,ih]

theorem native_safe (c : Shape) (bounds : RegionName → Nat)
    (hb : ∀ r, capacity c r ≤ bounds r) (jobs : List (GridIndex (grid c))) (s : State Expr) :
    ScheduleSafe bounds (jobs.map (nativeLaunch c)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons q rest ih =>
    refine ⟨?_,fun t _ => ih t⟩
    change FP.Structural.TraceSafe (FP.ProfiledRewrite.engine ScalarGraph.algebra) bounds
      (kernel c .fp32 .fp32 .fp32 .fp32 ScalarGraph.scaleTag) (FPGrid.launch c s q)
    apply IONativeSafety.source_safe _ _ _ (ScalarGraph.dots c)
    rw [native_position]
    exact IOBindings.capacity_bounds c q bounds hb

theorem native_implements (c : Shape) (bt : 0<c.BT) :
    (nativeIO c).Implements (PublicNative.expressionFormula c bt) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds hb initial _
    exact native_safe c bounds (IONativeBindings.capacities c xs bounds hb) _ initial
  · intro xs initial hin
    obtain ⟨final,he,hr⟩ := FPGrid.whole_grid ScalarGraph.algebra c ScalarGraph.scaleTag
      (ScalarGraph.dots c) initial
    refine ⟨final,?_,?_,?_⟩
    · exact (native_run c (FPGrid.programs c) initial).trans he
    · intro o j _
      have ho : o=(0 : Fin 1) := by
        apply Fin.ext
        change o.val=0
        have h : o.val<1 := o.isLt
        omega
      subst o
      obtain ⟨h,t,v,⟨⟩⟩ := j
      change mapCell id (final.mem "o" (h.val*(c.T*c.V)+t.val*c.V+v.val)) =
        Cell.mk .fp32 (PublicNative.expression c (PublicNative.arrays c xs)
          h.val (t.val/c.BT) (v.val/c.BV) (lane c bt t v))
      rw [FPGrid.returned ScalarGraph.algebra c bt ScalarGraph.scaleTag
        (ScalarGraph.dots c) initial final hr h t v]
      change Cell.mk .fp32 _ = Cell.mk .fp32 _
      rw [IONativeBindings.formula_binding c bt xs initial hin]
      rfl
    · intro r off outside
      apply hr.2
      by_cases hro : r="o"
      · subst r
        right
        intro q i hi heq
        have head : (FPGrid.launch c initial q).pids 2 < c.heads := by
          have hp := congrArg VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Math.Launch.head
            (native_position c q initial)
          change (FPGrid.launch c initial q).pids 2 = (launchIndex c q).head at hp
          rw [hp]
          exact (launch_lt c q).2.2
        have ho := outside.1 (0 : Fin 1)
          (⟨(FPGrid.launch c initial q).pids 2,head⟩,
           ⟨FPValue.row c (FPGrid.launch c initial q) i.1,hi.1⟩,
           ⟨FPValue.column c (FPGrid.launch c initial q) i.2.1,hi.2⟩,()) trivial
        change ("o" : RegionName)≠"o" ∨ off≠FPValue.oOffset c (FPGrid.launch c initial q) i at ho
        exact ho.elim (fun h => h rfl) (fun h => h heq)
      · exact Or.inl hro

/-- The complete original Graph observation, with arbitrary symbolic inputs,
is a pure output observation of the proved exact source implementation. -/
theorem native_observes (c : Shape) (bt : 0<c.BT) (scale : BitVec 32) :
    (nativeIO c).Observes (fun _ _ => ScalarGraph.lower scale) (PublicNative.formula c bt scale) := by
  have hformula : (fun xs => (nativeIO c).mapOutputs (fun _ _ => ScalarGraph.lower scale)
      (PublicNative.expressionFormula c bt xs)) = PublicNative.formula c bt scale := by
    funext xs o
    fin_cases o
    rfl
  rw [← hformula]
  exact ProgramIO.Observes.of_implements (native_implements c bt) (fun _ _ => ScalarGraph.lower scale)

#axiomsClean native_implements
#axiomsClean native_observes
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PublicIO
