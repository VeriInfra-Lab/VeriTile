import bench.tritonbench_g.sgmv_expand_slice.Proofs.IndexSource
import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceAccessKernel

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.IndexPair
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem active_split (add : Bool) (p : KernelArgs) : active add p =
    SourceParts.activePrelude (abi add) p ++
      [.forRangeInt "k" (SourceParts.loopParts (abi add) p).start (SourceParts.loopParts (abi add) p).stop
        (SourceParts.loopParts (abi add) p).stride (SourceParts.body (abi add) p)] ++ suffix add p := by rfl

theorem outer_split (add : Bool) (p : KernelArgs) :
    (IndexReuse.kernel (abi add) p).surfaceBody =
      SourceParts.prelude (abi add) p ++ SourceGuards.guarded (abi add) p (active add p) := by rfl

theorem offsets (M : Algebra α) (p : KernelArgs) (s t : State α) (start adapter length : Nat)
    (acc : Values α .real [32,32])
    (h : SourceBody.Ready p (SourceInit.coordinates p start adapter length (s.pids 0))
      (SourceInit.state M p s start adapter length) t acc) :
    t.regs .int [32] "offset_m" = some (fun i =>
      (SourceSetup.rowIndex (s.pids 0 / tiles p.columns) i.1 : Int)) ∧
    t.regs .int [32] "offset_n" = some (fun i =>
      (SourceSetup.columnIndex (s.pids 0 % tiles p.columns) i.1 : Int)) := by
  constructor
  · simpa [SourceInit.state,SourceSetup.prepared,State.setReg] using
      h.preserved .int [32] "offset_m" (by decide)
  · simpa [SourceInit.state,SourceSetup.prepared,State.setReg] using
      h.preserved .int [32] "offset_n" (by decide)

theorem active_eq (M : Algebra α) (dot : SourceBody.DotImplementation α) (add : Bool)
    (p : KernelArgs) (s : State α) (start adapter length : Nat)
    (hstart : (s.mem p.starts p.batch).read .int = (start : Int))
    (hlength : (s.mem p.lengths p.batch).read .int = (length : Int)) :
    run (SourceBody.install M dot) (active add p) (SourceInit.guardedState p s adapter) =
      run (SourceBody.install M dot) (SourceParts.active (abi add) p) (SourceInit.guardedState p s adapter) := by
  let initial := SourceInit.state M p s start adapter length
  have hp : run (SourceBody.install M dot) (SourceParts.activePrelude (abi add) p)
      (SourceInit.guardedState p s adapter) = some initial :=
    SourceInit.prefix_run (SourceBody.install M dot) (abi add) p s start adapter length hstart hlength
  obtain ⟨u,hu,hr⟩ := SourceBounds.loop_step M dot (abi add) p
    (SourceInit.coordinates p start adapter length (s.pids 0)) initial initial (SourceInit.zero M)
    (SourceInit.ready M p s start adapter length)
  have idx := offsets M p s u start adapter length _ hr
  have same := suffix_eq (SourceBody.install M dot) add p
    (SourceExecution.outputCoordinates p start length (s.pids 0)) s u _
    (SourceExecution.ready_suffix M p s u start adapter length _ hlength hr) idx.1 idx.2
  rw [active_split,SourceExecution.active_split]
  simpa only [run_append,hp,Option.bind_some,run,hu,bind,Option.bind] using same

theorem reached (M : Algebra α) (add : Bool) (p : KernelArgs) (code : List ComputeStmt)
    (s : State α) (length adapter : Nat)
    (hlength : (s.mem p.lengths p.batch).read .int = (length : Int))
    (hadapter : (s.mem p.adapters p.batch).read .int = (adapter : Int))
    (live : 0 < length) (tile : s.pids 0 / tiles p.columns * 32 < length) :
    run M (SourceGuards.guarded (abi add) p code) (SourcePrelude.prepared p s) =
      run M code (SourceInit.guardedState p s adapter) := by
  have hmreg : (SourcePrelude.prepared p s).regs .int [] "M" = some (fun _ => (length : Int)) := by
    simp [SourcePrelude.prepared,SourcePrelude.scalarBindings,State.setReg,hlength]
  have hrow : (SourcePrelude.prepared p s).regs .int [] "pid_m" =
      some (fun _ => ((s.pids 0 / tiles p.columns : Nat) : Int)) := by
    simp [SourcePrelude.prepared,SourcePrelude.scalarBindings,State.setReg]
  have hbatch : (SourcePrelude.prepared p s).regs .int [] "cur_batch" = some (fun _ => (p.batch : Int)) := by
    simp [SourcePrelude.prepared,SourcePrelude.scalarBindings,State.setReg]
  rw [SourceGuards.run_guarded _ _ _ _ _ (length : Int) (adapter : Int)
    (s.pids 0 / tiles p.columns) hmreg hrow hbatch hadapter]
  have hl : ¬ (length : Int) ≤ 0 := by omega
  have ht : ¬ (length : Int) ≤ ((s.pids 0 / tiles p.columns : Nat) : Int)*32 := by omega
  have had : (adapter : Int) ≠ -1 := by omega
  simp only [hl,ht,had,if_false,SourceInit.guardedState]

/-- Exact complete source state equality on each original accepted active job. -/
theorem execution_eq (M : Algebra α) (dot : SourceBody.DotImplementation α)
    (a : SourceABI) (hi : a.input=.single) (hw : a.weight=.half) (ho : a.output=.single)
    (p : KernelArgs) (s : State α) (start adapter length : Nat)
    (hstart : (s.mem p.starts p.batch).read .int = (start : Int))
    (hlength : (s.mem p.lengths p.batch).read .int = (length : Int))
    (hadapter : (s.mem p.adapters p.batch).read .int = (adapter : Int))
    (live : 0 < length) (tile : s.pids 0 / tiles p.columns * 32 < length) :
    FP.Structural.exec (SourceBody.install M dot) (IndexReuse.kernel a p) s =
      FP.Structural.exec (SourceBody.install M dot) (VeriTile.Bench.Repairs.SgmvExpandSlice.kernel a p) s := by
  rw [candidate_carrier,original_carrier a hi hw ho]
  change run _ (IndexReuse.kernel (abi a.addInputs) p).surfaceBody s =
    run _ (SourceParts.outer (abi a.addInputs) p) s
  rw [outer_split,SourceGuards.outer_split]
  simp only [run_append,SourcePrelude.run_prelude,Option.bind_some,
    reached _ _ _ _ s length adapter hlength hadapter live tile]
  exact active_eq M dot a.addInputs p s start adapter length hstart hlength

theorem active_safe (M : Algebra α) (dot : SourceBody.DotImplementation α) (add : Bool)
    (p : KernelArgs) (s : State α) (start adapter length : Nat)
    (hstart : (s.mem p.starts p.batch).read .int = (start : Int))
    (hlength : (s.mem p.lengths p.batch).read .int = (length : Int))
    (bounds : RegionName → Nat) (bs : p.batch < bounds p.starts) (bl : p.batch < bounds p.lengths)
    (body : ∀ block, block < (p.inner+15)/16 → SourceAccess.BodyBounds bounds p
      (SourceInit.coordinates p start adapter length (s.pids 0)) block)
    (output : ∀ i, SourceSuffix.active p (SourceExecution.outputCoordinates p start length (s.pids 0)) i →
      SourceSuffix.offset p (SourceExecution.outputCoordinates p start length (s.pids 0)) i < bounds p.output) :
    Safety.traceSafeList (SourceBody.install M dot) bounds (active add p) (SourceInit.guardedState p s adapter) := by
  let initial := SourceInit.state M p s start adapter length
  have prefixRun := SourceInit.prefix_run (SourceBody.install M dot) (abi add) p s start adapter length hstart hlength
  have ready := SourceInit.ready M p s start adapter length
  have runLoop := SourceBounds.loop_step M dot (abi add) p
    (SourceInit.coordinates p start adapter length (s.pids 0)) initial initial (SourceInit.zero M) ready
  rw [active_split,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨SourceAccess.active_prefix_safe (SourceBody.install M dot) (abi add) p s adapter bounds bs,?_⟩
  intro u hu
  have prefixExec : run (SourceBody.install M dot) (SourceParts.activePrelude (abi add) p)
      (SourceInit.guardedState p s adapter) = some initial := prefixRun
  rw [prefixExec] at hu
  cases hu
  simp only [List.singleton_append]
  rw [Safety.traceSafeList_cons]
  refine ⟨SourceAccess.loop_step_safe M dot (abi add) p _ initial initial _ ready bounds body,?_⟩
  intro v hv
  obtain ⟨last,executed,lastReady⟩ := runLoop
  rw [executed] at hv
  have same := Option.some.inj hv
  subst v
  have idx := offsets M p s last start adapter length _ lastReady
  apply (suffix_safe_iff (SourceBody.install M dot) add p
    (SourceExecution.outputCoordinates p start length (s.pids 0)) s last _
    (SourceExecution.ready_suffix M p s last start adapter length _ hlength lastReady) idx.1 idx.2 bounds).mpr
  exact SourceAccess.suffix_safe (SourceBody.install M dot) (abi add) rfl rfl p
    (SourceExecution.outputCoordinates p start length (s.pids 0)) s last _
    (SourceExecution.ready_suffix M p s last start adapter length _ hlength lastReady) hlength bounds bl output

/-- Actual source accesses stay within exactly the original job's bounds. -/
theorem kernel_safe (M : Algebra α) (dot : SourceBody.DotImplementation α)
    (a : SourceABI) (p : KernelArgs) (s : State α) (start adapter length : Nat)
    (hstart : (s.mem p.starts p.batch).read .int = (start : Int))
    (hlength : (s.mem p.lengths p.batch).read .int = (length : Int))
    (hadapter : (s.mem p.adapters p.batch).read .int = (adapter : Int))
    (live : 0 < length) (tile : s.pids 0 / tiles p.columns * 32 < length)
    (bounds : RegionName → Nat) (bs : p.batch < bounds p.starts) (bl : p.batch < bounds p.lengths)
    (ba : p.batch < bounds p.adapters)
    (body : ∀ block, block < (p.inner+15)/16 → SourceAccess.BodyBounds bounds p
      (SourceInit.coordinates p start adapter length (s.pids 0)) block)
    (output : ∀ i, SourceSuffix.active p (SourceExecution.outputCoordinates p start length (s.pids 0)) i →
      SourceSuffix.offset p (SourceExecution.outputCoordinates p start length (s.pids 0)) i < bounds p.output) :
    TraceSafe (SourceBody.install M dot) bounds (IndexReuse.kernel a p) s := by
  rw [candidate_carrier]
  change Safety.traceSafeList _ bounds _ s
  rw [outer_split,Safety.traceSafeList_append]
  refine ⟨SourceAccess.prelude_safe (SourceBody.install M dot) (abi a.addInputs) p s bounds bl,?_⟩
  intro t ht
  rw [SourcePrelude.run_prelude] at ht
  cases ht
  apply SourceAccess.guarded_safe (SourceBody.install M dot) (abi a.addInputs) p _ _ length adapter (s.pids 0 / tiles p.columns)
  · simp [SourcePrelude.prepared,SourcePrelude.scalarBindings,State.setReg,hlength]
  · simp [SourcePrelude.prepared,SourcePrelude.scalarBindings,State.setReg]
  · simp [SourcePrelude.prepared,SourcePrelude.scalarBindings,State.setReg]
  · simpa [SourcePrelude.prepared,SourcePrelude.scalarBindings,State.setReg] using hadapter
  · exact live
  · exact tile
  · exact ba
  · exact active_safe M dot a.addInputs p s start adapter length hstart hlength bounds bs bl body output

#axiomsClean execution_eq
#axiomsClean kernel_safe
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.IndexPair
