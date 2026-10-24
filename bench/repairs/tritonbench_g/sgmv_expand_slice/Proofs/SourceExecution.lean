import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceInit
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceBounds
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceGuards
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceSuffix

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceExecution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

def outputCoordinates (p : KernelArgs) (start length pid : Nat) : SourceSuffix.Coordinates :=
  ⟨start,length,pid / tiles p.columns,pid % tiles p.columns⟩

theorem active_split (abi : SourceABI) (p : KernelArgs) :
    SourceParts.active abi p = SourceParts.activePrelude abi p ++
      [.forRangeInt "k" (SourceParts.loopParts abi p).start (SourceParts.loopParts abi p).stop
        (SourceParts.loopParts abi p).stride (SourceParts.body abi p)] ++ SourceParts.suffix abi p := by
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases weight <;> cases output <;> cases starts <;> cases lengths <;> cases adapters <;> rfl

theorem accumulated_memory {α : Type} [Inhabited α] (M : Algebra α)
    (dot : SourceBody.DotImplementation α) (abi : SourceABI) (p : KernelArgs)
    (c : SourceBody.Coordinates) (s u : State α) (start : Nat)
    (initial : Values α .real [32,32]) (same : s.mem = u.mem) :
    SourceBody.accumulated M dot abi p c s start initial =
      SourceBody.accumulated M dot abi p c u start initial := by
  have ha (block : Nat) : SourceBody.aValues M abi p c s block = SourceBody.aValues M abi p c u block := by
    funext i
    simp [SourceBody.aValues,same]
  have hb (block : Nat) : SourceBody.bValues M abi p c s block = SourceBody.bValues M abi p c u block := by
    funext i
    simp [SourceBody.bValues,same]
  have hu : SourceBody.update M dot abi p c s = SourceBody.update M dot abi p c u := by
    funext block acc
    unfold SourceBody.update
    rw [ha,hb]
  unfold SourceBody.accumulated
  rw [hu]

def accumulated {α : Type} [Inhabited α] (M : Algebra α) (dot : SourceBody.DotImplementation α)
    (abi : SourceABI) (p : KernelArgs) (s : State α) (start adapter length : Nat) :=
  SourceBody.accumulated M dot abi p (SourceInit.coordinates p start adapter length (s.pids 0))
    s 0 (SourceInit.zero M)

theorem initial_memory {α : Type} [Inhabited α] (M : Algebra α) (p : KernelArgs)
    (s : State α) (start adapter length : Nat) :
    (SourceInit.state M p s start adapter length).mem = s.mem := by
  simp only [SourceInit.state,SourceSetup.prepared,SourceInit.guardedState,
    SourcePrelude.prepared,SourcePrelude.scalarBindings,List.foldl_cons,List.foldl_nil,
    State.setReg_mem]

theorem initial_undef {α : Type} [Inhabited α] (M : Algebra α) (p : KernelArgs)
    (s : State α) (start adapter length : Nat) :
    (SourceInit.state M p s start adapter length).undef = s.undef := by
  simp only [SourceInit.state,SourceSetup.prepared,SourceInit.guardedState,
    SourcePrelude.prepared,SourcePrelude.scalarBindings,List.foldl_cons,List.foldl_nil,
    State.setReg_undef]

theorem written_install {α : Type} [Inhabited α] (M : Algebra α)
    (dot : SourceBody.DotImplementation α) (abi : SourceABI) (p : KernelArgs)
    (c : SourceSuffix.Coordinates) (s : State α) (acc : Values α .real [32,32]) :
    SourceSuffix.written (SourceBody.install M dot) abi p c s acc =
      SourceSuffix.written M abi p c s acc := by
  simp only [SourceSuffix.written,SourceSuffix.outputValues,SourceSuffix.finish,
    SourceSuffix.castIfNeeded,SourceBody.install]

theorem ready_suffix {α : Type} [Inhabited α] (M : Algebra α) (p : KernelArgs)
    (s t : State α) (start adapter length : Nat) (acc : Values α .real [32,32])
    (hlength : (s.mem p.lengths p.batch).read .int = (length : Int))
    (h : SourceBody.Ready p (SourceInit.coordinates p start adapter length (s.pids 0))
      (SourceInit.state M p s start adapter length) t acc) :
    SourceSuffix.Ready p (outputCoordinates p start length (s.pids 0)) s t acc := by
  constructor
  case mem => exact h.mem.trans (initial_memory M p s start adapter length)
  case undef => exact h.undef.trans (initial_undef M p s start adapter length)
  case start => exact h.start
  case acc => exact h.acc
  case row =>
    simpa [SourceInit.state,SourceInit.guardedState,SourceSetup.prepared,SourcePrelude.prepared,
      SourcePrelude.scalarBindings,State.setReg,outputCoordinates] using
      h.preserved .int [] "pid_m" (by decide)
  case column =>
    simpa [SourceInit.state,SourceInit.guardedState,SourceSetup.prepared,SourcePrelude.prepared,
      SourcePrelude.scalarBindings,State.setReg,outputCoordinates] using
      h.preserved .int [] "pid_n" (by decide)
  case length =>
    simpa [SourceInit.state,SourceInit.guardedState,SourceSetup.prepared,SourcePrelude.prepared,
      SourcePrelude.scalarBindings,State.setReg,outputCoordinates,hlength] using
      h.preserved .int [] "M" (by decide)
  case columns =>
    simpa [SourceInit.state,SourceInit.guardedState,SourceSetup.prepared,SourcePrelude.prepared,
      SourcePrelude.scalarBindings,State.setReg] using h.preserved .int [] "N" (by decide)
  case batch =>
    simpa [SourceInit.state,SourceInit.guardedState,SourceSetup.prepared,SourcePrelude.prepared,
      SourcePrelude.scalarBindings,State.setReg] using h.preserved .int [] "cur_batch" (by decide)
  case rowStride =>
    simpa [SourceInit.state,SourceInit.guardedState,SourceSetup.prepared,SourcePrelude.prepared,
      SourcePrelude.scalarBindings,State.setReg] using h.preserved .int [] "cm_stride" (by decide)
  case columnStride =>
    simpa [SourceInit.state,SourceInit.guardedState,SourceSetup.prepared,SourcePrelude.prepared,
      SourcePrelude.scalarBindings,State.setReg] using h.preserved .int [] "cn_stride" (by decide)
  case slice =>
    simpa [SourceInit.state,SourceInit.guardedState,SourceSetup.prepared,SourcePrelude.prepared,
      SourcePrelude.scalarBindings,State.setReg] using h.preserved .int [] "slice_offset" (by decide)

theorem active_execution {α : Type} [Inhabited α] (M : Algebra α)
    (dot : SourceBody.DotImplementation α) (abi : SourceABI) (p : KernelArgs)
    (s : State α) (start adapter length : Nat)
    (hstart : (s.mem p.starts p.batch).read .int = (start : Int))
    (hlength : (s.mem p.lengths p.batch).read .int = (length : Int)) :
    ∃ final, run (SourceBody.install M dot) (SourceParts.active abi p)
        (SourceInit.guardedState p s adapter) = some final ∧
      final.mem = (SourceSuffix.written M abi p (outputCoordinates p start length (s.pids 0)) s
        (accumulated M dot abi p s start adapter length)).mem := by
  let initial := SourceInit.state M p s start adapter length
  have hp : run (SourceBody.install M dot) (SourceParts.activePrelude abi p)
      (SourceInit.guardedState p s adapter) = some initial :=
    SourceInit.prefix_run (SourceBody.install M dot) abi p s start adapter length hstart hlength
  obtain ⟨u,hu,hr⟩ := SourceBounds.loop_step M dot abi p
    (SourceInit.coordinates p start adapter length (s.pids 0)) initial initial (SourceInit.zero M)
    (SourceInit.ready M p s start adapter length)
  obtain ⟨final,hf,hm⟩ := SourceSuffix.suffix_run (SourceBody.install M dot) abi p
    (outputCoordinates p start length (s.pids 0)) s u _
    (ready_suffix M p s u start adapter length _ hlength hr) hlength
  refine ⟨final,?_,?_⟩
  · rw [active_split]
    simp only [run_append,hp,Option.bind_some,run,hu,hf,bind,Option.bind]
  · have ha := accumulated_memory M dot abi p
      (SourceInit.coordinates p start adapter length (s.pids 0)) initial s 0 (SourceInit.zero M) (initial_memory M p s start adapter length)
    rw [ha,written_install] at hm
    exact hm

theorem kernel_execution {α : Type} [Inhabited α] (M : Algebra α)
    (dot : SourceBody.DotImplementation α) (abi : SourceABI) (p : KernelArgs)
    (s : State α) (start adapter length : Nat)
    (hstart : (s.mem p.starts p.batch).read .int = (start : Int))
    (hlength : (s.mem p.lengths p.batch).read .int = (length : Int))
    (hadapter : (s.mem p.adapters p.batch).read .int = (adapter : Int))
    (live : 0 < length) (tile : s.pids 0 / tiles p.columns * 32 < length) :
    ∃ final, exec (SourceBody.install M dot) (kernel abi p) s = some final ∧
      final.mem = (SourceSuffix.written M abi p (outputCoordinates p start length (s.pids 0)) s
        (accumulated M dot abi p s start adapter length)).mem := by
  obtain ⟨final,ha,hm⟩ := active_execution M dot abi p s start adapter length hstart hlength
  refine ⟨final,?_,hm⟩
  change run (SourceBody.install M dot) (SourceParts.outer abi p) s = some final
  rw [SourceGuards.outer_split,run_append,SourcePrelude.run_prelude,Option.bind_some]
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
  simpa only [hl,ht,had,if_false] using ha

#axiomsClean active_split
#axiomsClean accumulated_memory
#axiomsClean ready_suffix
#axiomsClean active_execution
#axiomsClean kernel_execution
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceExecution
