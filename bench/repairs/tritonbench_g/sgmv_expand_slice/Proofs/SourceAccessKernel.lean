import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceAccessPrefix
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceAccessLoop
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceAccessSuffix
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceTile

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceAccess
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

theorem active_safe {α : Type} [Inhabited α] (M : Algebra α)
    (dot : SourceBody.DotImplementation α) (abi : SourceABI)
    (hw : abi.weight=.half) (ho : abi.output=.single) (p : KernelArgs)
    (s : State α) (start adapter length : Nat)
    (hstart : (s.mem p.starts p.batch).read .int = (start : Int))
    (hlength : (s.mem p.lengths p.batch).read .int = (length : Int))
    (bounds : RegionName → Nat) (bs : p.batch < bounds p.starts)
    (bl : p.batch < bounds p.lengths)
    (body : ∀ block, block < (p.inner+15)/16 → BodyBounds bounds p
      (SourceInit.coordinates p start adapter length (s.pids 0)) block)
    (output : ∀ i, SourceSuffix.active p (SourceExecution.outputCoordinates p start length (s.pids 0)) i →
      SourceSuffix.offset p (SourceExecution.outputCoordinates p start length (s.pids 0)) i < bounds p.output) :
    Safety.traceSafeList (SourceBody.install M dot) bounds (SourceParts.active abi p)
      (SourceInit.guardedState p s adapter) := by
  let initial := SourceInit.state M p s start adapter length
  have prefixRun := SourceInit.prefix_run (SourceBody.install M dot) abi p s start adapter length hstart hlength
  have ready := SourceInit.ready M p s start adapter length
  have runLoop := SourceBounds.loop_step M dot abi p
    (SourceInit.coordinates p start adapter length (s.pids 0)) initial initial (SourceInit.zero M) ready
  rw [SourceExecution.active_split,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨active_prefix_safe (SourceBody.install M dot) abi p s adapter bounds bs,?_⟩
  intro u hu
  have prefixExec : FP.Structural.run (SourceBody.install M dot) (SourceParts.activePrelude abi p)
      (SourceInit.guardedState p s adapter) = some initial := prefixRun
  rw [prefixExec] at hu
  cases hu
  simp only [List.singleton_append]
  rw [Safety.traceSafeList_cons]
  refine ⟨loop_step_safe M dot abi p _ initial initial _ ready bounds body,?_⟩
  intro v hv
  obtain ⟨last,executed,lastReady⟩ := runLoop
  rw [executed] at hv
  have same := Option.some.inj hv
  subst v
  exact suffix_safe (SourceBody.install M dot) abi hw ho p
    (SourceExecution.outputCoordinates p start length (s.pids 0)) s last _
    (SourceExecution.ready_suffix M p s last start adapter length _ hlength lastReady)
    hlength bounds bl output

theorem kernel_safe {α : Type} [Inhabited α] (M : Algebra α)
    (dot : SourceBody.DotImplementation α) (abi : SourceABI)
    (hw : abi.weight=.half) (ho : abi.output=.single) (p : KernelArgs)
    (s : State α) (start adapter length : Nat)
    (hstart : (s.mem p.starts p.batch).read .int = (start : Int))
    (hlength : (s.mem p.lengths p.batch).read .int = (length : Int))
    (hadapter : (s.mem p.adapters p.batch).read .int = (adapter : Int))
    (live : 0 < length) (tile : s.pids 0 / tiles p.columns * 32 < length)
    (bounds : RegionName → Nat) (bs : p.batch < bounds p.starts)
    (bl : p.batch < bounds p.lengths) (ba : p.batch < bounds p.adapters)
    (body : ∀ block, block < (p.inner+15)/16 → BodyBounds bounds p
      (SourceInit.coordinates p start adapter length (s.pids 0)) block)
    (output : ∀ i, SourceSuffix.active p (SourceExecution.outputCoordinates p start length (s.pids 0)) i →
      SourceSuffix.offset p (SourceExecution.outputCoordinates p start length (s.pids 0)) i < bounds p.output) :
    FP.Structural.TraceSafe (SourceBody.install M dot) bounds (kernel abi p) s := by
  change Safety.traceSafeList _ bounds (SourceParts.outer abi p) s
  rw [SourceGuards.outer_split,Safety.traceSafeList_append]
  refine ⟨prelude_safe (SourceBody.install M dot) abi p s bounds bl,?_⟩
  intro t ht
  rw [SourcePrelude.run_prelude] at ht
  cases ht
  apply guarded_safe (SourceBody.install M dot) abi p _ _ length adapter (s.pids 0 / tiles p.columns)
  · simp [SourcePrelude.prepared,SourcePrelude.scalarBindings,State.setReg,hlength]
  · simp [SourcePrelude.prepared,SourcePrelude.scalarBindings,State.setReg]
  · simp [SourcePrelude.prepared,SourcePrelude.scalarBindings,State.setReg]
  · simpa [SourcePrelude.prepared,SourcePrelude.scalarBindings,State.setReg] using hadapter
  · exact live
  · exact tile
  · exact ba
  · exact active_safe M dot abi hw ho p s start adapter length hstart hlength bounds bs bl body output

#axiomsClean active_safe
#axiomsClean kernel_safe
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceAccess
