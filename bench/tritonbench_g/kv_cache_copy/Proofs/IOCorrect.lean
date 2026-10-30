import bench.tritonbench_g.kv_cache_copy.IO
import bench.tritonbench_g.kv_cache_copy.Proofs.KernelValues
import bench.tritonbench_g.kv_cache_copy.Proofs.KernelSafety

namespace VeriTile.Bench.Repairs.KvCacheCopy.IOCorrect
open VeriTile.Triton FP.Structural Source KernelFP
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

/-- Assemble public metadata/payload bindings with the original complete
source execution, its actual trace safety, and its initial-value copy result. -/
theorem correct (rest : Algebra ℝ) (p : Parameters) (seqPid headPid : Nat) :
    (CopyIO.io rest p seqPid headPid).ImplementsOn
      (CopyIO.domain rest p seqPid headPid) (CopyIO.reference rest p seqPid headPid) := by
  intro vals unique bounds bounded initial binding
  let length : Int := vals (0 : Fin 4) ()
  let block : Int := vals (1 : Fin 4) ()
  let s : State ℝ := { initial with pids := CopyIO.ids seqPid headPid }
  have hposition : position s = length - 1 := by
    unfold position
    change (initial.mem "lengths" seqPid).read .int - 1 = length - 1
    exact congrArg (fun x : Int => x - 1) (binding (0 : Fin 4) () trivial)
  have hblock : selectedBlock p s = block := by
    unfold selectedBlock
    rw [hposition]
    change (initial.mem "table" (CopyIO.tableAddress p seqPid length)).read .int = block
    exact binding (1 : Fin 4) () trivial
  have hsource (key : Bool) (q : Nat) (i : TileIndex [p.width]) :
      sourceOffset p s.pids key q i = CopyIO.inputAddress p seqPid headPid key q i := by
    cases key <;> simp [sourceOffset, CopyIO.inputAddress, s, CopyIO.ids]
  have htarget (key : Bool) (q : Nat) (i : TileIndex [p.width]) :
      (targetInt p s key q i).toNat = CopyIO.outputAddress p headPid length block key q i := by
    unfold targetInt
    rw [hposition, hblock]
    cases key <;> simp [CopyIO.outputAddress, s, CopyIO.ids]
  have rawUnique (key : Bool) (u v : Nat) (hu : u < p.dims/p.width) (hv : v < p.dims/p.width)
      (i j : TileIndex [p.width])
      (eq : (targetInt p s key u i).toNat = (targetInt p s key v j).toNat) : u = v ∧ i = j := by
    apply unique key u v hu hv i j
    simpa only [htarget] using eq
  constructor
  · change ScheduleSafe bounds [_] initial
    rw [scheduleSafe_singleton]
    change FP.Structural.TraceSafe (CopyIO.realBackend rest) bounds (originalKernel p) s
    apply KernelSafety.trace (CopyIO.realBackend rest) p bounds s
    · exact bounded.2.1 (0 : Fin 4) () trivial
    · rw [hposition]
      exact bounded.2.1 (1 : Fin 4) () trivial
    · intro q hq key i
      rw [hsource]
      cases key
      · exact bounded.2.1 (3 : Fin 4) (⟨q, hq⟩, i) trivial
      · exact bounded.2.1 (2 : Fin 4) (⟨q, hq⟩, i) trivial
    · intro q hq key i
      rw [htarget]
      cases key
      · exact bounded.2.2.1 (1 : Fin 2) (⟨q, hq⟩, i) trivial
      · exact bounded.2.2.1 (0 : Fin 2) (⟨q, hq⟩, i) trivial
  · obtain ⟨final, run, memory⟩ := KernelFP.execute (CopyIO.realBackend rest) Bool.false p s
    have values (key : Bool) (q : Fin (p.dims/p.width)) (i : TileIndex [p.width]) :
        final.mem (outputRegion key) (CopyIO.outputAddress p headPid length block key q.val i) =
          .mk .real (if key then vals (2 : Fin 4) (q, i) else vals (3 : Fin 4) (q, i)) := by
      rw [memory, ← htarget]
      rw [copied_readback (CopyIO.realBackend rest) p s (p.dims/p.width) key q.val q.isLt i (rawUnique key)]
      rw [hsource]
      cases key
      · exact congrArg (Cell.mk .real) (binding (3 : Fin 4) (q, i) trivial)
      · exact congrArg (Cell.mk .real) (binding (2 : Fin 4) (q, i) trivial)
    refine ⟨final, ?_, ?_, ?_⟩
    · simpa only [CopyIO.io, runSchedule_singleton, Invocation.run, Invocation.initial,
        bindRegisters, List.foldl_nil, Option.getD_none, s] using run
    · intro port i _
      fin_cases port
      · change mapCell id (final.mem "keyCache"
          (CopyIO.outputAddress p headPid length block Bool.true i.1.val i.2)) = _
        have value := values Bool.true i.1 i.2
        simp only [outputRegion, ite_true] at value
        rw [value]
        rfl
      · change mapCell id (final.mem "valueCache"
          (CopyIO.outputAddress p headPid length block Bool.false i.1.val i.2)) = _
        have value := values Bool.false i.1 i.2
        simp only [outputRegion, Bool.false_eq_true, ite_false] at value
        rw [value]
        rfl
    · intro r o outside
      rw [memory]
      apply copied_frame
      intro key q hq
      by_cases different : r ≠ outputRegion key
      · exact Or.inl different
      · right
        intro i
        rw [htarget]
        cases key
        · exact (outside.1 (1 : Fin 2) (⟨q, hq⟩, i) trivial).resolve_left different
        · exact (outside.1 (0 : Fin 2) (⟨q, hq⟩, i) trivial).resolve_left different

#axiomsClean correct
end
end VeriTile.Bench.Repairs.KvCacheCopy.IOCorrect
