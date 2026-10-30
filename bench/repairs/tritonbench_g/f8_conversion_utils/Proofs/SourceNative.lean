import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.SourceFrame
import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.SourceBridge
import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.Lowering

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Source
open VeriTile.Triton FP.Structural Bytes
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

/-- Only the output storage representation has a same-width read/write
roundtrip. This does not require a reversible input-to-output conversion. -/
structure OutputStorage {α : Type} (p : Profile) (storeWord : α → Word p.outputWidth) where
  load : Word p.outputWidth → α
  roundtrip : ∀ word,storeWord (load word)=word

private theorem job_member (start count pid i : Nat) (small : count<2^31)
    (launched : pid<NativeIndex.grid count) :
    start+i∈Lowering.jobs start count pid ↔ ∃ j : Fin 1024,i=pid*1024+j.val ∧ i<count := by
  rw [Lowering.jobs_eq start count pid small launched]
  simp only [List.mem_map,programLanes,List.mem_filter,decide_eq_true_eq]
  constructor
  · rintro ⟨index,⟨⟨lane,member,equal⟩,active⟩,shifted⟩
    exact ⟨⟨lane,List.mem_range.mp member⟩,by dsimp only;omega,by omega⟩
  · rintro ⟨j,equal,active⟩
    exact ⟨i,⟨⟨j.val,List.mem_range.mpr j.isLt,equal.symm⟩,active⟩,rfl⟩

/-- Whole-state refinement of the original structural execution to the literal
finite native program. Every output cell is related, including inactive cells;
other typed regions retain their original cells. This relation connects the
same successful source run to the actual native transition, rather than
postulating a kernel-output conversion. -/
theorem program_refines {α : Type} [Inhabited α] (M : Algebra α) (p : Profile)
    (physical : Word p.inputWidth → Word p.outputWidth) (binding : ScalarBinding M p physical)
    (x : Native.Allocation n p.inputWidth) (y : Native.Allocation n p.outputWidth)
    (start count pid : Nat) (small : count<2^31) (launched : pid<NativeIndex.grid count)
    (globalActive : ∀ i,i<count → start+i<n) (original : Heap)
    (Y X : RegionName) (s : State α) (pidBound : s.pids 0=pid)
    (inputs : ∀ i,s.mem X i=inputCell p
      (binding.loadWord (read p.inputWidth original (Memory.address x.base p.inputWidth (start+i)))))
    (outputs : ∀ i,binding.storeWord (outputValue p (s.mem Y i))=
      read p.outputWidth original (Memory.address y.base p.outputWidth (start+i))) :
    ∃ final,FP.Structural.exec (M.withDefaultPrecision .fp32) (kernel p Y X count 1024) s=some final ∧
      (∀ i,binding.storeWord (outputValue p (final.mem Y i))=
        read p.outputWidth
          (Lowering.program (Bytes.primitives p.inputWidth p.outputWidth) p.twice physical
            (x.base+start*p.inputWidth) (y.base+start*p.outputWidth) count pid original)
          (Memory.address y.base p.outputWidth (start+i))) ∧
      (∀ r i,r≠Y → final.mem r i=s.mem r i) := by
  obtain ⟨final,executed,sourceMemory⟩ := executes (M.withDefaultPrecision .fp32) p Y X count 1024 s
  refine ⟨final,executed,?_,?_⟩
  · intro i
    rw [Lowering.program_eq _ p.twice physical x y start count pid small launched globalActive]
    change binding.storeWord (outputValue p (final.mem Y i))=
      (Bytes.primitives p.inputWidth p.outputWidth).readOutput
        (Memory.stores (Bytes.primitives p.inputWidth p.outputWidth) y.base
          (fun index => physical (read p.inputWidth original (Memory.address x.base p.inputWidth index)))
          (Memory.writes p.twice (Lowering.jobs start count pid)) original)
        (Memory.address y.base p.outputWidth (start+i))
    rw [Memory.stores_value]
    by_cases member : start+i∈Lowering.jobs start count pid
    · rw [if_pos ((Memory.mem_writes p.twice _ _).mpr member)]
      obtain ⟨j,rfl,active⟩ := (job_member start count pid i small launched).mp member
      rw [sourceMemory]
      have sourceValue := memory_readback (M.withDefaultPrecision .fp32) p Y X count 1024 s j
        (by simpa only [pidBound] using active)
      rw [pidBound] at sourceValue
      rw [sourceValue,inputs]
      have scalar := binding.conversion
        (read p.inputWidth original (Memory.address x.base p.inputWidth (start+(pid*1024+j.val))))
      cases p <;> simpa [outputValue,inputCell,Profile.input,Profile.output,ofFloat,toFloat,Cell.read] using scalar
    · rw [if_neg (fun h => member ((Memory.mem_writes p.twice _ _).mp h)),sourceMemory,
        memory_frame (M.withDefaultPrecision .fp32) p Y X count 1024 s Y i (Or.inr ?_)]
      · exact outputs i
      · intro j active equal
        apply member
        apply (job_member start count pid i small launched).mpr
        exact ⟨j,by simpa only [pidBound] using equal,by rw [equal];exact active⟩
  · intro r i different
    rw [sourceMemory]
    exact memory_frame (M.withDefaultPrecision .fp32) p Y X count 1024 s r i (Or.inl different)

#axiomsClean program_refines
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Source
