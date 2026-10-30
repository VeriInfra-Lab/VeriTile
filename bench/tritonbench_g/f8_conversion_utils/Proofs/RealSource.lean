import bench.tritonbench_g.f8_conversion_utils.Proofs.Mathematical_Real
import bench.tritonbench_g.f8_conversion_utils.Proofs.Cells
import bench.tritonbench_g.f8_conversion_utils.Proofs.Lowering

namespace VeriTile.Bench.Repairs.F8ConversionUtils.RealSource
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.F8ConversionUtils
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

def kernel (direction : Direction) (input : TypedSource.Input) (format : TypedSource.Format)
    (Y X : RegionName) (N B : Nat) : ComputeKernel :=
  match direction with
  | .decode => TypedSource.toHalf format Bool.false Y X N B
  | .encode => TypedSource.toFP8 input format Bool.false Y X N B

theorem body_eq (direction : Direction) (input : TypedSource.Input) (format : TypedSource.Format)
    (Y X : RegionName) (N B : Nat) :
    (kernel direction input format Y X N B).eraseDType.toAlgKernel.body=
      (RealProof.original direction.doubleStore Y X N B).toAlgKernel.body := by
  cases direction
  · exact Erasure.to_half_body format Bool.false Y X N B
  · exact Erasure.to_fp8_body input format Bool.false Y X N B

/-- Both E4/E5 formats and both encode input precisions erase to the
independent Real identity. The complete original body executes, including both
decode stores; arbitrary Real inputs have no finite-byte encoding premise. -/
theorem executes (direction : Direction) (input : TypedSource.Input) (format : TypedSource.Format)
    (Y X : RegionName) (N B : Nat) (s : BlockState) :
    ∃ final,exec (kernel direction input format Y X N B).eraseDType.toAlgKernel s=some final ∧
      (∀ j : Fin B,s.pid*B+j.val<N → final.readMem Y (s.pid*B+j.val)=s.readMem X (s.pid*B+j.val)) ∧
      (∀ r i,(r≠Y ∨ ∀ j : Fin B,s.pid*B+j.val<N → i≠s.pid*B+j.val) → final.mem r i=s.mem r i) := by
  obtain ⟨final,executed,readback,frame⟩ := RealProof.run direction.doubleStore Y X N B s
  refine ⟨final,?_,readback,frame⟩
  simpa only [exec,body_eq] using executed

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

/-- The same successful erased-source state refines the same full native
program in independent Real memory, on all active and inactive output cells. -/
theorem program_refines (direction : Direction) (input : TypedSource.Input) (format : TypedSource.Format)
    (x : Native.Allocation n iw) (y : Native.Allocation n ow)
    (start count pid : Nat) (small : count<2^31) (launched : pid<NativeIndex.grid count)
    (globalActive : ∀ i,i<count → start+i<n) (original : Cells.Heap)
    (Y X : RegionName) (s : BlockState) (pidBound : s.pid=pid)
    (inputs : ∀ i,s.readMem X i=original (Memory.address x.base iw (start+i)))
    (outputs : ∀ i,s.readMem Y i=original (Memory.address y.base ow (start+i))) :
    ∃ final,exec (kernel direction input format Y X count 1024).eraseDType.toAlgKernel s=some final ∧
      (∀ i,final.readMem Y i=
        Lowering.program (Cells.primitives iw ow x.positive y.positive) direction.doubleStore id
          (x.base+start*iw) (y.base+start*ow) count pid original (Memory.address y.base ow (start+i))) ∧
      (∀ r i,r≠Y → final.mem r i=s.mem r i) := by
  obtain ⟨final,executed,readback,frame⟩ := executes direction input format Y X count 1024 s
  refine ⟨final,executed,?_,fun r i different => frame r i (Or.inl different)⟩
  intro i
  rw [Lowering.program_eq _ direction.doubleStore id x y start count pid small launched globalActive]
  change final.readMem Y i=(Cells.primitives iw ow x.positive y.positive).readOutput
    (Memory.stores (Cells.primitives iw ow x.positive y.positive) y.base
      (fun index => original (Memory.address x.base iw index))
      (Memory.writes direction.doubleStore (Lowering.jobs start count pid)) original)
    (Memory.address y.base ow (start+i))
  rw [Memory.stores_value]
  by_cases member : start+i∈Lowering.jobs start count pid
  · rw [if_pos ((Memory.mem_writes direction.doubleStore _ _).mpr member)]
    obtain ⟨j,rfl,active⟩ := (job_member start count pid i small launched).mp member
    have sourceValue := readback j (by simpa only [pidBound] using active)
    rw [pidBound] at sourceValue
    exact sourceValue.trans (inputs _)
  · rw [if_neg (fun h => member ((Memory.mem_writes direction.doubleStore _ _).mp h))]
    have unchanged := frame Y i (Or.inr (by
      intro j active equal
      apply member
      apply (job_member start count pid i small launched).mpr
      exact ⟨j,by simpa only [pidBound] using equal,by rw [equal];exact active⟩))
    unfold BlockState.readMem
    rw [unchanged]
    exact outputs i

#axiomsClean executes
#axiomsClean program_refines
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.RealSource
