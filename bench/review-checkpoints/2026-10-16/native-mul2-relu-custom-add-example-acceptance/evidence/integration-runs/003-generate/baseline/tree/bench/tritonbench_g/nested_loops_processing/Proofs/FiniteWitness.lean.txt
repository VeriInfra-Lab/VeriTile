import bench.tritonbench_g.nested_loops_processing.Proofs.Contracts
import bench.tritonbench_g.nested_loops_processing.Proofs.RealBoth

/-! Closed finite allocations for the complete source contracts. Every storage
profile uses its erased mathematical payload channel, with nonconstant input
cells. The int32 and fp64 instances additionally read back actual overwritten
cells and an untouched sentinel after flattened execution. -/
namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FiniteWitness
open VeriTile Triton TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000

def allocation : FlatAlloc := FlatAlloc.ofList "flat" [("src",64), ("dst",64)]

def inputCell (storage : Storage) (address : Nat) : MemCell :=
  match storage with
  | .int8 | .int16 | .int32 | .int64 => MemCell.of .int (address : Int)
  | .uint8 | .uint16 | .uint32 | .uint64 => MemCell.of .nat address
  | .boolean => MemCell.of .bool (address % 2 == 1)
  | _ => MemCell.real address

def sentinelCell (storage : Storage) : MemCell :=
  match storage with
  | .int8 | .int16 | .int32 | .int64 => MemCell.of .int (-1)
  | .uint8 | .uint16 | .uint32 | .uint64 => MemCell.of .nat 99
  | .boolean => MemCell.of .bool Bool.false
  | _ => MemCell.real (-1)

def initial (storage : Storage) : BlockState where
  mem := fun r n ↦ if r = "src" then inputCell storage n else sentinelCell storage
  regs := fun _ _ _ ↦ none
  pids := fun _ ↦ 0
  numPids := fun _ ↦ 1
  undef := fun _ _ ↦ 0

theorem disjoint : allocation.Disjoint := FlatAlloc.ofList_disjoint _ _ (by decide)

theorem closed : ∀ r, r ∉ allocation.regions → allocation.extent r = 0 :=
  FlatAlloc.ofList_closed _ _

/-- No bounds, separation, injectivity, successful execution or expected output
are assumptions: both complete variants satisfy the finite flat contract for
every one of the fifteen homogeneous payload profiles. -/
theorem source_contract (storage : Storage) (v : Bool) :
    Contracts.FlatCopy storage v allocation (initial storage) "src" "dst" 8 1 := by
  refine ⟨FullReal.contiguous_tile_injective, ?_, ?_⟩
  · cases v
    · exact PayloadReal.original_flat_io storage allocation disjoint closed (initial storage)
        "src" "dst" 8 1 (by decide) (by decide) (by decide) rfl
    · exact PayloadReal.candidate_flat_io storage allocation disjoint closed (initial storage)
        "src" "dst" 8 1 (by decide) (by decide) (by decide) rfl
  · cases v
    · exact PayloadReal.original_safe storage allocation.extent "src" "dst" 8 1
        (initial storage) (by decide) (by decide)
    · exact PayloadReal.candidate_safe storage allocation.extent "src" "dst" 8 1
        (initial storage) (by decide) (by decide)

/-- The preserved full int32 helper also instantiates the same finite allocation
for both variants, including its complete raw-cell frame. -/
theorem integer_io (v : Bool) :
    FullReal.FlatCopyContract allocation
      (RealCandidate.algorithm .int32 v "src" "dst" 8 1)
      (initial .int32) "src" "dst" 8 1 := by
  cases v
  · exact FullReal.source_flat_io allocation disjoint closed (initial .int32)
      "src" "dst" 8 1 (by decide) (by decide) (by decide) rfl
  · exact FullReal.candidate_flat_io allocation disjoint closed (initial .int32)
      "src" "dst" 8 1 (by decide) (by decide) (by decide) rfl

/-- Finite flattened int32 execution retains the last writer and the sentinel;
the full frame is retained alongside these concrete readbacks. -/
theorem integer_cells (v : Bool) :
    ∃ final, exec (allocation.flattenKernel
      (RealCandidate.algorithm .int32 v "src" "dst" 8 1))
      (allocation.flattenState (initial .int32)) = some final ∧
    BlockState.WriteWithin
      (FullReal.flatFootprint allocation (FullReal.outputFootprint "dst" 8 1))
      (allocation.flattenState (initial .int32)) final ∧
    final.mem allocation.flat (allocation.addr "dst" 8) = MemCell.of .int 2 ∧
    final.mem allocation.flat (allocation.addr "dst" 47) = MemCell.of .int 27 ∧
    final.mem allocation.flat (allocation.addr "dst" 55) = MemCell.of .int 35 ∧
    final.mem allocation.flat (allocation.addr "dst" 56) = MemCell.of .int (-1) := by
  obtain ⟨final, he, hf, hv⟩ := integer_io v
  refine ⟨final, he, hf, ?_, ?_, ?_, ?_⟩
  · exact hv 8 (by decide)
  · exact hv 47 (by decide)
  · exact hv 55 (by decide)
  · exact hv 56 (by decide)

/-- The fp64 source's erased real contract has the same concrete nonempty
flattened result, without replacing floating payloads by integer cells. -/
theorem floating_cells (v : Bool) :
    ∃ final, exec (allocation.flattenKernel
      (RealCandidate.algorithm .fp64 v "src" "dst" 8 1))
      (allocation.flattenState (initial .fp64)) = some final ∧
    BlockState.WriteWithin
      (FullReal.flatFootprint allocation (FullReal.outputFootprint "dst" 8 1))
      (allocation.flattenState (initial .fp64)) final ∧
    final.mem allocation.flat (allocation.addr "dst" 8) = MemCell.real 2 ∧
    final.mem allocation.flat (allocation.addr "dst" 47) = MemCell.real 27 ∧
    final.mem allocation.flat (allocation.addr "dst" 55) = MemCell.real 35 ∧
    final.mem allocation.flat (allocation.addr "dst" 56) = MemCell.real (-1) := by
  obtain ⟨_, ⟨final, he, hf, hv⟩, _⟩ := source_contract .fp64 v
  refine ⟨final, he, hf, ?_, ?_, ?_, ?_⟩
  · exact hv 8 (by decide)
  · exact hv 47 (by decide)
  · exact hv 55 (by decide)
  · exact hv 56 (by decide)

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FiniteWitness

#guard_msgs (drop info) in
#auditModuleAxioms
