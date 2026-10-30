import bench.tritonbench_g.triton_argmax.Proofs.SegmentedGrid
import bench.tritonbench_g.triton_argmax.Proofs.DimMemory

/-! Grouping into legal launches preserves the complete logical program
sequence. This reuses prior per-program FP32/reduction/cast/memory execution;
it does not replace the remaining native compilation/ABI attachment. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.SegmentedExecution
open Source Memory SegmentedGrid

def schedule (start total : Nat) : List Nat :=
  if total=0 then [] else
    List.range' start (min total limit) ++ schedule (start+limit) (total-limit)
termination_by total
decreasing_by
  simp_wf
  norm_num [limit]
  omega

theorem schedule_range (start total : Nat) : schedule start total = List.range' start total := by
  induction total using Nat.strong_induction_on generalizing start with
  | h total ih =>
    by_cases empty : total=0
    · simp [schedule,empty]
    · have smaller : total-limit < total := by norm_num [limit] at *; omega
      rw [schedule,if_neg empty,ih (total-limit) smaller]
      by_cases short : total ≤ limit
      · simp [Nat.min_eq_left short,Nat.sub_eq_zero_of_le short]
      · have long : limit ≤ total := by omega
        rw [Nat.min_eq_right long,List.range'_append_1,Nat.add_sub_of_le long]

theorem schedule_zero (total : Nat) : schedule 0 total = List.range total := by
  rw [schedule_range,List.range_eq_range']

def firstGrid {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : GlobalMemory.Buffers) (s : M) : M :=
  (schedule 0 c.blocks).foldl (GlobalMemory.firstLive mem ops c p) s

theorem firstGrid_eq {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : GlobalMemory.Buffers) (s : M) :
    firstGrid mem ops c p s = GlobalMemory.firstGrid mem ops c p s := by
  simp only [firstGrid,GlobalMemory.firstGrid,schedule_zero]

def pairs (rows k : Nat) : List (Nat × Nat) :=
  (List.range (rows*k)).map fun p => (p/k,p%k)

theorem pairs_eq (rows k : Nat) :
    pairs rows k = (List.range rows).flatMap fun row =>
      (List.range k).map fun inner => (row,inner) := by
  induction rows with
  | zero => simp [pairs]
  | succ rows ih =>
    simp only [pairs,Nat.succ_mul,List.range_add,List.map_append,List.map_map]
    rw [show (List.range (rows*k)).map (fun p => (p/k,p%k))=pairs rows k from rfl,ih]
    rw [List.range_succ,List.flatMap_append]
    simp only [List.range_zero,List.nil_append,List.map_cons,List.map_nil,Nat.add_zero,
      List.flatMap_cons,List.flatMap_nil,List.append_nil]
    congr 1
    apply List.map_congr_left
    intro inner member
    have live := List.mem_range.mp member
    have inv := dimension_inverse (rows+1) k rows inner (by omega) live
    exact Prod.ext inv.2.1 inv.2.2

def dimJobs (c : DimConfig) : List DimMemory.Job :=
  (schedule 0 (((c.outer+127)/128)*c.inner)).flatMap fun p =>
    (List.range 128).map fun lane => ((p/c.inner)*128+lane,p%c.inner)

theorem dimJobs_eq (c : DimConfig) : dimJobs c = DimMemory.jobs c := by
  rw [dimJobs,schedule_zero]
  have paired := pairs_eq ((c.outer+127)/128) c.inner
  have lifted := congrArg (fun ps : List (Nat × Nat) => ps.flatMap fun pair =>
    (List.range 128).map fun lane => (pair.1*128+lane,pair.2)) paired
  simpa only [pairs,DimMemory.jobs,List.flatMap_map,List.flatMap_assoc] using lifted

def dimension {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : DimConfig) (p : DimMemory.Buffers) (s : M) : M :=
  (dimJobs c).foldl (DimMemory.liveRow mem ops c p) s

theorem dimension_eq {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : DimConfig) (p : DimMemory.Buffers) (s : M) :
    dimension mem ops c p s = DimMemory.run mem ops c p s := by
  simp only [dimension,DimMemory.run,dimJobs_eq]

#axiomsClean schedule_range
#axiomsClean schedule_zero
#axiomsClean firstGrid_eq
#axiomsClean pairs_eq
#axiomsClean dimJobs_eq
#axiomsClean dimension_eq
end VeriTile.Bench.Repairs.TritonArgmax.SegmentedExecution
