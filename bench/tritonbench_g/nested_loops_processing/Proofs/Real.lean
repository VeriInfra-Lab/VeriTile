import bench.tritonbench_g.nested_loops_processing.Proofs.Math

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing
open VeriTile.Bench.TritonBenchG.NestedLoopsProcessing
noncomputable section
open scoped Classical

theorem map_update (f : α → β) (memory : Nat → α) (address : Nat) (value : α) :
    (fun i=>f (Function.update memory address value i))=
      Function.update (fun i=>f (memory i)) address (f value) := by
  funext i
  by_cases same:i=address
  · subst i;simp
  · simp [Function.update_of_ne same]

theorem map_copy_tile (f : α → β) (input memory : Nat → α) (sm sn : Int) (t : Nat) :
    (fun i=>f (CopyMath.copyTile input sm sn memory t i))=
      CopyMath.copyTile (fun i=>f (input i)) sm sn (fun i=>f (memory i)) t := by
  simp only [CopyMath.copyTile,List.range_succ,List.range_zero,List.nil_append,
    List.foldl_append,List.foldl_nil,List.foldl_cons]
  simp only [map_update]

theorem map_expected (f : α → β) (input memory : Nat → α) (sm sn : Int) :
    (fun i=>f (CopyMath.expected input memory sm sn i))=
      CopyMath.expected (fun i=>f (input i)) (fun i=>f (memory i)) sm sn := by
  have fold : ∀ ts : List Nat,∀ memory : Nat → α,
      (fun i=>f (ts.foldl (CopyMath.copyTile input sm sn) memory i))=
        ts.foldl (CopyMath.copyTile (fun i=>f (input i)) sm sn) (fun i=>f (memory i)) := by
    intro ts
    induction ts with
    | nil=>intro;rfl
    | cons t ts ih=>
      intro memory
      simp only [List.foldl_cons,ih,map_copy_tile]
  exact fold (List.range 24) memory

theorem map_target (f : α → β) (zero : α) (count cols : Nat) (input memory : Nat → α) :
    (fun i=>f (target zero count cols input memory i))=
      target (f zero) count cols (fun i=>f (input i)) (fun i=>f (memory i)) := by
  unfold target
  rw [map_expected]
  congr 1
  funext i
  by_cases bound:i<count <;> simp [zeroExtend,bound]

/-- Exact signed-int32 interpretation: the program performs only copies.
There are no floating-point arithmetic ports or deferred numerical facts. -/
def realTarget (rows cols : Nat) : Nat → ℝ :=
  target 0 (rows*cols) cols arangeReal (fun _=>0)

theorem int32_real_exact (rows cols i : Nat) :
    (signedValue (target (0:Word) (rows*cols) cols arangeWord (fun _=>0) i):ℝ)=
      realTarget rows cols i := by
  have h:=congrFun (map_target (fun word:Word=>(signedValue word:ℝ)) 0
    (rows*cols) cols arangeWord (fun _=>0)) i
  simpa [realTarget,arangeReal,signedValue] using h

#axiomsClean map_expected
#axiomsClean map_target
#axiomsClean int32_real_exact
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing
