import bench.tritonbench_g.mean_reduction.Proofs.LogicalResult
import bench.tritonbench_g.mean_reduction.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.MeanReduction.MeanIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000

theorem input_reads {α : Type} [Inhabited α] (family : Nat → Algebra α) (variant : Bool)
    (M N : Nat) (jobs : List Nat) (values : Inputs α M N) (s : State α)
    (bound : (nativeIO family variant M N jobs).InputBinding values s) : Reads values s.mem := by
  intro i
  exact bound ⟨0,by simp [nativeIO,memory,inputChannels]⟩ (i,PUnit.unit) trivial

theorem input_bound {α : Type} (family : Nat → Algebra α) (variant : Bool)
    (M N : Nat) (jobs : List Nat) (values : Inputs α M N) (bounds : RegionName → Nat)
    (backed : (nativeIO family variant M N jobs).InBounds bounds values) : M*N≤bounds "X" := by
  by_contra hn
  have bad := backed.2.1 ⟨0,by simp [nativeIO,memory,inputChannels]⟩
    (⟨bounds "X",Nat.lt_of_not_ge hn⟩,PUnit.unit) trivial
  exact Nat.lt_irrefl (bounds "X") bad

theorem output_bound {α : Type} (family : Nat → Algebra α) (variant : Bool)
    (M N : Nat) (jobs : List Nat) (values : Inputs α M N) (bounds : RegionName → Nat)
    (backed : (nativeIO family variant M N jobs).InBounds bounds values) : M≤bounds "Mean" := by
  by_contra hn
  have bad := backed.2.2.1 ⟨0,by simp [nativeIO,memory,outputChannels]⟩
    (⟨bounds "Mean",Nat.lt_of_not_ge hn⟩,PUnit.unit) trivial
  exact Nat.lt_irrefl (bounds "Mean") bad

theorem schedule_safe {α : Type} [Inhabited α] (family : Nat → Algebra α) (variant : Bool)
    (M N : Nat) (jobs : List Nat) (bounds : RegionName → Nat)
    (input : M*N≤bounds "X") (output : M≤bounds "Mean") (s : State α) :
    ScheduleSafe bounds (jobs.map (invocation family variant M N)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons pid jobs ih =>
    refine ⟨?_,fun t _ => ih t⟩
    exact Access.variant_safe (family pid) variant M N
      ((invocation family variant M N pid).initial s) bounds input output

/-- Complete finite IO for either original source, retaining arbitrary
per-program primitive models and arbitrary complete launch permutation. -/
theorem native_implements {α : Type} [Inhabited α] (family : Nat → Algebra α) (variant : Bool)
    (M N : Nat) (jobs : List Nat) (order : jobs.Perm (List.range (gridSize M))) :
    (nativeIO family variant M N jobs).Implements (nativeTarget family M N) := by
  intro values _ bounds backed s bound
  obtain ⟨t,ran,result,frame⟩ := complete_result family variant M N jobs order values s
    (input_reads family variant M N jobs values s bound)
  refine ⟨schedule_safe family variant M N jobs bounds
    (input_bound family variant M N jobs values bounds backed)
    (output_bound family variant M N jobs values bounds backed) s,t,ran,?_,?_⟩
  · intro o j _
    fin_cases o
    change mapCell id (t.mem "Mean" j.1.val)=Cell.mk .real (nativeRow family M N values j.1)
    rw [result j.1]
    rfl
  · intro r offset outside
    apply frame r offset
    by_cases hr : r="Mean"
    · right
      by_contra hn
      have missed := outside.1 ⟨0,by simp [nativeIO,memory,outputChannels]⟩
        (⟨offset,Nat.lt_of_not_ge hn⟩,PUnit.unit) trivial
      exact missed.elim (fun ne => ne hr) (fun ne => ne rfl)
    · exact Or.inl hr

theorem native_equivalent {α : Type} [Inhabited α] (family : Nat → Algebra α)
    (M N : Nat) (jobs : List Nat) (order : jobs.Perm (List.range (gridSize M))) :
    (nativeIO family Bool.false M N jobs).Equivalent (candidate family M N jobs) :=
  ProgramIO.EquivalentOn.of_implementsOn
    (native_implements family Bool.false M N jobs order)
    (native_implements family Bool.true M N jobs order)

#axiomsClean native_implements
#axiomsClean native_equivalent
end
end VeriTile.Bench.Repairs.MeanReduction.MeanIO
