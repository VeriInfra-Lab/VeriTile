import bench.repairs.tritonbench_g.pow_scalar_tensor.IO
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.KernelSafety

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.Interface
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.PowScalarTensor WholeTile
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
noncomputable section
variable {α : Type} [Inhabited α]

theorem schedule_run (M : Algebra α) (base : ℝ) (candidate : Bool)
    (input output : SignedView.View) (n : Nat) (pids : List Nat) (s : State α) :
    runSchedule (pids.map (invocation M base candidate input output n)) s=
      StructuralControl.runPrograms M (sourceKernel (parameters base (KernelInterface.arguments input output n)) candidate).surfaceBody
        (KernelInterface.arguments input output n).C pids s := by
  induction pids generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    change (run M (sourceKernel (parameters base (KernelInterface.arguments input output n)) candidate).surfaceBody
      (StructuralControl.atProgram (KernelInterface.arguments input output n).C pid s)).bind _ = _
    simp only [StructuralControl.runPrograms]
    congr 1
    funext t
    exact ih t

theorem schedule_safe (M : Algebra α) (base : ℝ) (candidate : Bool)
    (input output : SignedView.View) (n : Nat) (pids : List Nat) (bounds : RegionName → Nat)
    (hi : ∀ i,i<n → input.address i<bounds input.region)
    (ho : ∀ i,i<n → output.address i<bounds output.region) (s : State α) :
    ScheduleSafe bounds (pids.map (invocation M base candidate input output n)) s := by
  induction pids generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    refine ⟨?_,fun t _ => ih t⟩
    apply SourceSafety.source_safe
    · exact hi
    · exact ho

/-- Actual full-grid source result through the shared typed IO relation. The
input frame follows from the unchanged disjoint-footprint premise. -/
theorem implements (M : Algebra α) (base : ℝ) (candidate : Bool)
    (input output : SignedView.View) {n inputCapacity outputCapacity : Nat} (hn : 0<n)
    (hin : input.Valid n inputCapacity) (hout : output.Valid n outputCapacity)
    (stride : output.stride≠0) (separate : SignedGrid.FootprintsDisjoint input output n) :
    (programIO M base candidate input output n).Implements (fun xs => result (fun i =>
      scalarResult M base (inputValue xs i))) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds hb initial _
    apply schedule_safe
    · intro i hi
      exact hb.2.1 ⟨0,by simp [programIO,memory,channels]⟩ (⟨i,hi⟩,PUnit.unit) trivial
    · intro i hi
      exact hb.2.2.1 ⟨0,by simp [programIO,memory,channels]⟩ (⟨i,hi⟩,PUnit.unit) trivial
  · intro xs initial binding
    obtain ⟨final,ran,values,_,frame,_⟩ := grid_result M base candidate
      (KernelInterface.buffer input) (KernelInterface.buffer output) hn input.stride output.stride
      hin hout stride separate initial
    refine ⟨final,?_,?_,?_⟩
    · exact (schedule_run M base candidate input output n _ initial).trans ran
    · intro o j _
      have ho : o.val<1 := o.isLt
      rcases o with ⟨o,ho⟩
      have same : o=0 := by simpa [channels] using Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ ho)
      subst o
      have read := binding ⟨0,by simp [programIO,memory,channels]⟩ j trivial
      have value := values j.1.val j.1.isLt
      change final.mem output.region (output.address j.1.val)=Cell.mk .fp32
        (scalarResult M base ((initial.mem input.region (input.address j.1.val)).read .fp32)) at value
      change mapCell id (final.mem output.region (output.address j.1.val))=Cell.mk .fp32 _
      rw [value]
      simpa only [mapCell,mapValue,id_eq,programIO,memory,channels,index,inputValue] using
        congrArg (fun x => Cell.mk .fp32 (scalarResult M base x)) read
    · intro r a outside
      apply frame r a
      by_cases different : r≠output.region
      · exact .inl different
      · right
        intro i active
        have no := outside.1 ⟨0,by simp [programIO,memory,channels]⟩ (⟨i,active⟩,PUnit.unit) trivial
        exact no.elim (fun bad => False.elim (bad (Classical.not_not.mp different))) id
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.Interface
