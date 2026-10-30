import bench.tritonbench_g.relu_strided_buffer.IO
import bench.tritonbench_g.relu_strided_buffer.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.ReluStridedBuffer.ReluIO
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer SignedView Whole StructuralControl
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
local instance wordInhabited : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

theorem invocation_initial (M : Algebra α) (folded : Bool) (x y : SignedView.View)
    (n pid : Nat) (s : State α) :
    (invocation M folded x y n pid).initial s=atProgram (Raw.arguments x y n).C pid s := by
  simp [invocation,Invocation.initial,atProgram,bindRegisters]

theorem schedule_run {α : Type} [Inhabited α] (M : Algebra α) (folded : Bool)
    (x y : SignedView.View) (n : Nat) (jobs : List Nat) (s : State α) :
    runSchedule (jobs.map (invocation M folded x y n)) s=
      runPrograms M (Access.selected folded (parameters (Raw.arguments x y n))).surfaceBody
        (Raw.arguments x y n).C jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid jobs ih =>
    simp only [List.map_cons,runSchedule,Invocation.run,invocation_initial,runPrograms]
    change (run M (Access.selected folded (parameters (Raw.arguments x y n))).surfaceBody
      (atProgram (Raw.arguments x y n).C pid s)).bind _=_
    cases run M (Access.selected folded (parameters (Raw.arguments x y n))).surfaceBody
      (atProgram (Raw.arguments x y n).C pid s) with
    | none => rfl
    | some t => exact ih t

theorem invocation_safe {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (compare : M.compareLt none .fp32=some cmp) (folded : Bool) (x y : SignedView.View)
    (n : Nat) (values : Inputs α x n) (bounds : RegionName → Nat)
    (backed : (kernelIO M folded x y n).InBounds bounds values) (pid : Nat) (s : State α) :
    (invocation M folded x y n pid).Safe bounds s := by
  apply Access.source_safe M cmp compare folded (parameters (Raw.arguments x y n))
  · intro i active
    exact backed.2.1 ⟨0,by simp [kernelIO,memory,inputChannels]⟩ (⟨i,active⟩,PUnit.unit) trivial
  · intro i active
    exact backed.2.2.1 ⟨0,by simp [kernelIO,memory,outputChannels]⟩ (⟨i,active⟩,PUnit.unit) trivial

theorem schedule_safe {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (compare : M.compareLt none .fp32=some cmp) (folded : Bool) (x y : SignedView.View)
    (n : Nat) (values : Inputs α x n) (bounds : RegionName → Nat)
    (backed : (kernelIO M folded x y n).InBounds bounds values) (jobs : List Nat) (s : State α) :
    ScheduleSafe bounds (jobs.map (invocation M folded x y n)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons pid jobs ih =>
    exact ⟨invocation_safe M cmp compare folded x y n values bounds backed pid s,fun t _ => ih t⟩

/-- Shared assembly reuses the exact signed full-grid result. Read bindings
are enough; no stricter cell-tag assumption or missing loop proof is assumed. -/
theorem implements {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (compare : M.compareLt none .fp32=some cmp) (f : α → α)
    (formula : ∀ z,StructuralTile.scalarResult M cmp z=f z) (folded : Bool)
    (x y : SignedView.View) (n inputCapacity outputCapacity : Nat) (positive : 0<n)
    (input : x.Valid n inputCapacity) (output : y.Valid n outputCapacity) (stride : y.stride≠0)
    (separate : SignedGrid.FootprintsDisjoint x y n) :
    (kernelIO M folded x y n).Implements (pointwise f x y n) := by
  intro values _ bounds backed s bound
  obtain ⟨t,ran,result,_,frame,_⟩ := Raw.grid M cmp compare x y positive input output stride separate s
  refine ⟨schedule_safe M cmp compare folded x y n values bounds backed _ s,t,?_,?_,?_⟩
  · change runSchedule ((List.range (Raw.arguments x y n).C).map (invocation M folded x y n)) s=_
    rw [schedule_run]
    cases folded
    · exact ran
    · exact (grid_folded_execution M (Raw.arguments x y n) s).symm.trans ran
  · intro o j _
    fin_cases o
    change mapCell id (t.mem y.region (y.address j.1.val))=Cell.mk .fp32 (f (element values j.1))
    rw [result j.1.val j.1.isLt]
    have read : (s.mem x.region (x.address j.1.val)).read .fp32=element values j.1 := by
      simpa [kernelIO,memory,inputChannels,element,mapValue] using
        bound ⟨0,by simp [kernelIO,memory,inputChannels]⟩ (j.1,PUnit.unit) trivial
    rw [read,formula]
    rfl
  · intro r address outside
    apply frame r address
    by_cases hr : r=y.region
    · right
      intro i active
      have missed := outside.1 ⟨0,by simp [kernelIO,memory,outputChannels]⟩
        (⟨i,active⟩,PUnit.unit) trivial
      exact missed.resolve_left (fun ne => ne hr)
    · exact Or.inl hr

theorem real_implements (rest : Algebra ℝ) (x y : SignedView.View)
    (n inputCapacity outputCapacity : Nat) (positive : 0<n)
    (input : x.Valid n inputCapacity) (output : y.Valid n outputCapacity) (stride : y.stride≠0)
    (separate : SignedGrid.FootprintsDisjoint x y n) :
    (kernelIO (realBackend rest) Bool.false x y n).Implements (mathematical x y n) :=
  implements (realBackend rest) (fun a b => decide (a<b)) (real_compare rest)
    (fun z => max 0 z) (real_scalar rest) Bool.false x y n inputCapacity outputCapacity
    positive input output stride separate

theorem native_scalar (rest : Algebra (FP.Value .fp32)) (z : FP.Value .fp32) :
    StructuralTile.scalarResult (fp32Backend rest) WordTile.orderedLt z=nativeRelu z := by
  simpa only [fp32Backend,fp32_result_classification,nativeRelu] using
    WordTile.scalarResult_word (zeroLiterals rest) .ieeeValue (zero_literal_exact rest) z

theorem native_implements (rest : Algebra (FP.Value .fp32)) (folded : Bool) (x y : SignedView.View)
    (n inputCapacity outputCapacity : Nat) (positive : 0<n)
    (input : x.Valid n inputCapacity) (output : y.Valid n outputCapacity) (stride : y.stride≠0)
    (separate : SignedGrid.FootprintsDisjoint x y n) :
    (kernelIO (fp32Backend rest) folded x y n).Implements (nativeTarget x y n) :=
  implements (fp32Backend rest) WordTile.orderedLt
    (WordTile.backend_compare (zeroLiterals rest) .ieeeValue) nativeRelu (native_scalar rest)
    folded x y n inputCapacity outputCapacity positive input output stride separate

theorem native_equivalent (rest : Algebra (FP.Value .fp32)) (x y : SignedView.View)
    (n inputCapacity outputCapacity : Nat) (positive : 0<n)
    (input : x.Valid n inputCapacity) (output : y.Valid n outputCapacity) (stride : y.stride≠0)
    (separate : SignedGrid.FootprintsDisjoint x y n) :
    (kernelIO (fp32Backend rest) Bool.false x y n).Equivalent (candidate (fp32Backend rest) x y n) :=
  ProgramIO.EquivalentOn.of_implementsOn
    (native_implements rest Bool.false x y n inputCapacity outputCapacity positive input output stride separate)
    (native_implements rest Bool.true x y n inputCapacity outputCapacity positive input output stride separate)

#axiomsClean real_implements
#axiomsClean native_implements
#axiomsClean native_equivalent
end
end VeriTile.Bench.Repairs.ReluStridedBuffer.ReluIO
