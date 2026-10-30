import bench.repairs.tritonbench_g.logsumexp_fwd.Correct
import bench.repairs.tritonbench_g.logsumexp_fwd.Misc.NativeCorrect
import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.KernelResult
import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Contract
import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.RealExecution
import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.NativeExecution

/-! Optional Tensor input bindings and Python-check correspondence. Primary
mathematical and native kernel entries do not import this bridge. -/
namespace VeriTile.Bench.Repairs.LogsumexpFwd.TensorBridge
open VeriTile.Triton FP.Structural Native
noncomputable section

/-- The unchanged Tensor wrapper supplies exactly these raw launch scalars. -/
def geometry (a : Arguments) : RawKernel.Geometry :=
  ⟨a.rows,a.columns,a.block,a.hasScale,block_positive a⟩

theorem run_correspondence [Inhabited A] (engines : Nat → Engine A) (a : Arguments)
    (scale : A) (jobs : List Nat) (s : State A) :
    RawKernel.run engines (geometry a) scale jobs s=Grid.run engines a scale jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons j jobs ih =>
    simp only [RawKernel.run,Grid.run]
    change (FP.Structural.exec (engines j).algebra (rowBlock "input" "scratch" a.columns a.block a.hasScale)
      (Grid.programState a scale j s)).bind (RawKernel.run engines (geometry a) scale jobs)=_
    congr 1
    funext previous
    exact ih previous

theorem python_checks_iff_preconditions (a : Arguments) :
    checkInputs a=.ok () ↔ InputContract a := checkInputs_iff a

theorem logsumexp_kernel_correct (a : Arguments) (h : InputContract a)
    (x : KernelBinding.Input a) (scale : ℝ) (read : Nat → ℝ)
    (s : State (WithBot ℝ)) (input : KernelBinding.RealCells x h read s)
    (jobs : List Nat) (schedule : jobs.Perm (List.range (a.rows*a.blocks))) :
    VeriTile.Spec.Real (∃ t,
      Grid.run (fun _ => RealModel.engine) a (scale : WithBot ℝ) jobs s=some t ∧
      (∀ row segment,row<a.rows → segment<a.blocks →
        t.mem "scratch" (row*a.blocks+segment)=.mk .real
          (KernelMath.block (x.realValue read) row segment a.columns a.block a.hasScale scale : WithBot ℝ)) ∧
      (∀ (r : RegionName) i,(r ≠ "scratch" ∨ a.rows*a.blocks ≤ i) → t.mem r i=s.mem r i)) := by
  have raw := RawReal.logsumexp_real_raw (geometry a) scale (x.realValue read) s
    (fun row hr col hc=>KernelBinding.real_cells x h read s input _
      (by simpa only [geometry,Nat.zero_mul,Nat.zero_add] using
        input_offset_bound a h row 0 col hr (by simpa using hc))) jobs schedule
  simpa only [run_correspondence] using raw

theorem logsumexp_kernel_fp32 (ops : Ops) (plans : Nat → Plans)
    (a : Arguments) (h : InputContract a) (x : KernelBinding.Input a)
    (scale : Word) (read : Nat → VeriTile.Torch.FP32.Word) (s : State Word)
    (input : KernelBinding.WordCells x h read s) (jobs : List Nat)
    (schedule : jobs.Perm (List.range (a.rows*a.blocks))) :
    ∃ t,Grid.run (fun j => engine ops (plans j)) a scale jobs s=some t ∧
      (∀ row segment,row<a.rows → segment<a.blocks →
        t.mem "scratch" (row*a.blocks+segment)=.mk .real
          (blockValue (engine ops (plans (row*a.blocks+segment))) (x.wordValue read)
            scale row segment a.columns a.block a.hasScale)) ∧
      (∀ (r : RegionName) i,(r ≠ "scratch" ∨ a.rows*a.blocks ≤ i) → t.mem r i=s.mem r i) := by
  have raw := RawNative.logsumexp_native_raw ops plans (geometry a) scale (x.wordValue read) s
    (fun row hr col hc=>KernelBinding.word_cells x h read s input _
      (by simpa only [geometry,Nat.zero_mul,Nat.zero_add] using
        input_offset_bound a h row 0 col hr (by simpa using hc))) jobs schedule
  simpa only [run_correspondence] using raw

#axiomsClean logsumexp_kernel_correct
#axiomsClean logsumexp_kernel_fp32
#axiomsClean run_correspondence
#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogsumexpFwd.TensorBridge
