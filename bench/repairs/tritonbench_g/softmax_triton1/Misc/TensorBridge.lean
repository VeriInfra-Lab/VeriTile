import bench.repairs.tritonbench_g.softmax_triton1.Correct
import bench.repairs.tritonbench_g.softmax_triton1.FPEquiv
import bench.repairs.tritonbench_g.softmax_triton1.Misc.NativeObservation
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Addresses
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.NativeKernelResult

/-! Optional actual-Tensor application of the mathematical row theorem and
Python-check correspondence. The actual Tensor native application is retained
here; NativeObservation.lean exposes the source-connected typed ProgramIO interface. -/
namespace VeriTile.Bench.Repairs.SoftmaxTriton1.TensorBridge
open VeriTile.Triton FP.Structural Contiguous
open Correct
noncomputable section

/-- Real memory interpretation for a physical-address input reader. -/
def realAccess : Access BlockState MemCell ℝ :=
  ⟨BlockState.mem,fun c => match c.readAs .real with | some (some v) => v | _ => 0⟩

/-- The public checked entry binds the original Tensor's actual backing
storage to the row values. Every valid program ID and every logical column
are covered, including nonzero storage offsets and singleton stride metadata.
No wrapper, allocation or returned-object proof is a premise. -/
theorem softmax_checked_kernel_correctness
    (x : Object) (h : InputContract x.metadata) (out : RegionName) (ob : Nat)
    (read : Nat → ℝ) (s : BlockState) (binding : InputMemory realAccess read x s)
    (pid : s.pid < rows x.metadata) :
    Spec.Real (∃ t,
      exec (softmax_kernel out x.region x.tensor.storageOffset ob
        (cols x.metadata) 1 (cols x.metadata) 1 (cols x.metadata)
        (blockSize (cols x.metadata))).toAlgKernel s = some t ∧
      (∀ c, c < cols x.metadata →
        t.readMem out (ob+s.pid*cols x.metadata+c) =
          reference (cols x.metadata) (logicalValue Neg.neg read x s.pid) c) ∧
      (∀ r a, (∀ c, c < cols x.metadata →
        ¬(out=r ∧ ob+s.pid*cols x.metadata+c=a)) → t.mem r a=s.mem r a)) := by
  let n := cols x.metadata
  let B := blockSize n
  have hB := (blockSize_bounds n).1
  have cover := (blockSize_bounds n).2
  obtain ⟨t,he⟩ := row_exec out x.region x.tensor.storageOffset ob n 1 n 1 n B hB s
  refine ⟨t,he,?_,?_⟩
  · intro c hc
    let i : Fin B := ⟨c,lt_of_lt_of_le hc cover⟩
    have value := row_value out x.region x.tensor.storageOffset ob n 1 n 1 n B hB
      (by intro i j _ _ eq; apply Fin.ext; simpa using eq) s t he i hc
    have same : ∀ j : Fin B, j.val < n →
        s.readMem x.region (x.tensor.storageOffset+s.pid*n+j.val*1) =
          logicalValue Neg.neg read x s.pid j.val := by
      intro j hj
      simpa [n,realAccess,Access.read,BlockState.readMem] using
        input_read realAccess Neg.neg read x h s binding s.pid j.val pid hj
    rw [VeriTile.Bench.TritonBenchG.SoftmaxTriton1.softmaxSpec_congr n B _
      (fun j => logicalValue Neg.neg read x s.pid j.val) same i,
      VeriTile.Bench.TritonBenchG.SoftmaxTriton1.softmaxSpec_eq_formula n B hB _ i hc,
      RealModel.padded_sum_eq n B cover (fun j => Real.exp (logicalValue Neg.neg read x s.pid j))] at value
    simpa [reference,i,n] using value
  · intro r a miss
    exact row_frame out x.region x.tensor.storageOffset ob n 1 n 1 n B hB s t he r a
      (by intro i hi; simpa using miss i.val hi)

theorem python_checks_iff_preconditions (x : ArgumentMetadata) :
    checkInput x=.ok () ↔ AdmittedArgument x := checkInput_iff x

/-- Successful native source execution and every logical output of one valid
row, bound to the original Tensor storage through the shared Tensor laws.
All raw FP32 inputs remain admitted; this result needs no reciprocal law.
The output descriptor's shape/injectivity are ordinary invocation premises. -/
theorem softmax_checked_fp32 (inst : NativeGraph.Instructions)
    (x : Object) (h : InputContract x.metadata) (y : Tensor)
    (shape : SameShape (descriptor x) y) (injective : y.Injective)
    (p : Nat) (hp : p < rows x.metadata)
    (read : Nat → NativeGraph.Word) (s : State NativeGraph.Word)
    (binding : InputMemory Floating.access read x s) :
    ∃ t, NativeGraph.evaluate inst 1 (descriptor x) y (blockSize (cols x.metadata)) p s = some t ∧
      (∀ c, c < cols x.metadata →
        (t.mem y.region (y.index p c)).read .real =
          FPEquiv.reference inst (cols x.metadata) (blockSize (cols x.metadata))
            (logicalValue FPEquiv.negateWord read x p) c) ∧
      FrameRows Floating.access y [p] s t := by
  have input : Inputs Floating.access (descriptor x) (logicalValue FPEquiv.negateWord read x) s := by
    intro r c hr hc
    simpa [descriptor,Tensor.index] using input_read Floating.access FPEquiv.negateWord read x h s binding r c hr hc
  obtain ⟨t,he⟩ := NativeGraph.source_terminates inst 1 (descriptor x) y
    (blockSize (cols x.metadata)) p (blockSize_bounds _).1 s
  have result := (NativeGraph.source inst 1).row (descriptor x) y
    (blockSize (cols x.metadata)) p s t (logicalValue FPEquiv.negateWord read x)
    shape injective (blockSize_bounds _).1 (blockSize_bounds _).2 hp input he
  exact ⟨t,he,result⟩

#axiomsClean softmax_checked_kernel_correctness
#axiomsClean python_checks_iff_preconditions
#axiomsClean softmax_checked_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton1.TensorBridge
