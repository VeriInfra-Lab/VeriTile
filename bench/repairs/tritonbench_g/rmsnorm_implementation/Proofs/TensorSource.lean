import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.TensorBinding
import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.Source

namespace VeriTile.Bench.Repairs.RmsnormImplementation.TensorSource
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsnormImplementation
open Whole
noncomputable section

/-- Each region begins at the actual tensor data pointer; storage offsets remain
in the physical memory map. Input-weight aliases are represented coherently. -/
def realState (p : Parameters) (memory : Memory ℝ) (b m : Nat) : BlockState where
  mem := fun r address => .real (
    if r="x" then memory p.input.allocation (p.input.base+address)
    else if r="weights" then memory p.weights.allocation (p.weights.base+address)
    else if r="out" then memory p.output.allocation (p.output.base+address)
    else 0)
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then b else if ax=1 then m else 0

def result (p : Parameters) (before : Memory ℝ) (final : BlockState) : Memory ℝ :=
  fun allocation offset =>
    if allocation=p.output.allocation ∧ p.output.base≤offset then
      final.readMem "out" (offset-p.output.base)
    else before allocation offset

def source (p : Parameters) : ComputeKernel :=
  kernel "x" "weights" "out" (p.input.stride 0) (p.input.stride 1) (p.input.stride 2)
    (p.weights.stride 0) (p.output.stride 0) (p.output.stride 1) (p.output.stride 2)
    p.width blockSize p.eps

def commutedSource (p : Parameters) : ComputeKernel :=
  indexCommutedKernel "x" "weights" "out" (p.input.stride 0) (p.input.stride 1) (p.input.stride 2)
    (p.weights.stride 0) (p.output.stride 0) (p.output.stride 1) (p.output.stride 2)
    p.width blockSize p.eps

theorem commuted_run (p : Parameters) (s : BlockState) :
    exec (commutedSource p).eraseDType.toAlgKernel s=exec (source p).eraseDType.toAlgKernel s :=
  Source.commuted_run _ _ _ _ _ _ _ _ _ _ _ _ _ _

def rowAddress (p : Parameters) (b m k : Nat) : Nat :=
  b*p.output.stride 0+m*p.output.stride 1+k*p.output.stride 2

@[simp] theorem input_read (p : Parameters) (memory : Memory ℝ) (b m k : Nat) :
    (realState p memory b m).readMem "x"
      (xColOffset (realState p memory b m) (p.input.stride 0) (p.input.stride 1) (p.input.stride 2) k)=
      memory p.input.allocation (inputOffset p b m k) := by
  simp [realState,BlockState.readMem,xColOffset,inputOffset,Nat.add_assoc]

@[simp] theorem weight_read (p : Parameters) (memory : Memory ℝ) (b m k : Nat) :
    (realState p memory b m).readMem "weights" (k*p.weights.stride 0)=
      memory p.weights.allocation (weightOffset p k) := by
  simp [realState,BlockState.readMem,weightOffset]

@[simp] theorem output_address (p : Parameters) (memory : Memory ℝ) (b m k : Nat) :
    outColOffset (realState p memory b m) (p.output.stride 0) (p.output.stride 1) (p.output.stride 2) k =
      rowAddress p b m k := by simp [outColOffset,realState,rowAddress]

theorem unique (p : Parameters) (memory : Memory ℝ) (b m : Nat)
    (hb : b<p.batches) (hm : m<p.rows) (hu : OutputUnique p) :
    Function.Injective (fun k : Fin p.width=>outColOffset (realState p memory b m)
      (p.output.stride 0) (p.output.stride 1) (p.output.stride 2) k.val) := by
  intro i j same
  apply Fin.ext
  apply (hu b m i.val b m j.val hb hm i.isLt hb hm j.isLt ?_).2.2
  simpa [output_address,rowAddress,outputOffset,Nat.add_assoc] using congrArg (p.output.base+·) same

theorem result_read (p : Parameters) (before : Memory ℝ) (final : BlockState) (b m k : Nat) :
    result p before final p.output.allocation (outputOffset p b m k)=
      final.readMem "out" (rowAddress p b m k) := by
  have eq : outputOffset p b m k=p.output.base+rowAddress p b m k := by
    simp [outputOffset,rowAddress,Nat.add_assoc]
  simp [result,eq]

theorem result_frame (p : Parameters) (before : Memory ℝ) (final : BlockState) (b m : Nat)
    (frame : ∀ address,(∀ k : Fin p.width,address≠rowAddress p b m k.val) →
      final.mem "out" address=(realState p before b m).mem "out" address)
    (allocation offset : Nat)
    (outside : ∀ k,k<p.width → ¬(allocation=p.output.allocation ∧ offset=outputOffset p b m k)) :
    result p before final allocation offset=before allocation offset := by
  unfold result
  split_ifs with active
  · rcases active with ⟨rfl,bound⟩
    have fr := frame (offset-p.output.base) (by
      intro k same
      apply outside k.val k.isLt
      refine ⟨rfl,?_⟩
      simp only [outputOffset,rowAddress] at same ⊢
      omega)
    unfold BlockState.readMem
    rw [fr]
    simp [realState,Nat.add_sub_of_le bound]
  · rfl

theorem mathematical_congr (n : Nat) (eps : ℝ) (xs ws ys vs : Nat → ℝ) (k : Nat)
    (hk : k<n) (hx : ∀ j,j<n → xs j=ys j) (hw : ∀ j,j<n → ws j=vs j) :
    VeriTile.Bench.Repairs.RmsnormImplementation.mathematical n eps xs ws k=
      VeriTile.Bench.Repairs.RmsnormImplementation.mathematical n eps ys vs k := by
  unfold VeriTile.Bench.Repairs.RmsnormImplementation.mathematical
  rw [hx k hk,hw k hk]
  congr 5
  apply Finset.sum_congr rfl
  intro j _
  rw [hx j.val j.isLt]

#axiomsClean unique
#axiomsClean result_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormImplementation.TensorSource
