import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.ActiveOutput
import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.Source

namespace VeriTile.Bench.Repairs.RmsnormTriton.TensorSource
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsnormTriton
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
  Source.candidate_run _ _ _ _ _ _ _ _ _ _ _ _ _ _

def rowAddress (p : Parameters) (b m k : Nat) : Nat :=
  b*p.output.stride 0+m*p.output.stride 1+k*p.output.stride 2

@[simp] theorem input_read (p : Parameters) (memory : Memory ℝ) (b m k : Nat) :
    (realState p memory b m).readMem "x"
      (Full.ScratchRms.xOff (realState p memory b m) (p.input.stride 0) (p.input.stride 1) (p.input.stride 2) k)=
      memory p.input.allocation (inputOffset p b m k) := by
  simp [realState,BlockState.readMem,Full.ScratchRms.xOff,inputOffset,Nat.add_assoc]

@[simp] theorem weight_read (p : Parameters) (memory : Memory ℝ) (b m k : Nat) :
    (realState p memory b m).readMem "weights" (k*p.weights.stride 0)=
      memory p.weights.allocation (weightOffset p k) := by
  simp [realState,BlockState.readMem,weightOffset]

@[simp] theorem output_address (p : Parameters) (memory : Memory ℝ) (b m k : Nat) :
    Full.Wb.outOff (realState p memory b m) (p.output.stride 0) (p.output.stride 1) (p.output.stride 2) k =
      rowAddress p b m k := by simp [Full.Wb.outOff,realState,rowAddress]

theorem unique (p : Parameters) (memory : Memory ℝ) (b m : Nat)
    (hb : b<p.batches) (hm : m<p.rows) (hu : OutputUnique p) :
    Function.Injective (fun k : Fin p.width=>Full.Wb.outOff (realState p memory b m)
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
    VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole.mathematical n eps xs ws k=
      VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole.mathematical n eps ys vs k := by
  unfold VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole.mathematical
  rw [hx k hk,hw k hk]
  congr 5
  apply Finset.sum_congr rfl
  intro j _
  rw [hx j.val j.isLt]

theorem run (p : Parameters) (before : Memory ℝ) (b m : Nat)
    (hb : b<p.batches) (hm : m<p.rows) (hu : OutputUnique p) :
    ∃ final,exec (source p).eraseDType.toAlgKernel (realState p before b m)=some final ∧
      (∀ k : Fin p.width,final.readMem "out" (rowAddress p b m k.val)=
        mathematical p.width p.eps
          (fun j=>before p.input.allocation (inputOffset p b m j))
          (fun j=>before p.weights.allocation (weightOffset p j)) k.val) ∧
      (∀ (r : RegionName) address,
        (r≠"out" ∨ ∀ k : Fin p.width,address≠rowAddress p b m k.val) →
        final.mem r address=(realState p before b m).mem r address) := by
  obtain ⟨final,ran,values,frame⟩ := ActiveOutput.run .triv "x" "weights" "out"
    (p.input.stride 0) (p.input.stride 1) (p.input.stride 2) (p.weights.stride 0)
    (p.output.stride 0) (p.output.stride 1) (p.output.stride 2) p.width blockSize p.eps
    (by decide) (by decide) (by decide) (realState p before b m) (unique p before b m hb hm hu)
  rw [execR_triv] at ran
  refine ⟨final,(Source.original_run _ _ _ _ _ _ _ _ _ _ _ _ _ _).trans ran,?_,?_⟩
  · intro k
    simpa [Full.Final2.rmsSpecFull,Full.Final2.meanSq,
      mathematical,WithBot.realRsqrt,WithBot.unbotD,div_eq_mul_inv] using values k
  · simpa only [output_address] using frame

/-- One actual typed JIT program, returning its complete physical-memory image. -/
def rowRun (p : Parameters) (b m : Nat) (before : Memory ℝ) : Option (Memory ℝ) :=
  (exec (source p).eraseDType.toAlgKernel (realState p before b m)).map (result p before)

theorem row_run (red : Reduction) (p : Parameters) (before : Memory ℝ) (b m : Nat)
    (hb : b<p.batches) (hm : m<p.rows) (sep : Separation p) (hu : OutputUnique p)
    (undefinedWeight : Nat → ℝ) :
    rowRun p b m before=some (executeRow realNumerics red p b m undefinedWeight before) := by
  obtain ⟨final,ran,values,frame⟩ := run p before b m hb hm hu
  unfold rowRun
  rw [ran,Option.map_some]
  apply congrArg some
  funext allocation offset
  by_cases hit : ∃ k,k<p.width ∧ allocation=p.output.allocation ∧ offset=outputOffset p b m k
  · rcases hit with ⟨k,hk,rfl,rfl⟩
    rw [result_read,values ⟨k,hk⟩,row_value realNumerics red p sep hu b m hb hm _ before k hk,
      logicalValue_real]
  · have outside : ∀ k,k<p.width → ¬(allocation=p.output.allocation ∧ offset=outputOffset p b m k) :=
      fun k hk same => hit ⟨k,hk,same⟩
    rw [result_frame p before final b m (fun address h => frame "out" address (Or.inr h))
      allocation offset outside,row_frame realNumerics red p b m undefinedWeight before allocation offset outside]

/-- Full source grid with the same arbitrary complete launch order. A failed
source program propagates failure; successful execution is proved below. -/
def grid (p : Parameters) : List (Nat × Nat) → Memory ℝ → Option (Memory ℝ)
  | [],before => some before
  | (b,m)::rest,before => (rowRun p b m before).bind (grid p rest)

theorem grid_run (red : Reduction) (p : Parameters) (sep : Separation p) (hu : OutputUnique p)
    (undefinedWeight : Nat → ℝ) (order : List (Nat × Nat))
    (valid : ∀ bm∈order,bm.1<p.batches ∧ bm.2<p.rows) (before : Memory ℝ) :
    grid p order before=some (executeGrid realNumerics red p undefinedWeight order before) := by
  induction order generalizing before with
  | nil => rfl
  | cons bm rest ih =>
    rcases bm with ⟨b,m⟩
    have active := valid (b,m) List.mem_cons_self
    simp only [grid,row_run red p before b m active.1 active.2 sep hu undefinedWeight,Option.bind_some,
      executeGrid]
    exact ih (fun bm h => valid bm (List.mem_cons_of_mem _ h)) _

theorem scheduled_run (red : Reduction) (p : Parameters) (sep : Separation p) (hu : OutputUnique p)
    (undefinedWeight : Nat → ℝ) (schedule : Schedule p) (before : Memory ℝ) :
    grid p schedule.order before=kernelRun realNumerics red p undefinedWeight schedule before := by
  exact grid_run red p sep hu undefinedWeight schedule.order
    (fun bm h => (programGrid_mem p bm.1 bm.2).mp (schedule.complete.mem_iff.mp h)) before

def candidateRowRun (p : Parameters) (b m : Nat) (before : Memory ℝ) : Option (Memory ℝ) :=
  (exec (commutedSource p).eraseDType.toAlgKernel (realState p before b m)).map (result p before)

def candidateGrid (p : Parameters) : List (Nat × Nat) → Memory ℝ → Option (Memory ℝ)
  | [],before => some before
  | (b,m)::rest,before => (candidateRowRun p b m before).bind (candidateGrid p rest)

theorem candidate_grid (p : Parameters) (order : List (Nat × Nat)) (before : Memory ℝ) :
    candidateGrid p order before=grid p order before := by
  induction order generalizing before with
  | nil => rfl
  | cons bm rest ih =>
    rcases bm with ⟨b,m⟩
    simp only [candidateGrid,grid,candidateRowRun,rowRun,commuted_run]
    apply congrArg (Option.bind _)
    funext memory
    exact ih memory

#axiomsClean row_run
#axiomsClean grid_run
#axiomsClean run
#axiomsClean unique
#axiomsClean result_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormTriton.TensorSource
