import bench.tritonbench_g.log_softmax.Kernels
import bench.tritonbench_g.log_softmax.Proofs.KernelGrid
import bench.tritonbench_g.log_softmax.Proofs.Contract

namespace VeriTile.Bench.Repairs.LogSoftmax.RawCorrect
open VeriTile.Triton FP.Structural Address
noncomputable section
open Classical

/-- Independent unshifted log-softmax over each complete reduction row. The
kernel executes the stable shifted formula, in the mathematical Real model.
The raw input-read equations bind only initial cells and logical arrays. NaN/Inf
words remain admitted by Python and the separate FP theorem. -/
theorem forward_grid_correctness (M N K : Nat)
    (BM : Nat) (retained : Grid.RetainedBlock BM)
    (plan : Plan N (Grid.blockColumns N)) (initial : State (WithBot ℝ))
    (xs : Nat → ℝ) (binding : KernelMath.InputValues initial "input" M N K xs) :
    Spec.Real (∃ final,
      KernelGrid.run .forward (KernelMath.forwardEngine RealProof.ops plan) Bool.false
        M N K BM (KernelGrid.programs M N K BM) initial=some final ∧
      (∀ q : Coord M N K,final.mem "output" (linear q)=.mk .real
        (some (RealProof.targetForward
          (fun j : Fin N => xs (offset N K q.1.val j.val q.2.2.val)) q.2.1))) ∧
      (∀ (r : RegionName) i,r≠"output" ∨ M*N*K≤ i → final.mem r i=initial.mem r i)) := by
  obtain ⟨final,ran,result,frame⟩ := KernelGrid.whole .forward
    (KernelMath.forwardEngine RealProof.ops plan) M N K BM
    (Grid.block_bounds retained).1 (block_positive N) plan.covers initial
  refine ⟨final,ran,?_,?_⟩
  · intro q
    rw [result q]
    unfold KernelGrid.cell
    apply congrArg (Cell.mk .real)
    rw [KernelGrid.forward_graph]
    rw [forward_congr_active RealProof.ops (KernelMath.uniform plan) _
      (fun j => some (xs (offset N K q.1.val j q.2.2.val)))
      (fun j hj => binding _ (VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress.non_inner_bound
        q.1.isLt hj q.2.2.isLt)) _ q.2.1.isLt]
    exact RealProof.forward_target (KernelMath.uniform plan) _ q.2.1
  · intro r i outside
    apply frame r i
    exact outside

/-- Conventional log-softmax VJP using the actual saved forward values and
upstream gradient: g_i - exp(saved_y_i) * sum(g). Both inputs and every exterior
memory cell are preserved. The source's explicit zero gradient padding is used. -/
theorem backward_grid_correctness (M N K : Nat)
    (BM : Nat) (retained : Grid.RetainedBlock BM)
    (plan : Plan N (Grid.blockColumns N)) (initial : State (WithBot ℝ))
    (ys gs : Nat → ℝ)
    (saved : KernelMath.InputValues initial "input" M N K ys)
    (upstream : KernelMath.InputValues initial "gradient" M N K gs) :
    Spec.Real (∃ final,
      KernelGrid.run .backward (KernelMath.backwardEngine RealProof.ops plan) Bool.false
        M N K BM (KernelGrid.programs M N K BM)
        initial=some final ∧
      (∀ q : Coord M N K,final.mem "output" (linear q)=.mk .real
        (some (RealProof.targetBackward N
          (fun j => ys (offset N K q.1.val j q.2.2.val))
          (fun j => gs (offset N K q.1.val j q.2.2.val)) q.2.1.val))) ∧
      (∀ (r : RegionName) i,r≠"output" ∨ M*N*K≤ i → final.mem r i=initial.mem r i)) := by
  obtain ⟨final,ran,result,frame⟩ := KernelGrid.whole .backward
    (KernelMath.backwardEngine RealProof.ops plan) M N K BM
    (Grid.block_bounds retained).1 (block_positive N) plan.covers initial
  refine ⟨final,ran,?_,?_⟩
  · intro q
    rw [result q]
    unfold KernelGrid.cell
    apply congrArg (Cell.mk .real)
    rw [KernelGrid.backward_graph]
    rw [backward_congr_active RealProof.ops (KernelMath.uniform plan) _ _
      (fun j => some (ys (offset N K q.1.val j q.2.2.val)))
      (fun j => some (gs (offset N K q.1.val j q.2.2.val)))
      (fun j hj => saved _ (VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress.non_inner_bound
        q.1.isLt hj q.2.2.isLt))
      (fun j hj => upstream _ (VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress.non_inner_bound
        q.1.isLt hj q.2.2.isLt)) _ q.2.1.isLt]
    exact RealProof.backward_target (KernelMath.uniform plan) _ _ _
  · intro r i outside
    apply frame r i
    exact outside

#specNonCircular RealProof.targetForward avoiding [Source.forward,Source.forwardCommuted]
#specNonCircular RealProof.targetBackward avoiding [Source.backward,Source.backwardCommuted]
#axiomsClean forward_grid_correctness
#axiomsClean backward_grid_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogSoftmax.RawCorrect
