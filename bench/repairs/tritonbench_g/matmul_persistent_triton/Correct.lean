import bench.repairs.tritonbench_g.matmul_persistent_triton.Contract
import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.HostReal

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton.Correct
open Host VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section

/-!
Independent Real specification of the complete repaired matmul wrapper.
Input is two rank-two raw/resolved matching FP16 or E4M3 tensors on the same
CUDA device, with compatible K, ordinary nonnegative strided backing and
representable Tensor metadata. Stride zero, nonzero offsets, stored negative
values and A/B storage aliasing are allowed. Output is a fresh contiguous C;
empty M/N and K=0 retain their exact host behavior. Real interpretation does
not claim equality between rounded native dot and exact matrix arithmetic.
-/

structure Specification {σ : Type} (live : σ → Nat → Prop) (args : Arguments)
    (initial final : World ℝ σ) (output : Tensor) : Prop where
  rows : output.rows=args.A.rows
  columns : output.columns=args.B.columns
  dtype : output.dtype=args.A.dtype
  device : output.device=args.A.device
  offset : output.offset=0
  rowStride : output.rowStride=max args.B.columns 1
  columnStride : output.columnStride=1
  raw : output.isNeg=Bool.false ∧ output.isConj=Bool.false
  bounded : output.Bounded
  fresh : ∀ storage,live initial.native storage → output.storage≠storage
  values : ∀ row<args.A.rows,∀ column<args.B.columns,
    final.memory (output.address row column)=
      RealMath.formula args.A.columns (Real.inputs ⟨args.A,args.B,output⟩ initial.memory) row column
  callerFrame : ∀ point,live initial.native point.1 → final.memory point=initial.memory point

theorem complete {σ : Type} (runtime : Runtime σ) (live : σ → Nat → Prop)
    (args : Arguments) (initial final : World ℝ σ) (output : Tensor)
    (library : Library runtime live) (input : InputContract live args initial.native)
    (execution : Host.run Real.arithmetic runtime args initial=some (output,final)) :
    Specification live args initial final output := by
  obtain ⟨rows,columns,dtype,device,offset,rs,cs,neg,conj,bounded,fresh,values,frame⟩ :=
    whole_real runtime live args initial final output library input execution
  exact ⟨rows,columns,dtype,device,offset,rs,cs,⟨neg,conj⟩,bounded,fresh,values,frame⟩


/-- Checked admission supplies every non-ordinary input premise to the
accepted complete mathematical wrapper theorem. -/
theorem checked_complete {σ : Type} (runtime : Runtime σ) (live : σ → Nat → Prop)
    (args : Arguments) (initial final : World ℝ σ) (output : Tensor)
    (library : Library runtime live) (guard : Checked.checkInputs args = .ok ())
    (aBound : args.A.Bounded) (bBound : args.B.Bounded)
    (aLive : live initial.native args.A.storage) (bLive : live initial.native args.B.storage)
    (execution : Host.run Real.arithmetic runtime args initial = some (output,final)) :
    Specification live args initial final output :=
  complete runtime live args initial final output library
    (Checked.admitted_input live args initial.native guard aBound bBound aLive bLive) execution

#axiomsClean checked_complete

#axiomsClean complete
#specNonCircular RealMath.formula avoiding [Host.run,Source.run,Source.accumulator]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulPersistentTriton.Correct
