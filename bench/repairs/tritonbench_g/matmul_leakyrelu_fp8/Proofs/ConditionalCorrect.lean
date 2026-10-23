import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.HostReal

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Correct
open Host VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section

/-! Complete selected-default-FP32 wrapper contract. A is contiguous and
A/B are raw/resolved on one device, with ordinary nonnegative strides and
backing metadata. Negative stored values and read-only aliasing are allowed.
Output is always fresh FP16, after the original optional left-side leaky-ReLU.
All empty/zero-K branches and arbitrary finite autotune trial counts are
composed. No native-to-Real accuracy claim is made. -/

structure Specification {σ : Type} (live : σ → Nat → Prop) (args : Arguments)
    (initial final : World ℝ σ) (output : Tensor) : Prop where
  rows : output.rows=args.A.rows
  columns : output.columns=args.B.columns
  dtype : output.dtype=.fp16
  device : output.device=args.A.device
  offset : output.offset=0
  rowStride : output.rowStride=max args.B.columns 1
  columnStride : output.columnStride=1
  raw : output.isNeg=Bool.false ∧ output.isConj=Bool.false
  bounded : output.Bounded
  fresh : ∀ storage,live initial.native storage → output.storage≠storage
  values : ∀ row<args.A.rows,∀ column<args.B.columns,
    final.memory (output.address row column)=
      Real.formula args.problem ⟨args.A,args.B,output⟩ initial.memory row column
  callerFrame : ∀ point,live initial.native point.1 → final.memory point=initial.memory point

theorem complete {σ : Type} (runtime : Runtime σ) (live : σ → Nat → Prop)
    (args : Arguments) (initial final : World ℝ σ) (output : Tensor)
    (library : Library runtime live) (input : InputContract live args initial.native)
    (execution : Host.run Real.arithmetic runtime args initial=some (output,final)) :
    Specification live args initial final output := by
  obtain ⟨rows,columns,dtype,device,offset,rs,cs,neg,conj,bounded,fresh,values,frame⟩ :=
    whole_real runtime live args initial final output library input execution
  exact ⟨rows,columns,dtype,device,offset,rs,cs,⟨neg,conj⟩,bounded,fresh,values,frame⟩

#axiomsClean complete
#specNonCircular Real.formula avoiding [Host.run,Source.run,Source.accumulator]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Correct
