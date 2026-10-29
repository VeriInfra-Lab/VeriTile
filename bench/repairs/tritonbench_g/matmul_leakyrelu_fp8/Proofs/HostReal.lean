import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.HostFrame
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Real

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Host
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 5000000

theorem Completion.output_real {σ : Type} {runtime : Runtime σ} {live : σ → Nat → Prop}
    {args : Arguments} {initial final : World ℝ σ} {output : Tensor}
    (e : Completion Real.arithmetic runtime live args initial final output)
    (row column : Nat) (hr : row<args.A.rows) (hc : column<args.B.columns) :
    final.memory (output.address row column)=
      Real.formula args.problem ⟨args.A,args.B,output⟩ initial.memory row column := by
  cases e.branch with
  | empty empty same =>
    change args.A.rows=0 ∨ args.B.columns=0 at empty
    omega
  | zero nonempty zeroK native executed same =>
    rw [same]
    change zeroMemory Real.arithmetic.inputZero args.problem ⟨args.A,args.B,output⟩
      initial.memory (output.address row column)=_
    have active : outputCell args.problem ⟨args.A,args.B,output⟩ (output.address row column) :=
      ⟨row,hr,column,hc,rfl⟩
    change args.A.columns=0 at zeroK
    rw [zeroMemory,if_pos active]
    simp [Real.arithmetic,Real.formula,Arguments.problem,zeroK,RealMath.zero_K,Real.activate]
  | launch nonempty positiveK trials selected executed =>
    have memory := (sequence_memory Real.arithmetic runtime args output (trials+1) _ final executed).1
    rw [memory,Source.replay_positive Real.arithmetic args.problem _ e.layout trials initial.memory]
    exact Real.output args.problem _ e.layout initial.memory row column hr hc

/-- Whole wrapper correctness includes allocation/return and every branch.
The output shape/layout and separation are derived from ordinary allocation,
never supplied as a task-specific input or final-state assumption. -/
theorem whole_real {σ : Type} (runtime : Runtime σ) (live : σ → Nat → Prop)
    (args : Arguments) (initial final : World ℝ σ) (output : Tensor)
    (library : Library runtime live) (input : InputContract live args initial.native)
    (executed : run Real.arithmetic runtime args initial=some (output,final)) :
    output.rows=args.A.rows ∧ output.columns=args.B.columns ∧ output.dtype=.fp16 ∧
    output.device=args.A.device ∧ output.offset=0 ∧ output.rowStride=max args.B.columns 1 ∧
    output.columnStride=1 ∧ output.isNeg=Bool.false ∧ output.isConj=Bool.false ∧ output.Bounded ∧
    (∀ storage,live initial.native storage → output.storage≠storage) ∧
    (∀ row<args.A.rows,∀ column<args.B.columns,
      final.memory (output.address row column)=
        Real.formula args.problem ⟨args.A,args.B,output⟩ initial.memory row column) ∧
    (∀ point,live initial.native point.1 → final.memory point=initial.memory point) := by
  obtain ⟨e⟩ := complete Real.arithmetic runtime live args initial final output library input executed
  have allocated := library.allocation _ _ _ _ _ _ _ e.allocation
  exact ⟨allocated.rowsExact,allocated.columnsExact,allocated.dtype,allocated.deviceExact,
    allocated.offset,allocated.rowStride,allocated.columnStride,allocated.raw.1,allocated.raw.2,
    allocated.bounded,allocated.fresh,(fun row hr column hc => e.output_real row column hr hc),e.caller_frame⟩

#axiomsClean whole_real
#axiomsClean Completion.caller_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Host
