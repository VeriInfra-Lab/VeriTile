import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourceLoop

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.SourceOutput
open VeriTile.Triton FP.Structural SourceBody SourcePrelude
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem runs {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM pid off : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (memory : RegionName → Nat → Cell α) (acc : Acc α BN BM) (s : State α)
    (domain : SourceDomain N M BN BM pid)
    (ctx : Context N M BN BM pid off memory acc s) :
    run A (SourceCode.original N M BN BM rowShape columnShape).after s =
      some (WholeSource.suffixState A (parameters N M BN BM) pid acc s) := by
  have hrow (i : Fin BN) : (pid*BN+i.val)%18446744073709551616=pid*BN+i.val :=
    Nat.mod_eq_of_lt (domain.row_exact_bound rowShape i)
  rcases ctx with ⟨hM,hOff,hsa,hsb,hsc,hn,hm,hmask,hap,hbp,hacc,hmem⟩
  simp [SourceCode.original, Source.whileLoop?, Kernels.source, ComputeKernel.surfaceBody,
    Random.i32ArangeSize, run, step, evalExpr, evalComputeOp, eval_natToUnsigned,
    eval_intToUnsigned, evalOp_unfold, numeric, numericLt, natLt, bop,store,
    Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, Option.bind,
    integralMod, IntegerExpr.sameShape, IntegerExpr.same_left, IntegerExpr.same_right,
    hsc,hn,hmask,hacc,WholeSource.suffixState,WholeSource.outputValues,WholeSource.storedValue,
    ofFloat,toFloat,ComputeDType.eraseDType, row, parameters,WholeSource.row,
    unsigned_mul,unsigned_nat,← Int.natCast_emod, hrow,-Int.ofNat_toNat]
  congr 12 <;> (first | rfl | (funext i; try simp [bop,Broadcast.leftIndex,
    Broadcast.rightIndex,unsigned_mul,unsigned_nat,← Int.natCast_emod,hrow,-Int.ofNat_toNat]))

  all_goals
    try simp only [← Int.natCast_mul,← Int.natCast_add,unsigned_nat,hrow]
    try rfl

/-- Actual full-body execution, all active output words, and the frame.
The accumulated graph is the already independently specified ordered dot
product; this theorem connects it to the raw while source by execution. -/
theorem tile_result {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (s : State α) (domain : SourceDomain N M BN BM (s.pids 0)) :
    ∃ t, exec A (Kernels.source N M BN BM rowShape columnShape) s = some t ∧
      (∀ i : Fin BN, row BN (s.pids 0) i.val<N →
        t.mem "C" (row BN (s.pids 0) i.val) = .mk .fp32
          (WholeSource.tileValue A (parameters N M BN BM) (s.pids 0) s.mem i)) ∧
      (∀ (r : RegionName) (o : Nat), (r≠"C" ∨ ∀ i : Fin BN, row BN (s.pids 0) i.val<N →
        o≠row BN (s.pids 0) i.val) → t.mem r o=s.mem r o) := by
  let p := parameters N M BN BM
  obtain ⟨last,middle,loopRan,ctx⟩ := SourceLoop.runs A N M BN BM (s.pids 0) 0
    rowShape columnShape s.mem (fun _ => A.literal (some .fp32) .real 0) _ domain
    (SourceBody.prefix_context A N M BN BM s)
  let acc := WholeSource.accumulated A p (s.pids 0) s.mem 0
    (fun _ => A.literal (some .fp32) .real 0)
  refine ⟨WholeSource.suffixState A p (s.pids 0) acc middle,?_,?_,?_⟩
  · change run A (Kernels.source N M BN BM rowShape columnShape).surfaceBody s=_
    rw [SourceCode.original_decomposition,run_append,run_append,SourcePrelude.runs A N M BN BM rowShape columnShape s domain]
    simp only [Option.bind_some,List.singleton_append,run,step,Option.bind_assoc,Option.bind_fun_some]
    unfold SourceLoop.condition at loopRan
    rw [loopRan]
    exact runs A N M BN BM (s.pids 0) last rowShape columnShape s.mem acc middle domain ctx
  · intro i hi
    simpa only [p,parameters,WholeSource.row,row,Nat.mul_one] using
      WholeSource.suffix_value A p (s.pids 0) rfl acc middle i hi
  · intro r o outside
    rw [WholeSource.suffix_frame]
    · exact congrFun (congrFun ctx.memory_eq r) o
    · simpa only [p,parameters,WholeSource.row,row,Nat.mul_one] using outside

end
end VeriTile.Bench.Repairs.MatrixVectorMultip.SourceOutput
