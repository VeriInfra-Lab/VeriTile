import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourceOutput
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.SourceSafety
open VeriTile.Triton FP.Structural SourceBody SourcePrelude
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem prefix_safe {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList A bounds (SourceCode.original N M BN BM rowShape columnShape).before s := by
  simp [SourceCode.original,Source.whileLoop?,Kernels.source,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,numericLt,natLt,bop,Region.cast,Option.bind]

theorem body_safe {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM pid off : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (memory : RegionName → Nat → Cell α) (acc : Acc α BN BM) (s : State α)
    (domain : SourceDomain N M BN BM pid) (active : off<M)
    (ctx : Context N M BN BM pid off memory acc s) (bounds : RegionName → Nat)
    (matrix : ∀ i j, i<N → j<M → i*M+j<bounds "A")
    (vector : ∀ j, j<M → j<bounds "B") :
    Safety.traceSafeList A bounds (SourceCode.original N M BN BM rowShape columnShape).body s := by
  have hlane (i : Fin BM) : (off+i.val)%18446744073709551616=off+i.val :=
    Nat.mod_eq_of_lt (domain.active_column_bound columnShape off active i)
  rcases ctx with ⟨hM,hOff,hsa,hsb,hsc,hn,hm,hmask,hap,hbp,hacc,hmem⟩
  simp [SourceCode.original,Source.whileLoop?,Kernels.source,ComputeKernel.surfaceBody,
    Random.i32ArangeSize,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,
    bop,Region.cast,Option.bind,store,hM,hOff,hsa,hsb,hsc,hn,hm,hmask,hap,hbp,hacc,hmem,
    ofFloat,toFloat,ComputeDType.eraseDType,integralMod,IntegerExpr.sameShape,
    Broadcast.leftIndex,Broadcast.rightIndex,unsigned_add,unsigned_mul,unsigned_nat,
    ← Int.natCast_emod,hlane,-Int.ofNat_toNat]
  constructor
  · intro i j hi hj
    have bound := domain.matrixOffsets _ _ hi hj
    have base : row BN pid i.val*M<18446744073709551616 := by omega
    simpa [matrixOffset,Nat.mod_eq_of_lt base,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
      using matrix (row BN pid i.val) (off+j.val) hi hj
  · intro j hj
    simpa [Nat.add_comm] using vector (off+j.val) hj

theorem suffix_safe {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM pid off : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (memory : RegionName → Nat → Cell α) (acc : Acc α BN BM) (s : State α)
    (domain : SourceDomain N M BN BM pid)
    (ctx : Context N M BN BM pid off memory acc s) (bounds : RegionName → Nat)
    (output : ∀ i : Fin BN, row BN pid i.val<N → row BN pid i.val<bounds "C") :
    Safety.traceSafeList A bounds (SourceCode.original N M BN BM rowShape columnShape).after s := by
  have hrow (i : Fin BN) : (pid*BN+i.val)%18446744073709551616=pid*BN+i.val :=
    Nat.mod_eq_of_lt (domain.row_exact_bound rowShape i)
  rcases ctx with ⟨hM,hOff,hsa,hsb,hsc,hn,hm,hmask,hap,hbp,hacc,hmem⟩
  simp [SourceCode.original,Source.whileLoop?,Kernels.source,ComputeKernel.surfaceBody,
    Random.i32ArangeSize,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,
    bop,Region.cast,Option.bind,store,hsc,hn,hmask,hacc,
    ofFloat,toFloat,ComputeDType.eraseDType,integralMod,IntegerExpr.sameShape,
    Broadcast.leftIndex,Broadcast.rightIndex,unsigned_add,unsigned_mul,unsigned_nat,
    ← Int.natCast_emod,hrow,row,-Int.ofNat_toNat]
  simpa only [← Int.natCast_mul,← Int.natCast_add,unsigned_nat,hrow,row] using output

theorem loop_safe {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM pid off : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (memory : RegionName → Nat → Cell α) (acc : Acc α BN BM) (s : State α)
    (domain : SourceDomain N M BN BM pid)
    (ctx : Context N M BN BM pid off memory acc s) (bounds : RegionName → Nat)
    (matrix : ∀ i j, i<N → j<M → i*M+j<bounds "A")
    (vector : ∀ j, j<M → j<bounds "B") :
    Safety.stepSafe A bounds (.whileLoop
      (SourceCode.original N M BN BM rowShape columnShape).condition
      (SourceCode.original N M BN BM rowShape columnShape).body) s := by
  rw [Safety.stepSafe_unfold]
  intro current reached
  have invariant : ∃ cursor carry, Context N M BN BM pid cursor memory carry current := by
    apply Control.While.reaches_preserves
      (fun t => ∃ cursor carry, Context N M BN BM pid cursor memory carry t) ?_ reached
    · exact ⟨off,acc,ctx⟩
    · intro before after holds guard ran
      obtain ⟨cursor,carry,hctx⟩ := holds
      change SourceLoop.condition A N M BN BM rowShape columnShape before = some Bool.true at guard
      rw [SourceLoop.condition_value A N M BN BM pid cursor rowShape columnShape memory carry before hctx] at guard
      have active : cursor<M := by simpa using guard
      rw [SourceBody.runs A N M BN BM pid cursor rowShape columnShape memory carry before domain active hctx] at ran
      cases ran
      exact ⟨cursor+BM,_,SourceBody.preserves A N M BN BM pid cursor memory carry before hctx⟩
  obtain ⟨cursor,carry,hctx⟩ := invariant
  constructor
  · simp [SourceCode.original,Source.whileLoop?,Kernels.source,ComputeKernel.surfaceBody,
      Safety.safeOp_unfold]
  · intro guard
    change SourceLoop.condition A N M BN BM rowShape columnShape current = some Bool.true at guard
    rw [SourceLoop.condition_value A N M BN BM pid cursor rowShape columnShape memory carry current hctx] at guard
    exact body_safe A N M BN BM pid cursor rowShape columnShape memory carry current domain
      (by simpa using guard) hctx bounds matrix vector

theorem source_safe {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (s : State α) (domain : SourceDomain N M BN BM (s.pids 0)) (bounds : RegionName → Nat)
    (matrix : ∀ i j, i<N → j<M → i*M+j<bounds "A")
    (vector : ∀ j, j<M → j<bounds "B")
    (output : ∀ i : Fin BN, row BN (s.pids 0) i.val<N → row BN (s.pids 0) i.val<bounds "C") :
    TraceSafe A bounds (Kernels.source N M BN BM rowShape columnShape) s := by
  unfold TraceSafe
  rw [SourceCode.original_decomposition,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prefix_safe A N M BN BM rowShape columnShape bounds s,?_⟩
  intro middle hprefix
  rw [SourcePrelude.runs A N M BN BM rowShape columnShape s domain] at hprefix
  cases hprefix
  rw [List.singleton_append,Safety.traceSafeList_cons]
  refine ⟨loop_safe A N M BN BM (s.pids 0) 0 rowShape columnShape s.mem _ _ domain
    (SourceBody.prefix_context A N M BN BM s) bounds matrix vector,?_⟩
  intro final loopRan
  obtain ⟨last,result,ran,ctx⟩ := SourceLoop.runs A N M BN BM (s.pids 0) 0
    rowShape columnShape s.mem (fun _ => A.literal (some .fp32) .real 0) _ domain
    (SourceBody.prefix_context A N M BN BM s)
  rw [step] at loopRan
  unfold SourceLoop.condition at ran
  rw [ran] at loopRan
  cases loopRan
  exact suffix_safe A N M BN BM (s.pids 0) last rowShape columnShape s.mem _ _ domain ctx bounds output

end
end VeriTile.Bench.Repairs.MatrixVectorMultip.SourceSafety
