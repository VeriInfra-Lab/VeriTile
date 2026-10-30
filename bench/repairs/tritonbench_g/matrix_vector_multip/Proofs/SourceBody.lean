import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourcePrelude
import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeSource

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.SourceBody
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
open SourcePrelude
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def parameters (N M BN BM : Nat) : Typed.Parameters :=
  ⟨"A", "B", "C", N, M, M, 1, 1, 1, BN, BM⟩
abbrev Acc (α : Type) (BN BM : Nat) := Values α .real [BN,BM]

structure Context {α : Type} (N M BN BM pid off : Nat)
    (memory : RegionName → Nat → Cell α) (acc : Acc α BN BM) (s : State α) : Prop where
  columns : s.regs .nat [] "M" = some (fun _ => M)
  counter : s.regs .nat [] "m" = some (fun _ => off)
  strideA : s.regs .nat [] "stride_am" = some (fun _ => 1)
  strideB : s.regs .nat [] "stride_bm" = some (fun _ => 1)
  strideC : s.regs .nat [] "stride_cn" = some (fun _ => 1)
  rows : s.regs .nat [BN,1] "offset_n" = some (fun i => row BN pid i.1.val)
  lanes : s.regs .nat [1,BM] "offset_m" = some (fun i => i.2.1.val)
  rowMask : s.regs .bool [BN,1] "n_mask" = some (fun i => decide (row BN pid i.1.val<N))
  pointerA : s.regs .ptr [BN,BM] "A_ptrs" =
    some (fun i : TileIndex [BN,BM] => (("A" : RegionName),matrixOffset M BN pid off i.1.val i.2.1.val))
  pointerB : s.regs .ptr [1,BM] "B_ptrs" = some (fun i : TileIndex [1,BM] => (("B" : RegionName),i.2.1.val+off))
  accumulator : s.regs .real [BN,BM] "acc" = some acc
  memory_eq : s.mem = memory

def aChunk {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM pid off : Nat)
    (memory : RegionName → Nat → Cell α) : Acc α BN BM :=
  fun i => WholeSource.load A memory "A" (matrixOffset M BN pid off i.1.val i.2.1.val)
    (row BN pid i.1.val<N ∧ off+i.2.1.val<M)

theorem aChunk_eq {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM pid off : Nat)
    (memory : RegionName → Nat → Cell α) (domain : SourceDomain N M BN BM pid) :
    aChunk A N M BN BM pid off memory =
      WholeSource.aChunk A (parameters N M BN BM) pid off memory := by
  funext i
  unfold aChunk WholeSource.aChunk WholeSource.load
  simp only [parameters, WholeSource.row, row, WholeSource.aOffset, Nat.mul_one]
  by_cases active : pid*BN+i.1.val<N ∧ off+i.2.1.val<M
  · have bound := domain.matrixOffsets _ _ active.1 active.2
    have hbase : (pid*BN+i.1.val)*M < 18446744073709551616 := by omega
    simp only [active, if_true, matrixOffset, row, Nat.mod_eq_of_lt hbase]
  · simp only [active, if_false]

def state {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM pid off : Nat)
    (memory : RegionName → Nat → Cell α) (acc : Acc α BN BM) (s : State α) : State α :=
  let p := parameters N M BN BM
  let s := s.setReg "m_mask" .bool [1,BM] (fun i => decide (off+i.2.1.val<M))
  let s := s.setReg "a" .real [BN,BM] (WholeSource.aChunk A p pid off memory)
  let s := s.setReg "b" .real [1,BM] (WholeSource.bChunk A p off memory)
  let s := s.setReg "acc" .real [BN,BM] (WholeSource.nextAcc A p pid off memory acc)
  let s := s.setReg "A_ptrs" .ptr [BN,BM]
    (fun i => ("A",matrixOffset M BN pid (off+BM) i.1.val i.2.1.val))
  let s := s.setReg "B_ptrs" .ptr [1,BM] (fun i => ("B",i.2.1.val+(off+BM)))
  s.setReg "m" .nat [] (fun _ => off+BM)

theorem runs {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM pid off : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (memory : RegionName → Nat → Cell α) (acc : Acc α BN BM) (s : State α)
    (domain : SourceDomain N M BN BM pid) (active : off<M)
    (ctx : Context N M BN BM pid off memory acc s) :
    run A (SourceCode.original N M BN BM rowShape columnShape).body s =
      some (state A N M BN BM pid off memory acc s) := by
  have hblock : BM%18446744073709551616=BM := Nat.mod_eq_of_lt (by have := columnShape.2.1; omega)
  have hcounter : (off+BM)%18446744073709551616=off+BM := Nat.mod_eq_of_lt (by
    have := columnShape.2.1; have := domain.columns; omega)
  have hlane (i : Fin BM) : (off+i.val)%18446744073709551616=off+i.val :=
    Nat.mod_eq_of_lt (domain.active_column_bound columnShape off active i)
  have hz : (0.0 : ℝ) = 0 := by norm_num
  have ha := aChunk_eq A N M BN BM pid off memory domain
  rcases ctx with ⟨hM,hOff,hsa,hsb,hsc,hn,hm,hmask,hap,hbp,hacc,hmem⟩
  simp [SourceCode.original, Source.whileLoop?, Kernels.source, ComputeKernel.surfaceBody,
    Random.i32ArangeSize, run, step, evalExpr, evalComputeOp, eval_natToUnsigned,
    eval_intToUnsigned, evalOp_unfold, numeric, numericLt, natLt, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, Option.bind,
    integralMod, IntegerExpr.sameShape, IntegerExpr.same_left, IntegerExpr.same_right,
    hM,hOff,hsa,hsb,hsc,hn,hm,hmask,hap,hbp,hacc,hmem,
    state, WholeSource.nextAcc, WholeSource.bChunk, WholeSource.load,
    ofFloat,toFloat,ComputeDType.eraseDType,
    unsigned_add,unsigned_mul,unsigned_nat,← Int.natCast_emod, hblock,hcounter,hlane,
    -Int.ofNat_toNat]
  congr 18 <;> (first | rfl | (funext i; simp [bop, Broadcast.leftIndex,
    Broadcast.rightIndex, unsigned_add,unsigned_mul,unsigned_nat,← Int.natCast_emod,
    hblock,hcounter,hlane, parameters, WholeSource.bOffset, matrixOffset,
    Nat.add_assoc, -Int.ofNat_toNat]))
  all_goals
    try simp only [WholeSource.nextAcc]
    try rw [← ha]
    try simp only [aChunk,WholeSource.bChunk,WholeSource.load,
      parameters,WholeSource.bOffset,Nat.mul_one,matrixOffset,Nat.add_assoc]
    try simp only [hz]
    try rfl
    try (simpa only [aChunk,WholeSource.load,matrixOffset,Nat.add_assoc,hz]
      using congrFun ha i)
    try (congr 2 <;> (first | rfl | (simpa only [aChunk,WholeSource.load,matrixOffset,Nat.add_assoc,hz] using congrFun ha i)))
    try (rcases i with ⟨i,j,⟨⟩⟩; rfl)

theorem preserves {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM pid off : Nat)
    (memory : RegionName → Nat → Cell α) (acc : Acc α BN BM) (s : State α)
    (ctx : Context N M BN BM pid off memory acc s) :
    Context N M BN BM pid (off+BM) memory
      (WholeSource.nextAcc A (parameters N M BN BM) pid off memory acc)
      (state A N M BN BM pid off memory acc s) := by
  rcases ctx with ⟨hM,hOff,hsa,hsb,hsc,hn,hm,hmask,hap,hbp,hacc,hmem⟩
  constructor <;> simp [state,hM,hsa,hsb,hsc,hn,hm,hmask,hmem]

theorem prefix_context {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM : Nat)
    (s : State α) :
    Context N M BN BM (s.pids 0) 0 s.mem (fun _ => A.literal (some .fp32) .real 0)
      (SourcePrelude.state A N M BN BM s) := by
  constructor <;> simp [SourcePrelude.state]

end
end VeriTile.Bench.Repairs.MatrixVectorMultip.SourceBody
