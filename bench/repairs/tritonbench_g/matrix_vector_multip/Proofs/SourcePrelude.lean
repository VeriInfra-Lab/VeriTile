import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourceCode
import bench.repairs.tritonbench_g.matrix_vector_multip.SourceDomain
import VeriTile.Triton.Float.UnsignedCast

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.SourcePrelude
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false

def row (BN pid lane : Nat) : Nat := pid*BN+lane
def matrixOffset (M BN pid off i j : Nat) : Nat := (row BN pid i*M)%18446744073709551616+j+off

def state {α : Type} (A : Algebra α) (N M BN BM : Nat) (s : State α) : State α :=
  let pid := s.pids 0
  let s := s.setReg "N" .nat [] (fun _ => N)
  let s := s.setReg "M" .nat [] (fun _ => M)
  let s := s.setReg "stride_an" .nat [] (fun _ => M)
  let s := s.setReg "stride_am" .nat [] (fun _ => 1)
  let s := s.setReg "stride_bm" .nat [] (fun _ => 1)
  let s := s.setReg "stride_cn" .nat [] (fun _ => 1)
  let s := s.setReg "pid" .nat [] (fun _ => pid)
  let s := s.setReg "offset_n" .nat [BN,1] (fun i => row BN pid i.1.val)
  let s := s.setReg "offset_m" .nat [1,BM] (fun i => i.2.1.val)
  let s := s.setReg "n_mask" .bool [BN,1] (fun i => decide (row BN pid i.1.val<N))
  let s := s.setReg "A_ptrs" .ptr [BN,BM]
    (fun i => ("A",matrixOffset M BN pid 0 i.1.val i.2.1.val))
  let s := s.setReg "B_ptrs" .ptr [1,BM] (fun i => ("B",i.2.1.val))
  let s := s.setReg "acc" .real [BN,BM] (fun _ => A.literal (some .fp32) .real 0)
  s.setReg "m" .nat [] (fun _ => 0)

theorem unsigned_nat (n : Nat) : ((n : Int) % (18446744073709551616 : Int)).toNat = n % 18446744073709551616 := by
  change ((n : Int) % ((18446744073709551616 : Nat) : Int)).toNat = _
  rw [← Int.natCast_emod, Int.toNat_natCast]

theorem unsigned_add (a b : Nat) :
    (((a : Int) + (b : Int)) % (18446744073709551616 : Int)).toNat = (a+b)%18446744073709551616 := by
  simpa only [Int.natCast_add] using unsigned_nat (a+b)

theorem unsigned_mul (a b : Nat) :
    (((a : Int) * (b : Int)) % (18446744073709551616 : Int)).toNat = (a*b)%18446744073709551616 := by
  simpa only [Int.natCast_mul] using unsigned_nat (a*b)

theorem runs {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (s : State α) (domain : SourceDomain N M BN BM (s.pids 0)) :
    run A (SourceCode.original N M BN BM rowShape columnShape).before s =
      some (state A N M BN BM s) := by
  have hn : N % 18446744073709551616 = N := Nat.mod_eq_of_lt (by have := domain.rows; omega)
  have hm : M % 18446744073709551616 = M := Nat.mod_eq_of_lt (by have := domain.columns; omega)
  have hp : s.pids 0 % 18446744073709551616 = s.pids 0 := Nat.mod_eq_of_lt (by have := domain.program; omega)
  have hpInt : (s.pids 0 : Int) % 18446744073709551616 = (s.pids 0 : Int) := by
    exact_mod_cast hp
  have hprod : s.pids 0*BN % 18446744073709551616 = s.pids 0*BN := by
    apply Nat.mod_eq_of_lt
    simpa using domain.row_exact_bound rowShape ⟨0,rowShape.1⟩
  have hnLane (i : Fin BN) : i.val % 18446744073709551616 = i.val := by
    apply Nat.mod_eq_of_lt
    have := rowShape.2.1
    have := i.isLt
    omega
  have hmLane (i : Fin BM) : i.val % 18446744073709551616 = i.val := by
    apply Nat.mod_eq_of_lt
    have := columnShape.2.1
    have := i.isLt
    omega
  have hrow (i : Fin BN) : (s.pids 0*BN+i.val) % 18446744073709551616 = s.pids 0*BN+i.val :=
    Nat.mod_eq_of_lt (domain.row_exact_bound rowShape i)
  simp [SourceCode.original, Source.whileLoop?, Kernels.source, ComputeKernel.surfaceBody,
    Random.i32ArangeSize, run, step, evalExpr, evalComputeOp, eval_natToUnsigned,
    eval_intToUnsigned, evalOp_unfold, numeric, numericLt, natLt, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, Option.bind,
    bind, pure, integralMod, IntegerExpr.sameShape,
    IntegerExpr.same_left, IntegerExpr.same_right,
    Option.map_some, Option.get_some, Option.bind_some, Option.bind_none,
    State.setReg_same, State.setReg_ne_name, State.setReg_pids,
    state, row, matrixOffset, hn, hm, hp, hpInt, hprod, hnLane, hmLane, hrow,
    unsigned_add, unsigned_mul, unsigned_nat, ← Int.natCast_emod, -Int.ofNat_toNat]
  try dsimp only [bop, Broadcast.leftIndex, Broadcast.rightIndex]
  try simp only [unsigned_mul,unsigned_add,unsigned_nat,hn,hm,hp,hpInt,hprod,hnLane,hmLane,hrow,
    Nat.zero_add,Nat.add_zero,Nat.mul_one,Nat.one_mul]
  congr 16 <;> (first | rfl | (funext i; try simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex,← Int.natCast_emod,unsigned_mul,unsigned_add,unsigned_nat,
    hn,hm,hp,hpInt,hprod,hnLane,hmLane,hrow,Nat.zero_add,Nat.add_zero,Nat.mul_one,Nat.one_mul]))
  all_goals
    try simp only [← Int.natCast_mul,← Int.natCast_add,unsigned_nat]
    try rfl

end
end VeriTile.Bench.Repairs.MatrixVectorMultip.SourcePrelude
