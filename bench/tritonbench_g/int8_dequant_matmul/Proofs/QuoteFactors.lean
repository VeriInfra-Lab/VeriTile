import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteGeometry
import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteModel

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
open VeriTile.Triton FP.Structural
open IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

private theorem signed_rem (a b : Nat) :
    (a:Int)-(b:Int)*((a:Int)/(b:Int))=(a%b : Nat) := by
  rw [Int.natCast_emod]
  exact (Int.emod_def (a:Int) (b:Int)).symm

theorem factorX (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (initial : State (Sum F H))
    (valid : initial.pids 0<Geometry.count (geometry s c)) :
    register (preludeExecution (floating q) q s c bias initial).val .real [c.BM,1] "x_factor"=
      (fun i=>(initial.mem "state_x" (row s c (initial.pids 0) i.1%s.M)).read .real) := by
  have pm:=congrFun (prelude_pm (floating q) q s c bias initial valid) PUnit.unit
  funext i
  simp [register,preludeExecution,floating,bop,Broadcast.leftIndex,Broadcast.rightIndex] at pm ⊢
  simp only [←neg_mul,←ite_mul] at pm ⊢
  rw [pm]
  have nr : ¬((blockM s c (initial.pids 0):Int)*c.BM+i.1.val)<0 := not_lt.mpr (by positivity)
  simp only [if_neg nr]
  have remEq:=signed_rem (row s c (initial.pids 0) i.1) s.M
  simp only [row,Nat.cast_add,Nat.cast_mul] at remEq
  rw [remEq]
  rfl

theorem factorW (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (initial : State (Sum F H))
    (valid : initial.pids 0<Geometry.count (geometry s c)) :
    register (preludeExecution (floating q) q s c bias initial).val .real [1,c.BN] "w_factor"=
      (fun i=>(initial.mem "state_w" (col s c (initial.pids 0) i.2.1%s.N)).read .real) := by
  have pn:=congrFun (prelude_pn (floating q) q s c bias initial valid) PUnit.unit
  funext i
  simp [register,preludeExecution,floating,bop,Broadcast.leftIndex,Broadcast.rightIndex] at pn ⊢
  simp only [←neg_mul,←ite_mul] at pn ⊢
  rw [pn]
  have nr : ¬((blockN s c (initial.pids 0):Int)*c.BN+i.2.1.val)<0 := not_lt.mpr (by positivity)
  simp only [if_neg nr]
  have remEq:=signed_rem (col s c (initial.pids 0) i.2.1) s.N
  simp only [col,Nat.cast_add,Nat.cast_mul] at remEq
  rw [remEq]
  rfl

#axiomsClean factorX
#axiomsClean factorW
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
