import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteLoop

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
open VeriTile.Triton FP.Structural
open IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

private theorem signed_min (a b : Nat) :
    (if (a:Int)<b then (a:Int) else b)=(min a b : Nat) := by
  by_cases h : a<b
  · simp [h,Int.ofNat_lt, Nat.min_eq_left (by omega : a≤b)]
  · simp [h,Int.ofNat_lt,Nat.min_eq_right (by omega : b≤a)]

private theorem signed_rem (a b : Nat) :
    (a:Int)-(b:Int)*((a:Int)/(b:Int))=(a%b : Nat) := by
  rw [Int.natCast_emod]
  exact (Int.emod_def (a:Int) (b:Int)).symm

def blockM (s : Shape) (c : Config) (pid : Nat) := Geometry.blockRow (geometry s c) pid
def blockN (s : Shape) (c : Config) (pid : Nat) := Geometry.blockColumn (geometry s c) pid

theorem prelude_pm {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (initial : State α)
    (valid : initial.pids 0<Geometry.count (geometry s c)) :
    register (preludeExecution M q s c bias initial).val .int [] "pid_m"=
      (fun _=>(blockM s c (initial.pids 0):Int)) := by
  let gm:=(s.M+c.BM-1)/c.BM
  let gn:=(s.N+c.BN-1)/c.BN
  have npos : 0<gn := by
    change initial.pids 0<gm*gn at valid
    nlinarith
  have firstBound : GroupedGrid.first 8 gn (initial.pids 0)≤gm :=
    (GroupedGrid.facts gm gn 8 (initial.pids 0) npos (by decide) valid).1.le
  have minus := Int.ofNat_sub firstBound
  have nonnegative : ¬(initial.pids 0:Int)<0 := by omega
  have groupSize :
      (if (gm:Int)-(GroupedGrid.first 8 gn (initial.pids 0):Int)<8 then
        (gm:Int)-(GroupedGrid.first 8 gn (initial.pids 0):Int) else 8)=
          (GroupedGrid.size gm gn 8 (initial.pids 0):Int) := by
    rw [←minus]
    exact signed_min _ 8
  funext i
  simp [register,preludeExecution,bop,Broadcast.leftIndex,Broadcast.rightIndex,
    nonnegative]
  simp only [←ite_mul]
  change (GroupedGrid.first 8 gn (initial.pids 0):Int)+
      ((initial.pids 0:Int)-
        (if (gm:Int)-(GroupedGrid.first 8 gn (initial.pids 0):Int)<8 then
          (gm:Int)-(GroupedGrid.first 8 gn (initial.pids 0):Int) else 8)*
        ((initial.pids 0:Int)/(if (gm:Int)-(GroupedGrid.first 8 gn (initial.pids 0):Int)<8 then
          (gm:Int)-(GroupedGrid.first 8 gn (initial.pids 0):Int) else 8)))=_
  rw [groupSize,signed_rem]
  simp only [blockM,geometry,Geometry.blockRow,Geometry.gridRows,Geometry.gridColumns,
    GroupedGrid.row,Int.natCast_add]
  rfl

theorem prelude_pn {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (initial : State α)
    (valid : initial.pids 0<Geometry.count (geometry s c)) :
    register (preludeExecution M q s c bias initial).val .int [] "pid_n"=
      (fun _=>(blockN s c (initial.pids 0):Int)) := by
  let gm:=(s.M+c.BM-1)/c.BM
  let gn:=(s.N+c.BN-1)/c.BN
  have npos : 0<gn := by
    change initial.pids 0<gm*gn at valid
    nlinarith
  have firstBound : GroupedGrid.first 8 gn (initial.pids 0)≤gm :=
    (GroupedGrid.facts gm gn 8 (initial.pids 0) npos (by decide) valid).1.le
  have minus := Int.ofNat_sub firstBound
  have nonnegative : ¬(initial.pids 0:Int)<0 := by omega
  have groupSize :
      (if (gm:Int)-(GroupedGrid.first 8 gn (initial.pids 0):Int)<8 then
        (gm:Int)-(GroupedGrid.first 8 gn (initial.pids 0):Int) else 8)=
          (GroupedGrid.size gm gn 8 (initial.pids 0):Int) := by
    rw [←minus]
    exact signed_min _ 8
  funext i
  simp [register,preludeExecution,bop,Broadcast.leftIndex,Broadcast.rightIndex,nonnegative]
  change (if (initial.pids 0:Int)<(8*gn:Int)*((initial.pids 0:Int)/(8*gn:Int)) then
      -(((8*gn:Int)*((initial.pids 0:Int)/(8*gn:Int))-(initial.pids 0:Int))/
        (if (gm:Int)-(GroupedGrid.first 8 gn (initial.pids 0):Int)<8 then
          (gm:Int)-(GroupedGrid.first 8 gn (initial.pids 0):Int) else 8))
    else ((initial.pids 0:Int)-(8*gn:Int)*((initial.pids 0:Int)/(8*gn:Int)))/
        (if (gm:Int)-(GroupedGrid.first 8 gn (initial.pids 0):Int)<8 then
          (gm:Int)-(GroupedGrid.first 8 gn (initial.pids 0):Int) else 8))=_
  have nonnegativeRem : ¬(initial.pids 0:Int)<(8*gn:Int)*((initial.pids 0:Int)/(8*gn:Int)) := by
    have h:=signed_rem (initial.pids 0) (8*gn)
    simp only [Nat.cast_mul,Nat.cast_ofNat] at h
    omega
  rw [if_neg nonnegativeRem,groupSize]
  have remEq:=signed_rem (initial.pids 0) (8*gn)
  simp only [Nat.cast_mul,Nat.cast_ofNat] at remEq
  rw [remEq]
  rfl

#axiomsClean prelude_pn

#axiomsClean prelude_pm

def row (s : Shape) (c : Config) (pid : Nat) (i : Fin c.BM) := blockM s c pid*c.BM+i.val
def col (s : Shape) (c : Config) (pid : Nat) (i : Fin c.BN) := blockN s c pid*c.BN+i.val

theorem prelude_rm {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (initial : State α)
    (valid : initial.pids 0<Geometry.count (geometry s c)) :
    register (preludeExecution M q s c bias initial).val .int [c.BM] "rm"=
      (fun i=>(row s c (initial.pids 0) i.1:Int)) := by
  have pm:=congrFun (prelude_pm M q s c bias initial valid) PUnit.unit
  funext i
  simp [register,preludeExecution,bop,Broadcast.leftIndex,Broadcast.rightIndex] at pm ⊢
  rw [pm]
  simp [row]

theorem prelude_rn {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (initial : State α)
    (valid : initial.pids 0<Geometry.count (geometry s c)) :
    register (preludeExecution M q s c bias initial).val .int [c.BN] "rn"=
      (fun i=>(col s c (initial.pids 0) i.1:Int)) := by
  have pn:=congrFun (prelude_pn M q s c bias initial valid) PUnit.unit
  funext i
  simp [register,preludeExecution,bop,Broadcast.leftIndex,Broadcast.rightIndex] at pn ⊢
  simp only [←neg_mul,←ite_mul] at pn ⊢
  rw [pn]
  simp [col]

theorem prelude_rk {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (initial : State α) :
    register (preludeExecution M q s c bias initial).val .int [c.BK] "rk"=
      (fun i=>(initial.pids 1*c.BK+i.1.val:Int)) := by
  funext i
  simp [register,preludeExecution,bop,Broadcast.leftIndex,Broadcast.rightIndex]

#axiomsClean prelude_rm
#axiomsClean prelude_rn
#axiomsClean prelude_rk

theorem prelude_a {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (initial : State α)
    (valid : initial.pids 0<Geometry.count (geometry s c)) :
    register (preludeExecution M q s c bias initial).val .ptr [c.BM,c.BK] "A"=
      (fun i : TileIndex [c.BM,c.BK]=>(("A" : RegionName),(row s c (initial.pids 0) i.1%s.M)*s.K+initial.pids 1*c.BK+i.2.1.val)) := by
  have pm:=congrFun (prelude_pm M q s c bias initial valid) PUnit.unit
  funext i
  simp [register,preludeExecution,bop,Broadcast.leftIndex,Broadcast.rightIndex] at pm ⊢
  simp only [←neg_mul,←ite_mul] at pm ⊢
  rw [pm]
  have nr : ¬((blockM s c (initial.pids 0):Int)*c.BM+i.1.val)<0 := not_lt.mpr (by positivity)
  simp only [if_neg nr]
  have remEq:=signed_rem (row s c (initial.pids 0) i.1) s.M
  simp only [row,Nat.cast_add,Nat.cast_mul] at remEq
  rw [remEq]
  simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
  simp [row,Nat.add_assoc]

theorem prelude_b {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (initial : State α)
    (valid : initial.pids 0<Geometry.count (geometry s c)) :
    register (preludeExecution M q s c bias initial).val .ptr [c.BK,c.BN] "B"=
      (fun i : TileIndex [c.BK,c.BN]=>(("B" : RegionName),(initial.pids 1*c.BK+i.1.val)*s.N+col s c (initial.pids 0) i.2.1%s.N)) := by
  have pn:=congrFun (prelude_pn M q s c bias initial valid) PUnit.unit
  funext i
  simp [register,preludeExecution,bop,Broadcast.leftIndex,Broadcast.rightIndex] at pn ⊢
  simp only [←neg_mul,←ite_mul] at pn ⊢
  rw [pn]
  have nr : ¬((blockN s c (initial.pids 0):Int)*c.BN+i.2.1.val)<0 := not_lt.mpr (by positivity)
  simp only [if_neg nr]
  have remEq:=signed_rem (col s c (initial.pids 0) i.2.1) s.N
  simp only [col,Nat.cast_add,Nat.cast_mul] at remEq
  rw [remEq]
  norm_cast

#axiomsClean prelude_a
#axiomsClean prelude_b
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
