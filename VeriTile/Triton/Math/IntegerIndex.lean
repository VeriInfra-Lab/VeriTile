import VeriTile.Triton.Core.Integer

/-! Exact transport between signed source indices and nonnegative logical
coordinates. These identities do not erase source casts, bound machine words,
or impose any floating-point primitive law. -/
namespace VeriTile.Triton.IntegerIndex

@[simp] theorem nat_div (x b : Nat) :
    (if (x : Int)<0 then -(-(x : Int)/(b : Int)) else (x : Int)/(b : Int))=((x/b : Nat) : Int) := by
  simp only [not_lt.mpr (Int.natCast_nonneg x),↓reduceIte,Int.natCast_ediv]

@[simp] theorem nat_mod (x b : Nat) :
    (x : Int)-(if (x : Int)<0 then -((b : Int)*(-(x : Int)/(b : Int)))
      else (b : Int)*((x : Int)/(b : Int)))=((x%b : Nat) : Int) := by
  simp only [not_lt.mpr (Int.natCast_nonneg x),↓reduceIte]
  rw [←Int.emod_def,←Int.natCast_emod]

@[simp] theorem nat_cdiv (x b : Nat) :
    (if (x : Int)+((b : Int)-1)<0 then
      -(-((x : Int)+((b : Int)-1))/(b : Int))
      else ((x : Int)+((b : Int)-1))/(b : Int))=(((x+b-1)/b : Nat) : Int) := by
  by_cases hz : b=0
  · simp [hz]
  · have hn : 1≤b := by omega
    have ha : (x : Int)+((b : Int)-1)=((x+b-1 : Nat) : Int) := by omega
    rw [ha]
    exact nat_div (x+b-1) b

@[simp] theorem nat_min (x y : Nat) :
    (if (x : Int)<(y : Int) then (x : Int) else (y : Int))=((min x y : Nat) : Int) := by
  split_ifs with h
  · rw [Nat.min_eq_left (by omega)]
  · rw [Nat.min_eq_right (by omega)]

@[simp] theorem nat_tail (i n cur : Nat) :
    decide ((i : Int)<(n : Int)-(cur : Int))=decide (cur+i<n) := by
  by_cases h : (i : Int)<(n : Int)-(cur : Int)
  · have hn : cur+i<n := by omega
    simp [h,hn]
  · have hn : ¬cur+i<n := by omega
    simp [h,hn]

end VeriTile.Triton.IntegerIndex
