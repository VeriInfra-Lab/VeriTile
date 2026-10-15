import bench.tritonbench_g.kcache_copy_triton.HostCorrect

/-! Exhaustive symbolic singleton cases for rank4→rank3 view coefficients.
Each dimension is either exactly one or an arbitrary Nat greater than one.
The local `consume` evaluations are kernel-checked symbolic equations; they
are not finite parameter probes. The companion public address theorem admits
zero dimensions through its empty Fin domain. -/
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.Host.View
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 1000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

private theorem coefficients_1111 (sa sb sh sd st oh od : Nat)
    (_view : inferViewStrides [1,1,1,1] [sa,sb,sh,sd] [1,1,1] = some [st,oh,od]) :
    (1 = 1 ∨ sa = 1*st) ∧ (1 = 1 ∨ sb = st) ∧
    (1 = 1 ∨ sh = oh) ∧ (1 = 1 ∨ sd = od) := by
  simp

private theorem coefficients_111L (d sa sb sh sd st oh od : Nat)
    (hd : 1 < d)
    (view : inferViewStrides [1,1,1,d] [sa,sb,sh,sd] [1,1,d] = some [st,oh,od]) :
    (1 = 1 ∨ sa = 1*st) ∧ (1 = 1 ∨ sb = st) ∧
    (1 = 1 ∨ sh = oh) ∧ (d = 1 ∨ sd = od) := by
  have hp : d ≠ 0 := by positivity
  have nd : d ≠ 1 := by omega
  have q0 : 1 < d := by exact hd
  have c0 (base : Nat) : consume [d,1,1] (d) base 1 = ([], d, [base,d * (base),d * (base)]) := by
    simp [consume, q0, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  simp only [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at c0
  simp only [inferViewStrides, List.prod_cons, List.prod_nil, Nat.mul_one, Nat.one_mul,
    List.cons_ne_nil, hp, ↓reduceIte, List.zip_cons_cons, List.zip_nil_left,
    List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append,
    List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some] at view
  
  all_goals simp [inferChunks, c0, nd,
    Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at view
  all_goals simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]
  all_goals rcases view with ⟨v1,v2,v3⟩ <;> simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

private theorem coefficients_11L1 (h sa sb sh sd st oh od : Nat)
    (hh : 1 < h)
    (view : inferViewStrides [1,1,h,1] [sa,sb,sh,sd] [1,h,1] = some [st,oh,od]) :
    (1 = 1 ∨ sa = 1*st) ∧ (1 = 1 ∨ sb = st) ∧
    (h = 1 ∨ sh = oh) ∧ (1 = 1 ∨ sd = od) := by
  have hp : h ≠ 0 := by positivity
  have nh : h ≠ 1 := by omega
  have q0 : 1 < h := by exact hh
  have c0 (base : Nat) : consume [1,h,1] (h) base 1 = ([], h, [base,base,h * (base)]) := by
    simp [consume, q0, nh, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c1 (base : Nat) : consume [1,h,1] (1) base 1 = ([h,1], 1, [base]) := by
    simp [consume, nh, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c2 (base : Nat) : consume [h,1] (h) base 1 = ([], h, [base,h * (base)]) := by
    simp [consume, q0, nh, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  simp only [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at c0 c1 c2
  simp only [inferViewStrides, List.prod_cons, List.prod_nil, Nat.mul_one, Nat.one_mul,
    List.cons_ne_nil, hp, ↓reduceIte, List.zip_cons_cons, List.zip_nil_left,
    List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append,
    List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some] at view
  by_cases hs : sh = sd
  all_goals simp [inferChunks, c0, c1, c2, nh, hs,
    Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at view
  all_goals simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]
  all_goals rcases view with ⟨v1,v2,v3⟩ <;> simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

private theorem coefficients_11LL (h d sa sb sh sd st oh od : Nat)
    (hh : 1 < h)
    (hd : 1 < d)
    (view : inferViewStrides [1,1,h,d] [sa,sb,sh,sd] [1,h,d] = some [st,oh,od]) :
    (1 = 1 ∨ sa = 1*st) ∧ (1 = 1 ∨ sb = st) ∧
    (h = 1 ∨ sh = oh) ∧ (d = 1 ∨ sd = od) := by
  have hp : h * (d) ≠ 0 := by positivity
  have nh : h ≠ 1 := by omega
  have nd : d ≠ 1 := by omega
  have q0 : 1 < h * (d) := by exact (one_lt_mul_of_le_of_lt (le_of_lt hh) hd)
  have q1 : d < h * (d) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right hh (show 0 < d by positivity)
  have q2 : 1 < d := by exact hd
  have q3 : 1 < h := by exact hh
  have c0 (base : Nat) : consume [d,h,1] (h * (d)) base 1 = ([], h * (d), [base,d * (base),h * (d * (base))]) := by
    simp [consume, q0, q1, nh, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c1 (base : Nat) : consume [d,h,1] (d) base 1 = ([h,1], d, [base]) := by
    simp [consume, q2, nh, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c2 (base : Nat) : consume [h,1] (h) base 1 = ([], h, [base,h * (base)]) := by
    simp [consume, q3, nh, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  simp only [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at c0 c1 c2
  simp only [inferViewStrides, List.prod_cons, List.prod_nil, Nat.mul_one, Nat.one_mul,
    List.cons_ne_nil, hp, ↓reduceIte, List.zip_cons_cons, List.zip_nil_left,
    List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append,
    List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some] at view
  by_cases hs : sh = d * (sd)
  all_goals simp [inferChunks, c0, c1, c2, nh, nd, hs,
    Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at view
  all_goals simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]
  all_goals rcases view with ⟨v1,v2,v3⟩ <;> simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

private theorem coefficients_1L11 (b sa sb sh sd st oh od : Nat)
    (hb : 1 < b)
    (view : inferViewStrides [1,b,1,1] [sa,sb,sh,sd] [b,1,1] = some [st,oh,od]) :
    (1 = 1 ∨ sa = b*st) ∧ (b = 1 ∨ sb = st) ∧
    (1 = 1 ∨ sh = oh) ∧ (1 = 1 ∨ sd = od) := by
  have hp : b ≠ 0 := by positivity
  have nb : b ≠ 1 := by omega
  have q0 : 1 < b := by exact hb
  have c0 (base : Nat) : consume [1,1,b] (b) base 1 = ([], b, [base,base,base]) := by
    simp [consume, q0, nb, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c1 (base : Nat) : consume [1,1,b] (1) base 1 = ([b], 1, [base,base]) := by
    simp [consume, nb, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c2 (base : Nat) : consume [b] (b) base 1 = ([], b, [base]) := by
    simp [consume, q0, nb, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  simp only [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at c0 c1 c2
  simp only [inferViewStrides, List.prod_cons, List.prod_nil, Nat.mul_one, Nat.one_mul,
    List.cons_ne_nil, hp, ↓reduceIte, List.zip_cons_cons, List.zip_nil_left,
    List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append,
    List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some] at view
  by_cases hbs : sb = sd
  all_goals simp [inferChunks, c0, c1, c2, nb, hbs,
    Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at view
  all_goals simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]
  all_goals rcases view with ⟨v1,v2,v3⟩ <;> simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

private theorem coefficients_1L1L (b d sa sb sh sd st oh od : Nat)
    (hb : 1 < b)
    (hd : 1 < d)
    (view : inferViewStrides [1,b,1,d] [sa,sb,sh,sd] [b,1,d] = some [st,oh,od]) :
    (1 = 1 ∨ sa = b*st) ∧ (b = 1 ∨ sb = st) ∧
    (1 = 1 ∨ sh = oh) ∧ (d = 1 ∨ sd = od) := by
  have hp : b * (d) ≠ 0 := by positivity
  have nb : b ≠ 1 := by omega
  have nd : d ≠ 1 := by omega
  have q0 : 1 < b * (d) := by exact (one_lt_mul_of_le_of_lt (le_of_lt hb) hd)
  have q1 : d < b * (d) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right hb (show 0 < d by positivity)
  have q2 : 1 < d := by exact hd
  have q3 : 1 < b := by exact hb
  have c0 (base : Nat) : consume [d,1,b] (b * (d)) base 1 = ([], b * (d), [base,d * (base),d * (base)]) := by
    simp [consume, q0, q1, nb, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c1 (base : Nat) : consume [d,1,b] (d) base 1 = ([b], d, [base,d * (base)]) := by
    simp [consume, q2, nb, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c2 (base : Nat) : consume [b] (b) base 1 = ([], b, [base]) := by
    simp [consume, q3, nb, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  simp only [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at c0 c1 c2
  simp only [inferViewStrides, List.prod_cons, List.prod_nil, Nat.mul_one, Nat.one_mul,
    List.cons_ne_nil, hp, ↓reduceIte, List.zip_cons_cons, List.zip_nil_left,
    List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append,
    List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some] at view
  by_cases hbs : sb = d * (sd)
  all_goals simp [inferChunks, c0, c1, c2, nb, nd, hbs,
    Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at view
  all_goals simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]
  all_goals rcases view with ⟨v1,v2,v3⟩ <;> simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

private theorem coefficients_1LL1 (b h sa sb sh sd st oh od : Nat)
    (hb : 1 < b)
    (hh : 1 < h)
    (view : inferViewStrides [1,b,h,1] [sa,sb,sh,sd] [b,h,1] = some [st,oh,od]) :
    (1 = 1 ∨ sa = b*st) ∧ (b = 1 ∨ sb = st) ∧
    (h = 1 ∨ sh = oh) ∧ (1 = 1 ∨ sd = od) := by
  have hp : b * (h) ≠ 0 := by positivity
  have nb : b ≠ 1 := by omega
  have nh : h ≠ 1 := by omega
  have q0 : 1 < b * (h) := by exact (one_lt_mul_of_le_of_lt (le_of_lt hb) hh)
  have q1 : h < b * (h) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right hb (show 0 < h by positivity)
  have q2 : 1 < h := by exact hh
  have q3 : 1 < b := by exact hb
  have c0 (base : Nat) : consume [1,h,b] (b * (h)) base 1 = ([], b * (h), [base,base,h * (base)]) := by
    simp [consume, q0, q1, nb, nh, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c1 (base : Nat) : consume [1,h,b] (h) base 1 = ([b], h, [base,base]) := by
    simp [consume, q2, nb, nh, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c2 (base : Nat) : consume [b] (b) base 1 = ([], b, [base]) := by
    simp [consume, q3, nb, nh, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c3 (base : Nat) : consume [1,h,b] (1) base 1 = ([h,b], 1, [base]) := by
    simp [consume, nb, nh, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c4 (base : Nat) : consume [h,b] (b * (h)) base 1 = ([], b * (h), [base,h * (base)]) := by
    simp [consume, q0, q1, nb, nh, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c5 (base : Nat) : consume [h,b] (h) base 1 = ([b], h, [base]) := by
    simp [consume, q2, nb, nh, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  simp only [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at c0 c1 c2 c3 c4 c5
  simp only [inferViewStrides, List.prod_cons, List.prod_nil, Nat.mul_one, Nat.one_mul,
    List.cons_ne_nil, hp, ↓reduceIte, List.zip_cons_cons, List.zip_nil_left,
    List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append,
    List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some] at view
  by_cases hs : sh = sd <;> by_cases hbs : sb = h * (sh)
  all_goals simp [inferChunks, c0, c1, c2, c3, c4, c5, nb, nh, hs, hbs,
    Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at view
  all_goals simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]
  all_goals rcases view with ⟨v1,v2,v3⟩ <;> simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

private theorem coefficients_1LLL (b h d sa sb sh sd st oh od : Nat)
    (hb : 1 < b)
    (hh : 1 < h)
    (hd : 1 < d)
    (view : inferViewStrides [1,b,h,d] [sa,sb,sh,sd] [b,h,d] = some [st,oh,od]) :
    (1 = 1 ∨ sa = b*st) ∧ (b = 1 ∨ sb = st) ∧
    (h = 1 ∨ sh = oh) ∧ (d = 1 ∨ sd = od) := by
  have hp : b * (h * (d)) ≠ 0 := by positivity
  have nb : b ≠ 1 := by omega
  have nh : h ≠ 1 := by omega
  have nd : d ≠ 1 := by omega
  have q0 : 1 < b * (h * (d)) := by exact (one_lt_mul_of_le_of_lt (le_of_lt hb) (one_lt_mul_of_le_of_lt (le_of_lt hh) hd))
  have q1 : d < b * (h * (d)) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right (one_lt_mul_of_le_of_lt (le_of_lt hb) hh) (show 0 < d by positivity)
  have q2 : h * (d) < b * (h * (d)) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right hb (show 0 < h * (d) by positivity)
  have q3 : 1 < h * (d) := by exact (one_lt_mul_of_le_of_lt (le_of_lt hh) hd)
  have q4 : d < h * (d) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right hh (show 0 < d by positivity)
  have q5 : 1 < b := by exact hb
  have q6 : 1 < d := by exact hd
  have q7 : 1 < b * (h) := by exact (one_lt_mul_of_le_of_lt (le_of_lt hb) hh)
  have q8 : h < b * (h) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right hb (show 0 < h by positivity)
  have q9 : 1 < h := by exact hh
  have c0 (base : Nat) : consume [d,h,b] (b * (h * (d))) base 1 = ([], b * (h * (d)), [base,d * (base),h * (d * (base))]) := by
    simp [consume, q0, q1, q2, nb, nh, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c1 (base : Nat) : consume [d,h,b] (h * (d)) base 1 = ([b], h * (d), [base,d * (base)]) := by
    simp [consume, q3, q4, nb, nh, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c2 (base : Nat) : consume [b] (b) base 1 = ([], b, [base]) := by
    simp [consume, q5, nb, nh, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c3 (base : Nat) : consume [d,h,b] (d) base 1 = ([h,b], d, [base]) := by
    simp [consume, q6, nb, nh, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c4 (base : Nat) : consume [h,b] (b * (h)) base 1 = ([], b * (h), [base,h * (base)]) := by
    simp [consume, q7, q8, nb, nh, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c5 (base : Nat) : consume [h,b] (h) base 1 = ([b], h, [base]) := by
    simp [consume, q9, nb, nh, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  simp only [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at c0 c1 c2 c3 c4 c5
  simp only [inferViewStrides, List.prod_cons, List.prod_nil, Nat.mul_one, Nat.one_mul,
    List.cons_ne_nil, hp, ↓reduceIte, List.zip_cons_cons, List.zip_nil_left,
    List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append,
    List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some] at view
  by_cases hs : sh = d * (sd) <;> by_cases hbs : sb = h * (sh)
  all_goals simp [inferChunks, c0, c1, c2, c3, c4, c5, nb, nh, nd, hs, hbs,
    Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at view
  all_goals simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]
  all_goals rcases view with ⟨v1,v2,v3⟩ <;> simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

private theorem coefficients_L111 (a sa sb sh sd st oh od : Nat)
    (ha : 1 < a)
    (view : inferViewStrides [a,1,1,1] [sa,sb,sh,sd] [a,1,1] = some [st,oh,od]) :
    (a = 1 ∨ sa = 1*st) ∧ (1 = 1 ∨ sb = st) ∧
    (1 = 1 ∨ sh = oh) ∧ (1 = 1 ∨ sd = od) := by
  have hp : a ≠ 0 := by positivity
  have na : a ≠ 1 := by omega
  have q0 : 1 < a := by exact ha
  have c0 (base : Nat) : consume [1,1,a] (a) base 1 = ([], a, [base,base,base]) := by
    simp [consume, q0, na, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c1 (base : Nat) : consume [1,1,a] (1) base 1 = ([a], 1, [base,base]) := by
    simp [consume, na, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c2 (base : Nat) : consume [a] (a) base 1 = ([], a, [base]) := by
    simp [consume, q0, na, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  simp only [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at c0 c1 c2
  simp only [inferViewStrides, List.prod_cons, List.prod_nil, Nat.mul_one, Nat.one_mul,
    List.cons_ne_nil, hp, ↓reduceIte, List.zip_cons_cons, List.zip_nil_left,
    List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append,
    List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some] at view
  by_cases has : sa = sd
  all_goals simp [inferChunks, c0, c1, c2, na, has,
    Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at view
  all_goals simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]
  all_goals rcases view with ⟨v1,v2,v3⟩ <;> simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

private theorem coefficients_L11L (a d sa sb sh sd st oh od : Nat)
    (ha : 1 < a)
    (hd : 1 < d)
    (view : inferViewStrides [a,1,1,d] [sa,sb,sh,sd] [a,1,d] = some [st,oh,od]) :
    (a = 1 ∨ sa = 1*st) ∧ (1 = 1 ∨ sb = st) ∧
    (1 = 1 ∨ sh = oh) ∧ (d = 1 ∨ sd = od) := by
  have hp : a * (d) ≠ 0 := by positivity
  have na : a ≠ 1 := by omega
  have nd : d ≠ 1 := by omega
  have q0 : 1 < a * (d) := by exact (one_lt_mul_of_le_of_lt (le_of_lt ha) hd)
  have q1 : d < a * (d) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right ha (show 0 < d by positivity)
  have q2 : 1 < d := by exact hd
  have q3 : 1 < a := by exact ha
  have c0 (base : Nat) : consume [d,1,a] (a * (d)) base 1 = ([], a * (d), [base,d * (base),d * (base)]) := by
    simp [consume, q0, q1, na, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c1 (base : Nat) : consume [d,1,a] (d) base 1 = ([a], d, [base,d * (base)]) := by
    simp [consume, q2, na, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c2 (base : Nat) : consume [a] (a) base 1 = ([], a, [base]) := by
    simp [consume, q3, na, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  simp only [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at c0 c1 c2
  simp only [inferViewStrides, List.prod_cons, List.prod_nil, Nat.mul_one, Nat.one_mul,
    List.cons_ne_nil, hp, ↓reduceIte, List.zip_cons_cons, List.zip_nil_left,
    List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append,
    List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some] at view
  by_cases has : sa = d * (sd)
  all_goals simp [inferChunks, c0, c1, c2, na, nd, has,
    Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at view
  all_goals simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]
  all_goals rcases view with ⟨v1,v2,v3⟩ <;> simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

private theorem coefficients_L1L1 (a h sa sb sh sd st oh od : Nat)
    (ha : 1 < a)
    (hh : 1 < h)
    (view : inferViewStrides [a,1,h,1] [sa,sb,sh,sd] [a,h,1] = some [st,oh,od]) :
    (a = 1 ∨ sa = 1*st) ∧ (1 = 1 ∨ sb = st) ∧
    (h = 1 ∨ sh = oh) ∧ (1 = 1 ∨ sd = od) := by
  have hp : a * (h) ≠ 0 := by positivity
  have na : a ≠ 1 := by omega
  have nh : h ≠ 1 := by omega
  have q0 : 1 < a * (h) := by exact (one_lt_mul_of_le_of_lt (le_of_lt ha) hh)
  have q1 : h < a * (h) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right ha (show 0 < h by positivity)
  have q2 : 1 < h := by exact hh
  have q3 : 1 < a := by exact ha
  have c0 (base : Nat) : consume [1,h,a] (a * (h)) base 1 = ([], a * (h), [base,base,h * (base)]) := by
    simp [consume, q0, q1, na, nh, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c1 (base : Nat) : consume [1,h,a] (h) base 1 = ([a], h, [base,base]) := by
    simp [consume, q2, na, nh, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c2 (base : Nat) : consume [a] (a) base 1 = ([], a, [base]) := by
    simp [consume, q3, na, nh, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c3 (base : Nat) : consume [1,h,a] (1) base 1 = ([h,a], 1, [base]) := by
    simp [consume, na, nh, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c4 (base : Nat) : consume [h,a] (a * (h)) base 1 = ([], a * (h), [base,h * (base)]) := by
    simp [consume, q0, q1, na, nh, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c5 (base : Nat) : consume [h,a] (h) base 1 = ([a], h, [base]) := by
    simp [consume, q2, na, nh, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  simp only [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at c0 c1 c2 c3 c4 c5
  simp only [inferViewStrides, List.prod_cons, List.prod_nil, Nat.mul_one, Nat.one_mul,
    List.cons_ne_nil, hp, ↓reduceIte, List.zip_cons_cons, List.zip_nil_left,
    List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append,
    List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some] at view
  by_cases hs : sh = sd <;> by_cases has : sa = h * (sh)
  all_goals simp [inferChunks, c0, c1, c2, c3, c4, c5, na, nh, hs, has,
    Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at view
  all_goals simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]
  all_goals rcases view with ⟨v1,v2,v3⟩ <;> simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

private theorem coefficients_L1LL (a h d sa sb sh sd st oh od : Nat)
    (ha : 1 < a)
    (hh : 1 < h)
    (hd : 1 < d)
    (view : inferViewStrides [a,1,h,d] [sa,sb,sh,sd] [a,h,d] = some [st,oh,od]) :
    (a = 1 ∨ sa = 1*st) ∧ (1 = 1 ∨ sb = st) ∧
    (h = 1 ∨ sh = oh) ∧ (d = 1 ∨ sd = od) := by
  have hp : a * (h * (d)) ≠ 0 := by positivity
  have na : a ≠ 1 := by omega
  have nh : h ≠ 1 := by omega
  have nd : d ≠ 1 := by omega
  have q0 : 1 < a * (h * (d)) := by exact (one_lt_mul_of_le_of_lt (le_of_lt ha) (one_lt_mul_of_le_of_lt (le_of_lt hh) hd))
  have q1 : d < a * (h * (d)) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right (one_lt_mul_of_le_of_lt (le_of_lt ha) hh) (show 0 < d by positivity)
  have q2 : h * (d) < a * (h * (d)) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right ha (show 0 < h * (d) by positivity)
  have q3 : 1 < h * (d) := by exact (one_lt_mul_of_le_of_lt (le_of_lt hh) hd)
  have q4 : d < h * (d) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right hh (show 0 < d by positivity)
  have q5 : 1 < a := by exact ha
  have q6 : 1 < d := by exact hd
  have q7 : 1 < a * (h) := by exact (one_lt_mul_of_le_of_lt (le_of_lt ha) hh)
  have q8 : h < a * (h) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right ha (show 0 < h by positivity)
  have q9 : 1 < h := by exact hh
  have c0 (base : Nat) : consume [d,h,a] (a * (h * (d))) base 1 = ([], a * (h * (d)), [base,d * (base),h * (d * (base))]) := by
    simp [consume, q0, q1, q2, na, nh, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c1 (base : Nat) : consume [d,h,a] (h * (d)) base 1 = ([a], h * (d), [base,d * (base)]) := by
    simp [consume, q3, q4, na, nh, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c2 (base : Nat) : consume [a] (a) base 1 = ([], a, [base]) := by
    simp [consume, q5, na, nh, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c3 (base : Nat) : consume [d,h,a] (d) base 1 = ([h,a], d, [base]) := by
    simp [consume, q6, na, nh, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c4 (base : Nat) : consume [h,a] (a * (h)) base 1 = ([], a * (h), [base,h * (base)]) := by
    simp [consume, q7, q8, na, nh, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c5 (base : Nat) : consume [h,a] (h) base 1 = ([a], h, [base]) := by
    simp [consume, q9, na, nh, nd, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  simp only [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at c0 c1 c2 c3 c4 c5
  simp only [inferViewStrides, List.prod_cons, List.prod_nil, Nat.mul_one, Nat.one_mul,
    List.cons_ne_nil, hp, ↓reduceIte, List.zip_cons_cons, List.zip_nil_left,
    List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append,
    List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some] at view
  by_cases hs : sh = d * (sd) <;> by_cases has : sa = h * (sh)
  all_goals simp [inferChunks, c0, c1, c2, c3, c4, c5, na, nh, nd, hs, has,
    Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at view
  all_goals simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]
  all_goals rcases view with ⟨v1,v2,v3⟩ <;> simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

private theorem coefficients_LL11 (a b sa sb sh sd st oh od : Nat)
    (ha : 1 < a)
    (hb : 1 < b)
    (view : inferViewStrides [a,b,1,1] [sa,sb,sh,sd] [a * (b),1,1] = some [st,oh,od]) :
    (a = 1 ∨ sa = b*st) ∧ (b = 1 ∨ sb = st) ∧
    (1 = 1 ∨ sh = oh) ∧ (1 = 1 ∨ sd = od) := by
  have hp : a * (b) ≠ 0 := by positivity
  have na : a ≠ 1 := by omega
  have nb : b ≠ 1 := by omega
  have nab : a * (b) ≠ 1 := ne_of_gt (one_lt_mul_of_le_of_lt (le_of_lt ha) hb)
  have q0 : 1 < a * (b) := by exact (one_lt_mul_of_le_of_lt (le_of_lt ha) hb)
  have q1 : 1 < b := by exact hb
  have q2 : b < a * (b) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right ha (show 0 < b by positivity)
  have c0 (base : Nat) : consume [1,1,a * (b)] (a * (b)) base 1 = ([], a * (b), [base,base,base]) := by
    simp [consume, q0, na, nb, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c1 (base : Nat) : consume [1,1,a * (b)] (b) base 1 = ([], a * (b), [base,base,base]) := by
    simp [consume, q1, na, nb, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c2 (base : Nat) : consume [1,1,a * (b)] (1) base 1 = ([a * (b)], 1, [base,base]) := by
    simp [consume, na, nb, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c3 (base : Nat) : consume [a * (b)] (a * (b)) base 1 = ([], a * (b), [base]) := by
    simp [consume, q0, na, nb, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c4 (base : Nat) : consume [a * (b)] (b) base 1 = ([], a * (b), [base]) := by
    simp [consume, q1, na, nb, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have e0 : a * (b) ≠ b := ne_of_gt q2
  simp only [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at c0 c1 c2 c3 c4 e0
  simp only [inferViewStrides, List.prod_cons, List.prod_nil, Nat.mul_one, Nat.one_mul,
    List.cons_ne_nil, hp, ↓reduceIte, List.zip_cons_cons, List.zip_nil_left,
    List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append,
    List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some] at view
  by_cases hbs : sb = sd <;> by_cases has : sa = b * (sb)
  all_goals simp [inferChunks, c0, c1, c2, c3, c4, e0, na, nb, hbs, has,
    Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at view
  all_goals simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]
  all_goals rcases view with ⟨v1,v2,v3⟩ <;> simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

private theorem coefficients_LL1L (a b d sa sb sh sd st oh od : Nat)
    (ha : 1 < a)
    (hb : 1 < b)
    (hd : 1 < d)
    (view : inferViewStrides [a,b,1,d] [sa,sb,sh,sd] [a * (b),1,d] = some [st,oh,od]) :
    (a = 1 ∨ sa = b*st) ∧ (b = 1 ∨ sb = st) ∧
    (1 = 1 ∨ sh = oh) ∧ (d = 1 ∨ sd = od) := by
  have hp : a * (b * (d)) ≠ 0 := by positivity
  have na : a ≠ 1 := by omega
  have nb : b ≠ 1 := by omega
  have nd : d ≠ 1 := by omega
  have nab : a * (b) ≠ 1 := ne_of_gt (one_lt_mul_of_le_of_lt (le_of_lt ha) hb)
  have q0 : 1 < a * (b * (d)) := by exact (one_lt_mul_of_le_of_lt (le_of_lt ha) (one_lt_mul_of_le_of_lt (le_of_lt hb) hd))
  have q1 : d < a * (b * (d)) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right (one_lt_mul_of_le_of_lt (le_of_lt ha) hb) (show 0 < d by positivity)
  have q2 : 1 < b * (d) := by exact (one_lt_mul_of_le_of_lt (le_of_lt hb) hd)
  have q3 : d < b * (d) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right hb (show 0 < d by positivity)
  have q4 : 1 < d := by exact hd
  have q5 : 1 < a * (b) := by exact (one_lt_mul_of_le_of_lt (le_of_lt ha) hb)
  have q6 : 1 < b := by exact hb
  have q7 : b < a * (b) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right ha (show 0 < b by positivity)
  have q8 : b * (d) < a * (b * (d)) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right ha (show 0 < b * (d) by positivity)
  have c0 (base : Nat) : consume [d,1,a * (b)] (a * (b * (d))) base 1 = ([], a * (b * (d)), [base,d * (base),d * (base)]) := by
    simp [consume, q0, q1, na, nb, nd, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c1 (base : Nat) : consume [d,1,a * (b)] (b * (d)) base 1 = ([], a * (b * (d)), [base,d * (base),d * (base)]) := by
    simp [consume, q2, q3, na, nb, nd, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c2 (base : Nat) : consume [d,1,a * (b)] (d) base 1 = ([a * (b)], d, [base,d * (base)]) := by
    simp [consume, q4, na, nb, nd, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c3 (base : Nat) : consume [a * (b)] (a * (b)) base 1 = ([], a * (b), [base]) := by
    simp [consume, q5, na, nb, nd, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c4 (base : Nat) : consume [a * (b)] (b) base 1 = ([], a * (b), [base]) := by
    simp [consume, q6, na, nb, nd, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have e0 : a * (b) ≠ b := ne_of_gt q7
  have e1 : a * (b * (d)) ≠ b * (d) := ne_of_gt q8
  simp only [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at c0 c1 c2 c3 c4 e0 e1
  simp only [inferViewStrides, List.prod_cons, List.prod_nil, Nat.mul_one, Nat.one_mul,
    List.cons_ne_nil, hp, ↓reduceIte, List.zip_cons_cons, List.zip_nil_left,
    List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append,
    List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some] at view
  by_cases hbs : sb = d * (sd) <;> by_cases has : sa = b * (sb)
  all_goals simp [inferChunks, c0, c1, c2, c3, c4, e0, e1, na, nb, nd, hbs, has,
    Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at view
  all_goals simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]
  all_goals rcases view with ⟨v1,v2,v3⟩ <;> simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

private theorem coefficients_LLL1 (a b h sa sb sh sd st oh od : Nat)
    (ha : 1 < a)
    (hb : 1 < b)
    (hh : 1 < h)
    (view : inferViewStrides [a,b,h,1] [sa,sb,sh,sd] [a * (b),h,1] = some [st,oh,od]) :
    (a = 1 ∨ sa = b*st) ∧ (b = 1 ∨ sb = st) ∧
    (h = 1 ∨ sh = oh) ∧ (1 = 1 ∨ sd = od) := by
  have hp : a * (b * (h)) ≠ 0 := by positivity
  have na : a ≠ 1 := by omega
  have nb : b ≠ 1 := by omega
  have nh : h ≠ 1 := by omega
  have nab : a * (b) ≠ 1 := ne_of_gt (one_lt_mul_of_le_of_lt (le_of_lt ha) hb)
  have q0 : 1 < a * (b * (h)) := by exact (one_lt_mul_of_le_of_lt (le_of_lt ha) (one_lt_mul_of_le_of_lt (le_of_lt hb) hh))
  have q1 : h < a * (b * (h)) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right (one_lt_mul_of_le_of_lt (le_of_lt ha) hb) (show 0 < h by positivity)
  have q2 : 1 < b * (h) := by exact (one_lt_mul_of_le_of_lt (le_of_lt hb) hh)
  have q3 : h < b * (h) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right hb (show 0 < h by positivity)
  have q4 : 1 < h := by exact hh
  have q5 : 1 < a * (b) := by exact (one_lt_mul_of_le_of_lt (le_of_lt ha) hb)
  have q6 : 1 < b := by exact hb
  have q7 : b < a * (b) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right ha (show 0 < b by positivity)
  have q8 : b * (h) < a * (b * (h)) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right ha (show 0 < b * (h) by positivity)
  have c0 (base : Nat) : consume [1,h,a * (b)] (a * (b * (h))) base 1 = ([], a * (b * (h)), [base,base,h * (base)]) := by
    simp [consume, q0, q1, na, nb, nh, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c1 (base : Nat) : consume [1,h,a * (b)] (b * (h)) base 1 = ([], a * (b * (h)), [base,base,h * (base)]) := by
    simp [consume, q2, q3, na, nb, nh, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c2 (base : Nat) : consume [1,h,a * (b)] (h) base 1 = ([a * (b)], h, [base,base]) := by
    simp [consume, q4, na, nb, nh, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c3 (base : Nat) : consume [a * (b)] (a * (b)) base 1 = ([], a * (b), [base]) := by
    simp [consume, q5, na, nb, nh, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c4 (base : Nat) : consume [a * (b)] (b) base 1 = ([], a * (b), [base]) := by
    simp [consume, q6, na, nb, nh, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c5 (base : Nat) : consume [1,h,a * (b)] (1) base 1 = ([h,a * (b)], 1, [base]) := by
    simp [consume, na, nb, nh, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c6 (base : Nat) : consume [h,a * (b)] (a * (b * (h))) base 1 = ([], a * (b * (h)), [base,h * (base)]) := by
    simp [consume, q0, q1, na, nb, nh, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c7 (base : Nat) : consume [h,a * (b)] (b * (h)) base 1 = ([], a * (b * (h)), [base,h * (base)]) := by
    simp [consume, q2, q3, na, nb, nh, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c8 (base : Nat) : consume [h,a * (b)] (h) base 1 = ([a * (b)], h, [base]) := by
    simp [consume, q4, na, nb, nh, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have e0 : a * (b) ≠ b := ne_of_gt q7
  have e1 : a * (b * (h)) ≠ b * (h) := ne_of_gt q8
  simp only [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at c0 c1 c2 c3 c4 c5 c6 c7 c8 e0 e1
  simp only [inferViewStrides, List.prod_cons, List.prod_nil, Nat.mul_one, Nat.one_mul,
    List.cons_ne_nil, hp, ↓reduceIte, List.zip_cons_cons, List.zip_nil_left,
    List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append,
    List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some] at view
  by_cases hs : sh = sd <;> by_cases hbs : sb = h * (sh) <;> by_cases has : sa = b * (sb)
  all_goals simp [inferChunks, c0, c1, c2, c3, c4, c5, c6, c7, c8, e0, e1, na, nb, nh, hs, hbs, has,
    Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at view
  all_goals simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]
  all_goals rcases view with ⟨v1,v2,v3⟩ <;> simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

private theorem coefficients_LLLL (a b h d sa sb sh sd st oh od : Nat)
    (ha : 1 < a)
    (hb : 1 < b)
    (hh : 1 < h)
    (hd : 1 < d)
    (view : inferViewStrides [a,b,h,d] [sa,sb,sh,sd] [a * (b),h,d] = some [st,oh,od]) :
    (a = 1 ∨ sa = b*st) ∧ (b = 1 ∨ sb = st) ∧
    (h = 1 ∨ sh = oh) ∧ (d = 1 ∨ sd = od) := by
  have hp : a * (b * (h * (d))) ≠ 0 := by positivity
  have na : a ≠ 1 := by omega
  have nb : b ≠ 1 := by omega
  have nh : h ≠ 1 := by omega
  have nd : d ≠ 1 := by omega
  have nab : a * (b) ≠ 1 := ne_of_gt (one_lt_mul_of_le_of_lt (le_of_lt ha) hb)
  have q0 : 1 < a * (b * (h * (d))) := by exact (one_lt_mul_of_le_of_lt (le_of_lt ha) (one_lt_mul_of_le_of_lt (le_of_lt hb) (one_lt_mul_of_le_of_lt (le_of_lt hh) hd)))
  have q1 : d < a * (b * (h * (d))) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right (one_lt_mul_of_le_of_lt (le_of_lt ha) (one_lt_mul_of_le_of_lt (le_of_lt hb) hh)) (show 0 < d by positivity)
  have q2 : h * (d) < a * (b * (h * (d))) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right (one_lt_mul_of_le_of_lt (le_of_lt ha) hb) (show 0 < h * (d) by positivity)
  have q3 : 1 < b * (h * (d)) := by exact (one_lt_mul_of_le_of_lt (le_of_lt hb) (one_lt_mul_of_le_of_lt (le_of_lt hh) hd))
  have q4 : d < b * (h * (d)) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right (one_lt_mul_of_le_of_lt (le_of_lt hb) hh) (show 0 < d by positivity)
  have q5 : h * (d) < b * (h * (d)) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right hb (show 0 < h * (d) by positivity)
  have q6 : 1 < h * (d) := by exact (one_lt_mul_of_le_of_lt (le_of_lt hh) hd)
  have q7 : d < h * (d) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right hh (show 0 < d by positivity)
  have q8 : 1 < a * (b) := by exact (one_lt_mul_of_le_of_lt (le_of_lt ha) hb)
  have q9 : 1 < b := by exact hb
  have q10 : 1 < d := by exact hd
  have q11 : 1 < a * (b * (h)) := by exact (one_lt_mul_of_le_of_lt (le_of_lt ha) (one_lt_mul_of_le_of_lt (le_of_lt hb) hh))
  have q12 : h < a * (b * (h)) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right (one_lt_mul_of_le_of_lt (le_of_lt ha) hb) (show 0 < h by positivity)
  have q13 : 1 < b * (h) := by exact (one_lt_mul_of_le_of_lt (le_of_lt hb) hh)
  have q14 : h < b * (h) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right hb (show 0 < h by positivity)
  have q15 : 1 < h := by exact hh
  have q16 : b < a * (b) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right ha (show 0 < b by positivity)
  have q17 : b * (h) < a * (b * (h)) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right ha (show 0 < b * (h) by positivity)
  have q18 : b * (h * (d)) < a * (b * (h * (d))) := by simpa only [Nat.one_mul,Nat.mul_one,Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] using Nat.mul_lt_mul_of_pos_right ha (show 0 < b * (h * (d)) by positivity)
  have c0 (base : Nat) : consume [d,h,a * (b)] (a * (b * (h * (d)))) base 1 = ([], a * (b * (h * (d))), [base,d * (base),h * (d * (base))]) := by
    simp [consume, q0, q1, q2, na, nb, nh, nd, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c1 (base : Nat) : consume [d,h,a * (b)] (b * (h * (d))) base 1 = ([], a * (b * (h * (d))), [base,d * (base),h * (d * (base))]) := by
    simp [consume, q3, q4, q5, na, nb, nh, nd, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c2 (base : Nat) : consume [d,h,a * (b)] (h * (d)) base 1 = ([a * (b)], h * (d), [base,d * (base)]) := by
    simp [consume, q6, q7, na, nb, nh, nd, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c3 (base : Nat) : consume [a * (b)] (a * (b)) base 1 = ([], a * (b), [base]) := by
    simp [consume, q8, na, nb, nh, nd, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c4 (base : Nat) : consume [a * (b)] (b) base 1 = ([], a * (b), [base]) := by
    simp [consume, q9, na, nb, nh, nd, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c5 (base : Nat) : consume [d,h,a * (b)] (d) base 1 = ([h,a * (b)], d, [base]) := by
    simp [consume, q10, na, nb, nh, nd, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c6 (base : Nat) : consume [h,a * (b)] (a * (b * (h))) base 1 = ([], a * (b * (h)), [base,h * (base)]) := by
    simp [consume, q11, q12, na, nb, nh, nd, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c7 (base : Nat) : consume [h,a * (b)] (b * (h)) base 1 = ([], a * (b * (h)), [base,h * (base)]) := by
    simp [consume, q13, q14, na, nb, nh, nd, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have c8 (base : Nat) : consume [h,a * (b)] (h) base 1 = ([a * (b)], h, [base]) := by
    simp [consume, q15, na, nb, nh, nd, nab, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
  have e0 : a * (b) ≠ b := ne_of_gt q16
  have e1 : a * (b * (h)) ≠ b * (h) := ne_of_gt q17
  have e2 : a * (b * (h * (d))) ≠ b * (h * (d)) := ne_of_gt q18
  simp only [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at c0 c1 c2 c3 c4 c5 c6 c7 c8 e0 e1 e2
  simp only [inferViewStrides, List.prod_cons, List.prod_nil, Nat.mul_one, Nat.one_mul,
    List.cons_ne_nil, hp, ↓reduceIte, List.zip_cons_cons, List.zip_nil_left,
    List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append,
    List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some] at view
  by_cases hs : sh = d * (sd) <;> by_cases hbs : sb = h * (sh) <;> by_cases has : sa = b * (sb)
  all_goals simp [inferChunks, c0, c1, c2, c3, c4, c5, c6, c7, c8, e0, e1, e2, na, nb, nh, nd, hs, hbs, has,
    Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm] at view
  all_goals simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]
  all_goals rcases view with ⟨v1,v2,v3⟩ <;> simp_all [Nat.mul_assoc,Nat.mul_left_comm,Nat.mul_comm]

/-- Every positive rank4→rank3 view preserves the coefficient of each
non-singleton coordinate. The cases above cover the complete singleton/control
flow partition; all non-singleton sizes and all strides remain arbitrary. -/
theorem coefficients_positive (a b h d sa sb sh sd st oh od : Nat)
    (ha : 0 < a) (hb : 0 < b) (hh : 0 < h) (hd : 0 < d)
    (view : inferViewStrides [a,b,h,d] [sa,sb,sh,sd] [a*b,h,d] = some [st,oh,od]) :
    (a = 1 ∨ sa = b*st) ∧ (b = 1 ∨ sb = st) ∧
    (h = 1 ∨ sh = oh) ∧ (d = 1 ∨ sd = od) := by
  rcases (show a = 1 ∨ 1 < a by omega) with rfl | ha <;>
    rcases (show b = 1 ∨ 1 < b by omega) with rfl | hb <;>
    rcases (show h = 1 ∨ 1 < h by omega) with rfl | hh <;>
    rcases (show d = 1 ∨ 1 < d by omega) with rfl | hd
  · simpa only [Nat.one_mul,Nat.mul_one] using coefficients_1111 sa sb sh sd st oh od
      (by simpa only [Nat.one_mul,Nat.mul_one] using view)
  · simpa only [Nat.one_mul,Nat.mul_one] using coefficients_111L d sa sb sh sd st oh od hd
      (by simpa only [Nat.one_mul,Nat.mul_one] using view)
  · simpa only [Nat.one_mul,Nat.mul_one] using coefficients_11L1 h sa sb sh sd st oh od hh
      (by simpa only [Nat.one_mul,Nat.mul_one] using view)
  · simpa only [Nat.one_mul,Nat.mul_one] using coefficients_11LL h d sa sb sh sd st oh od hh hd
      (by simpa only [Nat.one_mul,Nat.mul_one] using view)
  · simpa only [Nat.one_mul,Nat.mul_one] using coefficients_1L11 b sa sb sh sd st oh od hb
      (by simpa only [Nat.one_mul,Nat.mul_one] using view)
  · simpa only [Nat.one_mul,Nat.mul_one] using coefficients_1L1L b d sa sb sh sd st oh od hb hd
      (by simpa only [Nat.one_mul,Nat.mul_one] using view)
  · simpa only [Nat.one_mul,Nat.mul_one] using coefficients_1LL1 b h sa sb sh sd st oh od hb hh
      (by simpa only [Nat.one_mul,Nat.mul_one] using view)
  · simpa only [Nat.one_mul,Nat.mul_one] using coefficients_1LLL b h d sa sb sh sd st oh od hb hh hd
      (by simpa only [Nat.one_mul,Nat.mul_one] using view)
  · simpa only [Nat.one_mul,Nat.mul_one] using coefficients_L111 a sa sb sh sd st oh od ha
      (by simpa only [Nat.one_mul,Nat.mul_one] using view)
  · simpa only [Nat.one_mul,Nat.mul_one] using coefficients_L11L a d sa sb sh sd st oh od ha hd
      (by simpa only [Nat.one_mul,Nat.mul_one] using view)
  · simpa only [Nat.one_mul,Nat.mul_one] using coefficients_L1L1 a h sa sb sh sd st oh od ha hh
      (by simpa only [Nat.one_mul,Nat.mul_one] using view)
  · simpa only [Nat.one_mul,Nat.mul_one] using coefficients_L1LL a h d sa sb sh sd st oh od ha hh hd
      (by simpa only [Nat.one_mul,Nat.mul_one] using view)
  · simpa only [Nat.one_mul,Nat.mul_one] using coefficients_LL11 a b sa sb sh sd st oh od ha hb
      (by simpa only [Nat.one_mul,Nat.mul_one] using view)
  · simpa only [Nat.one_mul,Nat.mul_one] using coefficients_LL1L a b d sa sb sh sd st oh od ha hb hd
      (by simpa only [Nat.one_mul,Nat.mul_one] using view)
  · simpa only [Nat.one_mul,Nat.mul_one] using coefficients_LLL1 a b h sa sb sh sd st oh od ha hb hh
      (by simpa only [Nat.one_mul,Nat.mul_one] using view)
  · simpa only [Nat.one_mul,Nat.mul_one] using coefficients_LLLL a b h d sa sb sh sd st oh od ha hb hh hd
      (by simpa only [Nat.one_mul,Nat.mul_one] using view)

#axiomsClean coefficients_positive
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.Host.View
