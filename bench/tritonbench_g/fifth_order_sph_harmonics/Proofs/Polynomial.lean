import VeriTile.Triton
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Polynomial
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 48000

/-- Independent eleven degree-five decimal-coefficient polynomials. These are
not asserted to be the ideal square-root-normalized harmonic polynomials.
The definition is a monomial coefficient table and never mentions execution,
FP decoding, kernels, successful host calls or final memory. -/
def value (channel : Fin 11) (x y z : ℝ) : ℝ :=
  match channel.val with
  | 0 => 0 + (232681380862329 / 100000000000000 : ℝ) * x^5 * y^0 * z^0 + (-232681380862329 / 10000000000000 : ℝ) * x^3 * y^0 * z^2 + (29085172607791 / 2500000000000 : ℝ) * x^1 * y^0 * z^4
  | 1 => 0 + (-294321253055229 / 10000000000000 : ℝ) * x^3 * y^1 * z^1 + (294321253055229 / 10000000000000 : ℝ) * x^1 * y^1 * z^3
  | 2 => 0 + (34686092313779 / 20000000000000 : ℝ) * x^5 * y^0 * z^0 + (-34686092313779 / 2500000000000 : ℝ) * x^3 * y^2 * z^0 + (-34686092313779 / 10000000000000 : ℝ) * x^3 * y^0 * z^2 + (104058276941337 / 2500000000000 : ℝ) * x^1 * y^2 * z^2 + (-104058276941337 / 20000000000000 : ℝ) * x^1 * y^0 * z^4
  | 3 => 0 + (-10620403417479 / 625000000000 : ℝ) * x^3 * y^1 * z^1 + (339852909359329 / 10000000000000 : ℝ) * x^1 * y^3 * z^1 + (-10620403417479 / 625000000000 : ℝ) * x^1 * y^1 * z^3
  | 4 => 0 + (80282703616657 / 50000000000000 : ℝ) * x^5 * y^0 * z^0 + (-192678488679977 / 10000000000000 : ℝ) * x^3 * y^2 * z^0 + (80282703616657 / 25000000000000 : ℝ) * x^3 * y^0 * z^2 + (128452325786651 / 10000000000000 : ℝ) * x^1 * y^4 * z^0 + (-192678488679977 / 10000000000000 : ℝ) * x^1 * y^2 * z^2 + (80282703616657 / 50000000000000 : ℝ) * x^1 * y^0 * z^4
  | 5 => 0 + (621867148191637 / 100000000000000 : ℝ) * x^4 * y^1 * z^0 + (-16583123951777 / 1000000000000 : ℝ) * x^2 * y^3 * z^0 + (124373429638327 / 10000000000000 : ℝ) * x^2 * y^1 * z^2 + (16583123951777 / 5000000000000 : ℝ) * x^0 * y^5 * z^0 + (-16583123951777 / 1000000000000 : ℝ) * x^0 * y^3 * z^2 + (621867148191637 / 100000000000000 : ℝ) * x^0 * y^1 * z^4
  | 6 => 0 + (80282703616657 / 50000000000000 : ℝ) * x^4 * y^0 * z^1 + (-192678488679977 / 10000000000000 : ℝ) * x^2 * y^2 * z^1 + (80282703616657 / 25000000000000 : ℝ) * x^2 * y^0 * z^3 + (128452325786651 / 10000000000000 : ℝ) * x^0 * y^4 * z^1 + (-192678488679977 / 10000000000000 : ℝ) * x^0 * y^2 * z^3 + (80282703616657 / 50000000000000 : ℝ) * x^0 * y^0 * z^5
  | 7 => 0 + (849632273398321 / 100000000000000 : ℝ) * x^4 * y^1 * z^0 + (-10620403417479 / 625000000000 : ℝ) * x^2 * y^3 * z^0 + (10620403417479 / 625000000000 : ℝ) * x^0 * y^3 * z^2 + (-849632273398321 / 100000000000000 : ℝ) * x^0 * y^1 * z^4
  | 8 => 0 + (104058276941337 / 20000000000000 : ℝ) * x^4 * y^0 * z^1 + (-104058276941337 / 2500000000000 : ℝ) * x^2 * y^2 * z^1 + (34686092313779 / 10000000000000 : ℝ) * x^2 * y^0 * z^3 + (34686092313779 / 2500000000000 : ℝ) * x^0 * y^2 * z^3 + (-34686092313779 / 20000000000000 : ℝ) * x^0 * y^0 * z^5
  | 9 => 0 + (91975391579759 / 12500000000000 : ℝ) * x^4 * y^1 * z^0 + (-441481879582843 / 10000000000000 : ℝ) * x^2 * y^1 * z^2 + (91975391579759 / 12500000000000 : ℝ) * x^0 * y^1 * z^4
  | _ => 0 + (29085172607791 / 2500000000000 : ℝ) * x^4 * y^0 * z^1 + (-232681380862329 / 10000000000000 : ℝ) * x^2 * y^0 * z^3 + (232681380862329 / 100000000000000 : ℝ) * x^0 * y^0 * z^5

/-- Exact partial derivatives of the same polynomial coefficient table. -/
def jacobian (channel : Fin 11) (component : Fin 3) (x y z : ℝ) : ℝ :=
  match channel.val, component.val with
  | 0, 0 => 0 + (232681380862329 / 20000000000000 : ℝ) * x^4 * y^0 * z^0 + (-698044142586987 / 10000000000000 : ℝ) * x^2 * y^0 * z^2 + (29085172607791 / 2500000000000 : ℝ) * x^0 * y^0 * z^4
  | 0, 1 => 0 + (0 / 1 : ℝ) * x^0 * y^0 * z^0
  | 0, 2 => 0 + (-232681380862329 / 5000000000000 : ℝ) * x^3 * y^0 * z^1 + (29085172607791 / 625000000000 : ℝ) * x^1 * y^0 * z^3
  | 1, 0 => 0 + (-882963759165687 / 10000000000000 : ℝ) * x^2 * y^1 * z^1 + (294321253055229 / 10000000000000 : ℝ) * x^0 * y^1 * z^3
  | 1, 1 => 0 + (-294321253055229 / 10000000000000 : ℝ) * x^3 * y^0 * z^1 + (294321253055229 / 10000000000000 : ℝ) * x^1 * y^0 * z^3
  | 1, 2 => 0 + (-294321253055229 / 10000000000000 : ℝ) * x^3 * y^1 * z^0 + (882963759165687 / 10000000000000 : ℝ) * x^1 * y^1 * z^2
  | 2, 0 => 0 + (34686092313779 / 4000000000000 : ℝ) * x^4 * y^0 * z^0 + (-104058276941337 / 2500000000000 : ℝ) * x^2 * y^2 * z^0 + (-104058276941337 / 10000000000000 : ℝ) * x^2 * y^0 * z^2 + (104058276941337 / 2500000000000 : ℝ) * x^0 * y^2 * z^2 + (-104058276941337 / 20000000000000 : ℝ) * x^0 * y^0 * z^4
  | 2, 1 => 0 + (-34686092313779 / 1250000000000 : ℝ) * x^3 * y^1 * z^0 + (104058276941337 / 1250000000000 : ℝ) * x^1 * y^1 * z^2
  | 2, 2 => 0 + (-34686092313779 / 5000000000000 : ℝ) * x^3 * y^0 * z^1 + (104058276941337 / 1250000000000 : ℝ) * x^1 * y^2 * z^1 + (-104058276941337 / 5000000000000 : ℝ) * x^1 * y^0 * z^3
  | 3, 0 => 0 + (-31861210252437 / 625000000000 : ℝ) * x^2 * y^1 * z^1 + (339852909359329 / 10000000000000 : ℝ) * x^0 * y^3 * z^1 + (-10620403417479 / 625000000000 : ℝ) * x^0 * y^1 * z^3
  | 3, 1 => 0 + (-10620403417479 / 625000000000 : ℝ) * x^3 * y^0 * z^1 + (1019558728077987 / 10000000000000 : ℝ) * x^1 * y^2 * z^1 + (-10620403417479 / 625000000000 : ℝ) * x^1 * y^0 * z^3
  | 3, 2 => 0 + (-10620403417479 / 625000000000 : ℝ) * x^3 * y^1 * z^0 + (339852909359329 / 10000000000000 : ℝ) * x^1 * y^3 * z^0 + (-31861210252437 / 625000000000 : ℝ) * x^1 * y^1 * z^2
  | 4, 0 => 0 + (80282703616657 / 10000000000000 : ℝ) * x^4 * y^0 * z^0 + (-578035466039931 / 10000000000000 : ℝ) * x^2 * y^2 * z^0 + (240848110849971 / 25000000000000 : ℝ) * x^2 * y^0 * z^2 + (128452325786651 / 10000000000000 : ℝ) * x^0 * y^4 * z^0 + (-192678488679977 / 10000000000000 : ℝ) * x^0 * y^2 * z^2 + (80282703616657 / 50000000000000 : ℝ) * x^0 * y^0 * z^4
  | 4, 1 => 0 + (-192678488679977 / 5000000000000 : ℝ) * x^3 * y^1 * z^0 + (128452325786651 / 2500000000000 : ℝ) * x^1 * y^3 * z^0 + (-192678488679977 / 5000000000000 : ℝ) * x^1 * y^1 * z^2
  | 4, 2 => 0 + (80282703616657 / 12500000000000 : ℝ) * x^3 * y^0 * z^1 + (-192678488679977 / 5000000000000 : ℝ) * x^1 * y^2 * z^1 + (80282703616657 / 12500000000000 : ℝ) * x^1 * y^0 * z^3
  | 5, 0 => 0 + (621867148191637 / 25000000000000 : ℝ) * x^3 * y^1 * z^0 + (-16583123951777 / 500000000000 : ℝ) * x^1 * y^3 * z^0 + (124373429638327 / 5000000000000 : ℝ) * x^1 * y^1 * z^2
  | 5, 1 => 0 + (621867148191637 / 100000000000000 : ℝ) * x^4 * y^0 * z^0 + (-49749371855331 / 1000000000000 : ℝ) * x^2 * y^2 * z^0 + (124373429638327 / 10000000000000 : ℝ) * x^2 * y^0 * z^2 + (16583123951777 / 1000000000000 : ℝ) * x^0 * y^4 * z^0 + (-49749371855331 / 1000000000000 : ℝ) * x^0 * y^2 * z^2 + (621867148191637 / 100000000000000 : ℝ) * x^0 * y^0 * z^4
  | 5, 2 => 0 + (124373429638327 / 5000000000000 : ℝ) * x^2 * y^1 * z^1 + (-16583123951777 / 500000000000 : ℝ) * x^0 * y^3 * z^1 + (621867148191637 / 25000000000000 : ℝ) * x^0 * y^1 * z^3
  | 6, 0 => 0 + (80282703616657 / 12500000000000 : ℝ) * x^3 * y^0 * z^1 + (-192678488679977 / 5000000000000 : ℝ) * x^1 * y^2 * z^1 + (80282703616657 / 12500000000000 : ℝ) * x^1 * y^0 * z^3
  | 6, 1 => 0 + (-192678488679977 / 5000000000000 : ℝ) * x^2 * y^1 * z^1 + (128452325786651 / 2500000000000 : ℝ) * x^0 * y^3 * z^1 + (-192678488679977 / 5000000000000 : ℝ) * x^0 * y^1 * z^3
  | 6, 2 => 0 + (80282703616657 / 50000000000000 : ℝ) * x^4 * y^0 * z^0 + (-192678488679977 / 10000000000000 : ℝ) * x^2 * y^2 * z^0 + (240848110849971 / 25000000000000 : ℝ) * x^2 * y^0 * z^2 + (128452325786651 / 10000000000000 : ℝ) * x^0 * y^4 * z^0 + (-578035466039931 / 10000000000000 : ℝ) * x^0 * y^2 * z^2 + (80282703616657 / 10000000000000 : ℝ) * x^0 * y^0 * z^4
  | 7, 0 => 0 + (849632273398321 / 25000000000000 : ℝ) * x^3 * y^1 * z^0 + (-10620403417479 / 312500000000 : ℝ) * x^1 * y^3 * z^0
  | 7, 1 => 0 + (849632273398321 / 100000000000000 : ℝ) * x^4 * y^0 * z^0 + (-31861210252437 / 625000000000 : ℝ) * x^2 * y^2 * z^0 + (31861210252437 / 625000000000 : ℝ) * x^0 * y^2 * z^2 + (-849632273398321 / 100000000000000 : ℝ) * x^0 * y^0 * z^4
  | 7, 2 => 0 + (10620403417479 / 312500000000 : ℝ) * x^0 * y^3 * z^1 + (-849632273398321 / 25000000000000 : ℝ) * x^0 * y^1 * z^3
  | 8, 0 => 0 + (104058276941337 / 5000000000000 : ℝ) * x^3 * y^0 * z^1 + (-104058276941337 / 1250000000000 : ℝ) * x^1 * y^2 * z^1 + (34686092313779 / 5000000000000 : ℝ) * x^1 * y^0 * z^3
  | 8, 1 => 0 + (-104058276941337 / 1250000000000 : ℝ) * x^2 * y^1 * z^1 + (34686092313779 / 1250000000000 : ℝ) * x^0 * y^1 * z^3
  | 8, 2 => 0 + (104058276941337 / 20000000000000 : ℝ) * x^4 * y^0 * z^0 + (-104058276941337 / 2500000000000 : ℝ) * x^2 * y^2 * z^0 + (104058276941337 / 10000000000000 : ℝ) * x^2 * y^0 * z^2 + (104058276941337 / 2500000000000 : ℝ) * x^0 * y^2 * z^2 + (-34686092313779 / 4000000000000 : ℝ) * x^0 * y^0 * z^4
  | 9, 0 => 0 + (91975391579759 / 3125000000000 : ℝ) * x^3 * y^1 * z^0 + (-441481879582843 / 5000000000000 : ℝ) * x^1 * y^1 * z^2
  | 9, 1 => 0 + (91975391579759 / 12500000000000 : ℝ) * x^4 * y^0 * z^0 + (-441481879582843 / 10000000000000 : ℝ) * x^2 * y^0 * z^2 + (91975391579759 / 12500000000000 : ℝ) * x^0 * y^0 * z^4
  | 9, 2 => 0 + (-441481879582843 / 5000000000000 : ℝ) * x^2 * y^1 * z^1 + (91975391579759 / 3125000000000 : ℝ) * x^0 * y^1 * z^3
  | 10, 0 => 0 + (29085172607791 / 625000000000 : ℝ) * x^3 * y^0 * z^1 + (-232681380862329 / 5000000000000 : ℝ) * x^1 * y^0 * z^3
  | 10, 1 => 0 + (0 / 1 : ℝ) * x^0 * y^0 * z^0
  | 10, 2 => 0 + (29085172607791 / 2500000000000 : ℝ) * x^4 * y^0 * z^0 + (-698044142586987 / 10000000000000 : ℝ) * x^2 * y^0 * z^2 + (232681380862329 / 20000000000000 : ℝ) * x^0 * y^0 * z^4
  | _, _ => 0

theorem derivative_0_x (x y z : ℝ) :
    HasDerivAt (fun t=>value 0 t y z) (jacobian 0 0 x y z) x := by
  convert ((((hasDerivAt_const x (0 : ℝ)).add ((((hasDerivAt_const x (232681380862329 / 100000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 5)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^0)))).add ((((hasDerivAt_const x (-232681380862329 / 10000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 3)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^2)))).add ((((hasDerivAt_const x (29085172607791 / 2500000000000 : ℝ)).mul ((hasDerivAt_id x).pow 1)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^4)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_0_y (x y z : ℝ) :
    HasDerivAt (fun t=>value 0 x t z) (jacobian 0 1 x y z) y := by
  convert ((((hasDerivAt_const y (0 : ℝ)).add ((((hasDerivAt_const y (232681380862329 / 100000000000000 : ℝ)).mul (hasDerivAt_const y (x^5))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^0)))).add ((((hasDerivAt_const y (-232681380862329 / 10000000000000 : ℝ)).mul (hasDerivAt_const y (x^3))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^2)))).add ((((hasDerivAt_const y (29085172607791 / 2500000000000 : ℝ)).mul (hasDerivAt_const y (x^1))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^4)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_0_z (x y z : ℝ) :
    HasDerivAt (fun t=>value 0 x y t) (jacobian 0 2 x y z) z := by
  convert ((((hasDerivAt_const z (0 : ℝ)).add ((((hasDerivAt_const z (232681380862329 / 100000000000000 : ℝ)).mul (hasDerivAt_const z (x^5))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 0))).add ((((hasDerivAt_const z (-232681380862329 / 10000000000000 : ℝ)).mul (hasDerivAt_const z (x^3))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 2))).add ((((hasDerivAt_const z (29085172607791 / 2500000000000 : ℝ)).mul (hasDerivAt_const z (x^1))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 4))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_1_x (x y z : ℝ) :
    HasDerivAt (fun t=>value 1 t y z) (jacobian 1 0 x y z) x := by
  convert (((hasDerivAt_const x (0 : ℝ)).add ((((hasDerivAt_const x (-294321253055229 / 10000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 3)).mul (hasDerivAt_const x (y^1))).mul (hasDerivAt_const x (z^1)))).add ((((hasDerivAt_const x (294321253055229 / 10000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 1)).mul (hasDerivAt_const x (y^1))).mul (hasDerivAt_const x (z^3)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_1_y (x y z : ℝ) :
    HasDerivAt (fun t=>value 1 x t z) (jacobian 1 1 x y z) y := by
  convert (((hasDerivAt_const y (0 : ℝ)).add ((((hasDerivAt_const y (-294321253055229 / 10000000000000 : ℝ)).mul (hasDerivAt_const y (x^3))).mul ((hasDerivAt_id y).pow 1)).mul (hasDerivAt_const y (z^1)))).add ((((hasDerivAt_const y (294321253055229 / 10000000000000 : ℝ)).mul (hasDerivAt_const y (x^1))).mul ((hasDerivAt_id y).pow 1)).mul (hasDerivAt_const y (z^3)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_1_z (x y z : ℝ) :
    HasDerivAt (fun t=>value 1 x y t) (jacobian 1 2 x y z) z := by
  convert (((hasDerivAt_const z (0 : ℝ)).add ((((hasDerivAt_const z (-294321253055229 / 10000000000000 : ℝ)).mul (hasDerivAt_const z (x^3))).mul (hasDerivAt_const z (y^1))).mul ((hasDerivAt_id z).pow 1))).add ((((hasDerivAt_const z (294321253055229 / 10000000000000 : ℝ)).mul (hasDerivAt_const z (x^1))).mul (hasDerivAt_const z (y^1))).mul ((hasDerivAt_id z).pow 3))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_2_x (x y z : ℝ) :
    HasDerivAt (fun t=>value 2 t y z) (jacobian 2 0 x y z) x := by
  convert ((((((hasDerivAt_const x (0 : ℝ)).add ((((hasDerivAt_const x (34686092313779 / 20000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 5)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^0)))).add ((((hasDerivAt_const x (-34686092313779 / 2500000000000 : ℝ)).mul ((hasDerivAt_id x).pow 3)).mul (hasDerivAt_const x (y^2))).mul (hasDerivAt_const x (z^0)))).add ((((hasDerivAt_const x (-34686092313779 / 10000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 3)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^2)))).add ((((hasDerivAt_const x (104058276941337 / 2500000000000 : ℝ)).mul ((hasDerivAt_id x).pow 1)).mul (hasDerivAt_const x (y^2))).mul (hasDerivAt_const x (z^2)))).add ((((hasDerivAt_const x (-104058276941337 / 20000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 1)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^4)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_2_y (x y z : ℝ) :
    HasDerivAt (fun t=>value 2 x t z) (jacobian 2 1 x y z) y := by
  convert ((((((hasDerivAt_const y (0 : ℝ)).add ((((hasDerivAt_const y (34686092313779 / 20000000000000 : ℝ)).mul (hasDerivAt_const y (x^5))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^0)))).add ((((hasDerivAt_const y (-34686092313779 / 2500000000000 : ℝ)).mul (hasDerivAt_const y (x^3))).mul ((hasDerivAt_id y).pow 2)).mul (hasDerivAt_const y (z^0)))).add ((((hasDerivAt_const y (-34686092313779 / 10000000000000 : ℝ)).mul (hasDerivAt_const y (x^3))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^2)))).add ((((hasDerivAt_const y (104058276941337 / 2500000000000 : ℝ)).mul (hasDerivAt_const y (x^1))).mul ((hasDerivAt_id y).pow 2)).mul (hasDerivAt_const y (z^2)))).add ((((hasDerivAt_const y (-104058276941337 / 20000000000000 : ℝ)).mul (hasDerivAt_const y (x^1))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^4)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_2_z (x y z : ℝ) :
    HasDerivAt (fun t=>value 2 x y t) (jacobian 2 2 x y z) z := by
  convert ((((((hasDerivAt_const z (0 : ℝ)).add ((((hasDerivAt_const z (34686092313779 / 20000000000000 : ℝ)).mul (hasDerivAt_const z (x^5))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 0))).add ((((hasDerivAt_const z (-34686092313779 / 2500000000000 : ℝ)).mul (hasDerivAt_const z (x^3))).mul (hasDerivAt_const z (y^2))).mul ((hasDerivAt_id z).pow 0))).add ((((hasDerivAt_const z (-34686092313779 / 10000000000000 : ℝ)).mul (hasDerivAt_const z (x^3))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 2))).add ((((hasDerivAt_const z (104058276941337 / 2500000000000 : ℝ)).mul (hasDerivAt_const z (x^1))).mul (hasDerivAt_const z (y^2))).mul ((hasDerivAt_id z).pow 2))).add ((((hasDerivAt_const z (-104058276941337 / 20000000000000 : ℝ)).mul (hasDerivAt_const z (x^1))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 4))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_3_x (x y z : ℝ) :
    HasDerivAt (fun t=>value 3 t y z) (jacobian 3 0 x y z) x := by
  convert ((((hasDerivAt_const x (0 : ℝ)).add ((((hasDerivAt_const x (-10620403417479 / 625000000000 : ℝ)).mul ((hasDerivAt_id x).pow 3)).mul (hasDerivAt_const x (y^1))).mul (hasDerivAt_const x (z^1)))).add ((((hasDerivAt_const x (339852909359329 / 10000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 1)).mul (hasDerivAt_const x (y^3))).mul (hasDerivAt_const x (z^1)))).add ((((hasDerivAt_const x (-10620403417479 / 625000000000 : ℝ)).mul ((hasDerivAt_id x).pow 1)).mul (hasDerivAt_const x (y^1))).mul (hasDerivAt_const x (z^3)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_3_y (x y z : ℝ) :
    HasDerivAt (fun t=>value 3 x t z) (jacobian 3 1 x y z) y := by
  convert ((((hasDerivAt_const y (0 : ℝ)).add ((((hasDerivAt_const y (-10620403417479 / 625000000000 : ℝ)).mul (hasDerivAt_const y (x^3))).mul ((hasDerivAt_id y).pow 1)).mul (hasDerivAt_const y (z^1)))).add ((((hasDerivAt_const y (339852909359329 / 10000000000000 : ℝ)).mul (hasDerivAt_const y (x^1))).mul ((hasDerivAt_id y).pow 3)).mul (hasDerivAt_const y (z^1)))).add ((((hasDerivAt_const y (-10620403417479 / 625000000000 : ℝ)).mul (hasDerivAt_const y (x^1))).mul ((hasDerivAt_id y).pow 1)).mul (hasDerivAt_const y (z^3)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_3_z (x y z : ℝ) :
    HasDerivAt (fun t=>value 3 x y t) (jacobian 3 2 x y z) z := by
  convert ((((hasDerivAt_const z (0 : ℝ)).add ((((hasDerivAt_const z (-10620403417479 / 625000000000 : ℝ)).mul (hasDerivAt_const z (x^3))).mul (hasDerivAt_const z (y^1))).mul ((hasDerivAt_id z).pow 1))).add ((((hasDerivAt_const z (339852909359329 / 10000000000000 : ℝ)).mul (hasDerivAt_const z (x^1))).mul (hasDerivAt_const z (y^3))).mul ((hasDerivAt_id z).pow 1))).add ((((hasDerivAt_const z (-10620403417479 / 625000000000 : ℝ)).mul (hasDerivAt_const z (x^1))).mul (hasDerivAt_const z (y^1))).mul ((hasDerivAt_id z).pow 3))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_4_x (x y z : ℝ) :
    HasDerivAt (fun t=>value 4 t y z) (jacobian 4 0 x y z) x := by
  convert (((((((hasDerivAt_const x (0 : ℝ)).add ((((hasDerivAt_const x (80282703616657 / 50000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 5)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^0)))).add ((((hasDerivAt_const x (-192678488679977 / 10000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 3)).mul (hasDerivAt_const x (y^2))).mul (hasDerivAt_const x (z^0)))).add ((((hasDerivAt_const x (80282703616657 / 25000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 3)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^2)))).add ((((hasDerivAt_const x (128452325786651 / 10000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 1)).mul (hasDerivAt_const x (y^4))).mul (hasDerivAt_const x (z^0)))).add ((((hasDerivAt_const x (-192678488679977 / 10000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 1)).mul (hasDerivAt_const x (y^2))).mul (hasDerivAt_const x (z^2)))).add ((((hasDerivAt_const x (80282703616657 / 50000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 1)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^4)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_4_y (x y z : ℝ) :
    HasDerivAt (fun t=>value 4 x t z) (jacobian 4 1 x y z) y := by
  convert (((((((hasDerivAt_const y (0 : ℝ)).add ((((hasDerivAt_const y (80282703616657 / 50000000000000 : ℝ)).mul (hasDerivAt_const y (x^5))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^0)))).add ((((hasDerivAt_const y (-192678488679977 / 10000000000000 : ℝ)).mul (hasDerivAt_const y (x^3))).mul ((hasDerivAt_id y).pow 2)).mul (hasDerivAt_const y (z^0)))).add ((((hasDerivAt_const y (80282703616657 / 25000000000000 : ℝ)).mul (hasDerivAt_const y (x^3))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^2)))).add ((((hasDerivAt_const y (128452325786651 / 10000000000000 : ℝ)).mul (hasDerivAt_const y (x^1))).mul ((hasDerivAt_id y).pow 4)).mul (hasDerivAt_const y (z^0)))).add ((((hasDerivAt_const y (-192678488679977 / 10000000000000 : ℝ)).mul (hasDerivAt_const y (x^1))).mul ((hasDerivAt_id y).pow 2)).mul (hasDerivAt_const y (z^2)))).add ((((hasDerivAt_const y (80282703616657 / 50000000000000 : ℝ)).mul (hasDerivAt_const y (x^1))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^4)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_4_z (x y z : ℝ) :
    HasDerivAt (fun t=>value 4 x y t) (jacobian 4 2 x y z) z := by
  convert (((((((hasDerivAt_const z (0 : ℝ)).add ((((hasDerivAt_const z (80282703616657 / 50000000000000 : ℝ)).mul (hasDerivAt_const z (x^5))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 0))).add ((((hasDerivAt_const z (-192678488679977 / 10000000000000 : ℝ)).mul (hasDerivAt_const z (x^3))).mul (hasDerivAt_const z (y^2))).mul ((hasDerivAt_id z).pow 0))).add ((((hasDerivAt_const z (80282703616657 / 25000000000000 : ℝ)).mul (hasDerivAt_const z (x^3))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 2))).add ((((hasDerivAt_const z (128452325786651 / 10000000000000 : ℝ)).mul (hasDerivAt_const z (x^1))).mul (hasDerivAt_const z (y^4))).mul ((hasDerivAt_id z).pow 0))).add ((((hasDerivAt_const z (-192678488679977 / 10000000000000 : ℝ)).mul (hasDerivAt_const z (x^1))).mul (hasDerivAt_const z (y^2))).mul ((hasDerivAt_id z).pow 2))).add ((((hasDerivAt_const z (80282703616657 / 50000000000000 : ℝ)).mul (hasDerivAt_const z (x^1))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 4))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_5_x (x y z : ℝ) :
    HasDerivAt (fun t=>value 5 t y z) (jacobian 5 0 x y z) x := by
  convert (((((((hasDerivAt_const x (0 : ℝ)).add ((((hasDerivAt_const x (621867148191637 / 100000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 4)).mul (hasDerivAt_const x (y^1))).mul (hasDerivAt_const x (z^0)))).add ((((hasDerivAt_const x (-16583123951777 / 1000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 2)).mul (hasDerivAt_const x (y^3))).mul (hasDerivAt_const x (z^0)))).add ((((hasDerivAt_const x (124373429638327 / 10000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 2)).mul (hasDerivAt_const x (y^1))).mul (hasDerivAt_const x (z^2)))).add ((((hasDerivAt_const x (16583123951777 / 5000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 0)).mul (hasDerivAt_const x (y^5))).mul (hasDerivAt_const x (z^0)))).add ((((hasDerivAt_const x (-16583123951777 / 1000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 0)).mul (hasDerivAt_const x (y^3))).mul (hasDerivAt_const x (z^2)))).add ((((hasDerivAt_const x (621867148191637 / 100000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 0)).mul (hasDerivAt_const x (y^1))).mul (hasDerivAt_const x (z^4)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_5_y (x y z : ℝ) :
    HasDerivAt (fun t=>value 5 x t z) (jacobian 5 1 x y z) y := by
  convert (((((((hasDerivAt_const y (0 : ℝ)).add ((((hasDerivAt_const y (621867148191637 / 100000000000000 : ℝ)).mul (hasDerivAt_const y (x^4))).mul ((hasDerivAt_id y).pow 1)).mul (hasDerivAt_const y (z^0)))).add ((((hasDerivAt_const y (-16583123951777 / 1000000000000 : ℝ)).mul (hasDerivAt_const y (x^2))).mul ((hasDerivAt_id y).pow 3)).mul (hasDerivAt_const y (z^0)))).add ((((hasDerivAt_const y (124373429638327 / 10000000000000 : ℝ)).mul (hasDerivAt_const y (x^2))).mul ((hasDerivAt_id y).pow 1)).mul (hasDerivAt_const y (z^2)))).add ((((hasDerivAt_const y (16583123951777 / 5000000000000 : ℝ)).mul (hasDerivAt_const y (x^0))).mul ((hasDerivAt_id y).pow 5)).mul (hasDerivAt_const y (z^0)))).add ((((hasDerivAt_const y (-16583123951777 / 1000000000000 : ℝ)).mul (hasDerivAt_const y (x^0))).mul ((hasDerivAt_id y).pow 3)).mul (hasDerivAt_const y (z^2)))).add ((((hasDerivAt_const y (621867148191637 / 100000000000000 : ℝ)).mul (hasDerivAt_const y (x^0))).mul ((hasDerivAt_id y).pow 1)).mul (hasDerivAt_const y (z^4)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_5_z (x y z : ℝ) :
    HasDerivAt (fun t=>value 5 x y t) (jacobian 5 2 x y z) z := by
  convert (((((((hasDerivAt_const z (0 : ℝ)).add ((((hasDerivAt_const z (621867148191637 / 100000000000000 : ℝ)).mul (hasDerivAt_const z (x^4))).mul (hasDerivAt_const z (y^1))).mul ((hasDerivAt_id z).pow 0))).add ((((hasDerivAt_const z (-16583123951777 / 1000000000000 : ℝ)).mul (hasDerivAt_const z (x^2))).mul (hasDerivAt_const z (y^3))).mul ((hasDerivAt_id z).pow 0))).add ((((hasDerivAt_const z (124373429638327 / 10000000000000 : ℝ)).mul (hasDerivAt_const z (x^2))).mul (hasDerivAt_const z (y^1))).mul ((hasDerivAt_id z).pow 2))).add ((((hasDerivAt_const z (16583123951777 / 5000000000000 : ℝ)).mul (hasDerivAt_const z (x^0))).mul (hasDerivAt_const z (y^5))).mul ((hasDerivAt_id z).pow 0))).add ((((hasDerivAt_const z (-16583123951777 / 1000000000000 : ℝ)).mul (hasDerivAt_const z (x^0))).mul (hasDerivAt_const z (y^3))).mul ((hasDerivAt_id z).pow 2))).add ((((hasDerivAt_const z (621867148191637 / 100000000000000 : ℝ)).mul (hasDerivAt_const z (x^0))).mul (hasDerivAt_const z (y^1))).mul ((hasDerivAt_id z).pow 4))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_6_x (x y z : ℝ) :
    HasDerivAt (fun t=>value 6 t y z) (jacobian 6 0 x y z) x := by
  convert (((((((hasDerivAt_const x (0 : ℝ)).add ((((hasDerivAt_const x (80282703616657 / 50000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 4)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^1)))).add ((((hasDerivAt_const x (-192678488679977 / 10000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 2)).mul (hasDerivAt_const x (y^2))).mul (hasDerivAt_const x (z^1)))).add ((((hasDerivAt_const x (80282703616657 / 25000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 2)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^3)))).add ((((hasDerivAt_const x (128452325786651 / 10000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 0)).mul (hasDerivAt_const x (y^4))).mul (hasDerivAt_const x (z^1)))).add ((((hasDerivAt_const x (-192678488679977 / 10000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 0)).mul (hasDerivAt_const x (y^2))).mul (hasDerivAt_const x (z^3)))).add ((((hasDerivAt_const x (80282703616657 / 50000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 0)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^5)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_6_y (x y z : ℝ) :
    HasDerivAt (fun t=>value 6 x t z) (jacobian 6 1 x y z) y := by
  convert (((((((hasDerivAt_const y (0 : ℝ)).add ((((hasDerivAt_const y (80282703616657 / 50000000000000 : ℝ)).mul (hasDerivAt_const y (x^4))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^1)))).add ((((hasDerivAt_const y (-192678488679977 / 10000000000000 : ℝ)).mul (hasDerivAt_const y (x^2))).mul ((hasDerivAt_id y).pow 2)).mul (hasDerivAt_const y (z^1)))).add ((((hasDerivAt_const y (80282703616657 / 25000000000000 : ℝ)).mul (hasDerivAt_const y (x^2))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^3)))).add ((((hasDerivAt_const y (128452325786651 / 10000000000000 : ℝ)).mul (hasDerivAt_const y (x^0))).mul ((hasDerivAt_id y).pow 4)).mul (hasDerivAt_const y (z^1)))).add ((((hasDerivAt_const y (-192678488679977 / 10000000000000 : ℝ)).mul (hasDerivAt_const y (x^0))).mul ((hasDerivAt_id y).pow 2)).mul (hasDerivAt_const y (z^3)))).add ((((hasDerivAt_const y (80282703616657 / 50000000000000 : ℝ)).mul (hasDerivAt_const y (x^0))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^5)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_6_z (x y z : ℝ) :
    HasDerivAt (fun t=>value 6 x y t) (jacobian 6 2 x y z) z := by
  convert (((((((hasDerivAt_const z (0 : ℝ)).add ((((hasDerivAt_const z (80282703616657 / 50000000000000 : ℝ)).mul (hasDerivAt_const z (x^4))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 1))).add ((((hasDerivAt_const z (-192678488679977 / 10000000000000 : ℝ)).mul (hasDerivAt_const z (x^2))).mul (hasDerivAt_const z (y^2))).mul ((hasDerivAt_id z).pow 1))).add ((((hasDerivAt_const z (80282703616657 / 25000000000000 : ℝ)).mul (hasDerivAt_const z (x^2))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 3))).add ((((hasDerivAt_const z (128452325786651 / 10000000000000 : ℝ)).mul (hasDerivAt_const z (x^0))).mul (hasDerivAt_const z (y^4))).mul ((hasDerivAt_id z).pow 1))).add ((((hasDerivAt_const z (-192678488679977 / 10000000000000 : ℝ)).mul (hasDerivAt_const z (x^0))).mul (hasDerivAt_const z (y^2))).mul ((hasDerivAt_id z).pow 3))).add ((((hasDerivAt_const z (80282703616657 / 50000000000000 : ℝ)).mul (hasDerivAt_const z (x^0))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 5))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_7_x (x y z : ℝ) :
    HasDerivAt (fun t=>value 7 t y z) (jacobian 7 0 x y z) x := by
  convert (((((hasDerivAt_const x (0 : ℝ)).add ((((hasDerivAt_const x (849632273398321 / 100000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 4)).mul (hasDerivAt_const x (y^1))).mul (hasDerivAt_const x (z^0)))).add ((((hasDerivAt_const x (-10620403417479 / 625000000000 : ℝ)).mul ((hasDerivAt_id x).pow 2)).mul (hasDerivAt_const x (y^3))).mul (hasDerivAt_const x (z^0)))).add ((((hasDerivAt_const x (10620403417479 / 625000000000 : ℝ)).mul ((hasDerivAt_id x).pow 0)).mul (hasDerivAt_const x (y^3))).mul (hasDerivAt_const x (z^2)))).add ((((hasDerivAt_const x (-849632273398321 / 100000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 0)).mul (hasDerivAt_const x (y^1))).mul (hasDerivAt_const x (z^4)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_7_y (x y z : ℝ) :
    HasDerivAt (fun t=>value 7 x t z) (jacobian 7 1 x y z) y := by
  convert (((((hasDerivAt_const y (0 : ℝ)).add ((((hasDerivAt_const y (849632273398321 / 100000000000000 : ℝ)).mul (hasDerivAt_const y (x^4))).mul ((hasDerivAt_id y).pow 1)).mul (hasDerivAt_const y (z^0)))).add ((((hasDerivAt_const y (-10620403417479 / 625000000000 : ℝ)).mul (hasDerivAt_const y (x^2))).mul ((hasDerivAt_id y).pow 3)).mul (hasDerivAt_const y (z^0)))).add ((((hasDerivAt_const y (10620403417479 / 625000000000 : ℝ)).mul (hasDerivAt_const y (x^0))).mul ((hasDerivAt_id y).pow 3)).mul (hasDerivAt_const y (z^2)))).add ((((hasDerivAt_const y (-849632273398321 / 100000000000000 : ℝ)).mul (hasDerivAt_const y (x^0))).mul ((hasDerivAt_id y).pow 1)).mul (hasDerivAt_const y (z^4)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_7_z (x y z : ℝ) :
    HasDerivAt (fun t=>value 7 x y t) (jacobian 7 2 x y z) z := by
  convert (((((hasDerivAt_const z (0 : ℝ)).add ((((hasDerivAt_const z (849632273398321 / 100000000000000 : ℝ)).mul (hasDerivAt_const z (x^4))).mul (hasDerivAt_const z (y^1))).mul ((hasDerivAt_id z).pow 0))).add ((((hasDerivAt_const z (-10620403417479 / 625000000000 : ℝ)).mul (hasDerivAt_const z (x^2))).mul (hasDerivAt_const z (y^3))).mul ((hasDerivAt_id z).pow 0))).add ((((hasDerivAt_const z (10620403417479 / 625000000000 : ℝ)).mul (hasDerivAt_const z (x^0))).mul (hasDerivAt_const z (y^3))).mul ((hasDerivAt_id z).pow 2))).add ((((hasDerivAt_const z (-849632273398321 / 100000000000000 : ℝ)).mul (hasDerivAt_const z (x^0))).mul (hasDerivAt_const z (y^1))).mul ((hasDerivAt_id z).pow 4))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_8_x (x y z : ℝ) :
    HasDerivAt (fun t=>value 8 t y z) (jacobian 8 0 x y z) x := by
  convert ((((((hasDerivAt_const x (0 : ℝ)).add ((((hasDerivAt_const x (104058276941337 / 20000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 4)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^1)))).add ((((hasDerivAt_const x (-104058276941337 / 2500000000000 : ℝ)).mul ((hasDerivAt_id x).pow 2)).mul (hasDerivAt_const x (y^2))).mul (hasDerivAt_const x (z^1)))).add ((((hasDerivAt_const x (34686092313779 / 10000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 2)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^3)))).add ((((hasDerivAt_const x (34686092313779 / 2500000000000 : ℝ)).mul ((hasDerivAt_id x).pow 0)).mul (hasDerivAt_const x (y^2))).mul (hasDerivAt_const x (z^3)))).add ((((hasDerivAt_const x (-34686092313779 / 20000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 0)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^5)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_8_y (x y z : ℝ) :
    HasDerivAt (fun t=>value 8 x t z) (jacobian 8 1 x y z) y := by
  convert ((((((hasDerivAt_const y (0 : ℝ)).add ((((hasDerivAt_const y (104058276941337 / 20000000000000 : ℝ)).mul (hasDerivAt_const y (x^4))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^1)))).add ((((hasDerivAt_const y (-104058276941337 / 2500000000000 : ℝ)).mul (hasDerivAt_const y (x^2))).mul ((hasDerivAt_id y).pow 2)).mul (hasDerivAt_const y (z^1)))).add ((((hasDerivAt_const y (34686092313779 / 10000000000000 : ℝ)).mul (hasDerivAt_const y (x^2))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^3)))).add ((((hasDerivAt_const y (34686092313779 / 2500000000000 : ℝ)).mul (hasDerivAt_const y (x^0))).mul ((hasDerivAt_id y).pow 2)).mul (hasDerivAt_const y (z^3)))).add ((((hasDerivAt_const y (-34686092313779 / 20000000000000 : ℝ)).mul (hasDerivAt_const y (x^0))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^5)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_8_z (x y z : ℝ) :
    HasDerivAt (fun t=>value 8 x y t) (jacobian 8 2 x y z) z := by
  convert ((((((hasDerivAt_const z (0 : ℝ)).add ((((hasDerivAt_const z (104058276941337 / 20000000000000 : ℝ)).mul (hasDerivAt_const z (x^4))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 1))).add ((((hasDerivAt_const z (-104058276941337 / 2500000000000 : ℝ)).mul (hasDerivAt_const z (x^2))).mul (hasDerivAt_const z (y^2))).mul ((hasDerivAt_id z).pow 1))).add ((((hasDerivAt_const z (34686092313779 / 10000000000000 : ℝ)).mul (hasDerivAt_const z (x^2))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 3))).add ((((hasDerivAt_const z (34686092313779 / 2500000000000 : ℝ)).mul (hasDerivAt_const z (x^0))).mul (hasDerivAt_const z (y^2))).mul ((hasDerivAt_id z).pow 3))).add ((((hasDerivAt_const z (-34686092313779 / 20000000000000 : ℝ)).mul (hasDerivAt_const z (x^0))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 5))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_9_x (x y z : ℝ) :
    HasDerivAt (fun t=>value 9 t y z) (jacobian 9 0 x y z) x := by
  convert ((((hasDerivAt_const x (0 : ℝ)).add ((((hasDerivAt_const x (91975391579759 / 12500000000000 : ℝ)).mul ((hasDerivAt_id x).pow 4)).mul (hasDerivAt_const x (y^1))).mul (hasDerivAt_const x (z^0)))).add ((((hasDerivAt_const x (-441481879582843 / 10000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 2)).mul (hasDerivAt_const x (y^1))).mul (hasDerivAt_const x (z^2)))).add ((((hasDerivAt_const x (91975391579759 / 12500000000000 : ℝ)).mul ((hasDerivAt_id x).pow 0)).mul (hasDerivAt_const x (y^1))).mul (hasDerivAt_const x (z^4)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_9_y (x y z : ℝ) :
    HasDerivAt (fun t=>value 9 x t z) (jacobian 9 1 x y z) y := by
  convert ((((hasDerivAt_const y (0 : ℝ)).add ((((hasDerivAt_const y (91975391579759 / 12500000000000 : ℝ)).mul (hasDerivAt_const y (x^4))).mul ((hasDerivAt_id y).pow 1)).mul (hasDerivAt_const y (z^0)))).add ((((hasDerivAt_const y (-441481879582843 / 10000000000000 : ℝ)).mul (hasDerivAt_const y (x^2))).mul ((hasDerivAt_id y).pow 1)).mul (hasDerivAt_const y (z^2)))).add ((((hasDerivAt_const y (91975391579759 / 12500000000000 : ℝ)).mul (hasDerivAt_const y (x^0))).mul ((hasDerivAt_id y).pow 1)).mul (hasDerivAt_const y (z^4)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_9_z (x y z : ℝ) :
    HasDerivAt (fun t=>value 9 x y t) (jacobian 9 2 x y z) z := by
  convert ((((hasDerivAt_const z (0 : ℝ)).add ((((hasDerivAt_const z (91975391579759 / 12500000000000 : ℝ)).mul (hasDerivAt_const z (x^4))).mul (hasDerivAt_const z (y^1))).mul ((hasDerivAt_id z).pow 0))).add ((((hasDerivAt_const z (-441481879582843 / 10000000000000 : ℝ)).mul (hasDerivAt_const z (x^2))).mul (hasDerivAt_const z (y^1))).mul ((hasDerivAt_id z).pow 2))).add ((((hasDerivAt_const z (91975391579759 / 12500000000000 : ℝ)).mul (hasDerivAt_const z (x^0))).mul (hasDerivAt_const z (y^1))).mul ((hasDerivAt_id z).pow 4))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_10_x (x y z : ℝ) :
    HasDerivAt (fun t=>value 10 t y z) (jacobian 10 0 x y z) x := by
  convert ((((hasDerivAt_const x (0 : ℝ)).add ((((hasDerivAt_const x (29085172607791 / 2500000000000 : ℝ)).mul ((hasDerivAt_id x).pow 4)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^1)))).add ((((hasDerivAt_const x (-232681380862329 / 10000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 2)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^3)))).add ((((hasDerivAt_const x (232681380862329 / 100000000000000 : ℝ)).mul ((hasDerivAt_id x).pow 0)).mul (hasDerivAt_const x (y^0))).mul (hasDerivAt_const x (z^5)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_10_y (x y z : ℝ) :
    HasDerivAt (fun t=>value 10 x t z) (jacobian 10 1 x y z) y := by
  convert ((((hasDerivAt_const y (0 : ℝ)).add ((((hasDerivAt_const y (29085172607791 / 2500000000000 : ℝ)).mul (hasDerivAt_const y (x^4))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^1)))).add ((((hasDerivAt_const y (-232681380862329 / 10000000000000 : ℝ)).mul (hasDerivAt_const y (x^2))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^3)))).add ((((hasDerivAt_const y (232681380862329 / 100000000000000 : ℝ)).mul (hasDerivAt_const y (x^0))).mul ((hasDerivAt_id y).pow 0)).mul (hasDerivAt_const y (z^5)))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

theorem derivative_10_z (x y z : ℝ) :
    HasDerivAt (fun t=>value 10 x y t) (jacobian 10 2 x y z) z := by
  convert ((((hasDerivAt_const z (0 : ℝ)).add ((((hasDerivAt_const z (29085172607791 / 2500000000000 : ℝ)).mul (hasDerivAt_const z (x^4))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 1))).add ((((hasDerivAt_const z (-232681380862329 / 10000000000000 : ℝ)).mul (hasDerivAt_const z (x^2))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 3))).add ((((hasDerivAt_const z (232681380862329 / 100000000000000 : ℝ)).mul (hasDerivAt_const z (x^0))).mul (hasDerivAt_const z (y^0))).mul ((hasDerivAt_id z).pow 5))) using 1 <;> simp [value, jacobian] <;> try ring <;> simp

def coordinate (q : Fin 3) (x y z : ℝ) : ℝ := ![x,y,z] q

theorem exact_partial_derivative (channel : Fin 11) (component : Fin 3) (x y z : ℝ) :
    HasDerivAt (fun t=>value channel
      (if component=0 then t else x) (if component=1 then t else y) (if component=2 then t else z))
      (jacobian channel component x y z) (coordinate component x y z) := by
  fin_cases channel <;> fin_cases component
  · simpa [Polynomial.coordinate] using derivative_0_x x y z
  · simpa [Polynomial.coordinate] using derivative_0_y x y z
  · simpa [Polynomial.coordinate] using derivative_0_z x y z
  · simpa [Polynomial.coordinate] using derivative_1_x x y z
  · simpa [Polynomial.coordinate] using derivative_1_y x y z
  · simpa [Polynomial.coordinate] using derivative_1_z x y z
  · simpa [Polynomial.coordinate] using derivative_2_x x y z
  · simpa [Polynomial.coordinate] using derivative_2_y x y z
  · simpa [Polynomial.coordinate] using derivative_2_z x y z
  · simpa [Polynomial.coordinate] using derivative_3_x x y z
  · simpa [Polynomial.coordinate] using derivative_3_y x y z
  · simpa [Polynomial.coordinate] using derivative_3_z x y z
  · simpa [Polynomial.coordinate] using derivative_4_x x y z
  · simpa [Polynomial.coordinate] using derivative_4_y x y z
  · simpa [Polynomial.coordinate] using derivative_4_z x y z
  · simpa [Polynomial.coordinate] using derivative_5_x x y z
  · simpa [Polynomial.coordinate] using derivative_5_y x y z
  · simpa [Polynomial.coordinate] using derivative_5_z x y z
  · simpa [Polynomial.coordinate] using derivative_6_x x y z
  · simpa [Polynomial.coordinate] using derivative_6_y x y z
  · simpa [Polynomial.coordinate] using derivative_6_z x y z
  · simpa [Polynomial.coordinate] using derivative_7_x x y z
  · simpa [Polynomial.coordinate] using derivative_7_y x y z
  · simpa [Polynomial.coordinate] using derivative_7_z x y z
  · simpa [Polynomial.coordinate] using derivative_8_x x y z
  · simpa [Polynomial.coordinate] using derivative_8_y x y z
  · simpa [Polynomial.coordinate] using derivative_8_z x y z
  · simpa [Polynomial.coordinate] using derivative_9_x x y z
  · simpa [Polynomial.coordinate] using derivative_9_y x y z
  · simpa [Polynomial.coordinate] using derivative_9_z x y z
  · simpa [Polynomial.coordinate] using derivative_10_x x y z
  · simpa [Polynomial.coordinate] using derivative_10_y x y z
  · simpa [Polynomial.coordinate] using derivative_10_z x y z

/-- Standard Jacobian-transpose times the actual eleven upstream components. -/
def vjp (x y z : ℝ) (upstream : Fin 11 → ℝ) (component : Fin 3) : ℝ :=
  ∑ channel, upstream channel * jacobian channel component x y z

#specNonCircular value avoiding [VeriTile.Triton.FP.Structural.exec]
#axiomsClean exact_partial_derivative
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Polynomial
