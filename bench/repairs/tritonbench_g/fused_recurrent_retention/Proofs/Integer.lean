import bench.repairs.tritonbench_g.fused_recurrent_retention.AddressSource
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Backward
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section

theorem IExpr.machine_bits (e : IExpr) : e.machine=BitVec.ofInt 32 e.value := by
  induction e with
  | atom x => rfl
  | add a b ha hb => simp only [machine,value,ha,hb,BitVec.ofInt_add]
  | sub a b ha hb => simp only [machine,value,ha,hb,Int.sub_eq_add_neg,BitVec.ofInt_add,BitVec.ofInt_neg,BitVec.sub_eq_add_neg]
  | mul a b ha hb => simp only [machine,value,ha,hb,BitVec.ofInt_mul]

theorem IExpr.safe_value (e : IExpr) (h : e.Safe) : signed32 e.value := by
  cases e with
  | atom x => exact h
  | add a b => exact h.2.2
  | sub a b => exact h.2.2
  | mul a b => exact h.2.2

theorem IExpr.machine_exact (e : IExpr) (h : e.Safe) : e.machine.toInt=e.value := by
  rw [e.machine_bits]
  exact BitVec.toInt_ofInt_eq_self (by decide) (e.safe_value h).1 (e.safe_value h).2

private def N (n : Nat) : IExpr := .atom n
private def A := IExpr.add
private def M := IExpr.mul
private def S := IExpr.sub

/-- The signed32 subexpressions formed by both JITs. Padded lanes are included.
Pointer additions are separate signed64 operations; their sums are not falsely
required to fit signed32. Time includes T (dead forward increments), and the
reverse rebase includes -1 when T=0. -/
def integerRoots (s : Shape) (b : InputBuffers) (iv ik h lk lv : Nat) : List IExpr :=
  let key := A (M (N ik) (N s.BK)) (N lk)
  let val := A (M (N iv) (N s.BV)) (N lv)
  let hh := N (h%s.H)
  let batch := N (h/s.H)
  let doBase := A (A (M batch (N (b.do_.strides 0))) (M hh (N (b.do_.strides 1))))
    (M val (N (b.do_.strides 3)))
  let finalBase := A (A (A (M batch (N (b.finalGrad.strides 0)))
    (M hh (N (b.finalGrad.strides 1)))) (M key (N (b.finalGrad.strides 2))))
    (M val (N (b.finalGrad.strides 3)))
  [N s.B,N s.H,N s.T,N s.DK,N s.DV,N s.BK,N s.BV,N s.NK,N s.NV,
   M (N s.B) (N s.H),N iv,N ik,N h,N lk,N lv,hh,batch,
   N (b.q.strides 1),N (b.q.strides 2),N (b.q.strides 3),
   N (b.v.strides 1),N (b.v.strides 2),N (b.v.strides 3),
   N (b.do_.strides 0),N (b.do_.strides 1),N (b.do_.strides 2),N (b.do_.strides 3),
   N (b.finalGrad.strides 0),N (b.finalGrad.strides 1),N (b.finalGrad.strides 2),N (b.finalGrad.strides 3),
   key,val,M (N h) (N (b.q.strides 1)),M (N h) (N (b.v.strides 1)),
   M (A (N h) (M (M (N ik) (N s.B)) (N s.H))) (N (b.v.strides 1)),
   M (A (N h) (M (M (N iv) (N s.B)) (N s.H))) (N (b.q.strides 1)),
   M (M (N h) (N s.DK)) (N s.DV),M key (N s.DV),
   doBase,finalBase,M (S (N s.T) (N 1)) (N s.DK),
   M (S (N s.T) (N 1)) (N s.DV),M (S (N s.T) (N 1)) (N (b.do_.strides 2))]

/-- All pointer displacements, including masked lanes, T=0 rebases, the final
forward increment and the final reverse retreat. Offsets are in elements. -/
def pointerDisplacements (s : Shape) (b : InputBuffers) (iv ik h lk lv : Nat) (time : Int) : List Int :=
  let k : Int := ik*s.BK+lk
  let v : Int := iv*s.BV+lv
  let qbase : Int := h*b.q.strides 1
  let vbase : Int := h*b.v.strides 1
  let dobase : Int := (h/s.H)*b.do_.strides 0+(h%s.H)*b.do_.strides 1
  [b.q.offset+qbase+k+time*s.DK,b.k.offset+qbase+k+time*s.DK,
   b.v.offset+vbase+v+time*s.DV,
   b.do_.offset+dobase+v*b.do_.strides 3+time*b.do_.strides 2,
   (h+ik*s.B*s.H)*b.v.strides 1+v+time*s.DV,
   (h+iv*s.B*s.H)*b.q.strides 1+k+time*s.DK,
   b.seed.offset+h*s.DK*s.DV+k*s.DV+v,
   h*s.DK*s.DV+k*s.DV+v,
   b.finalGrad.offset+(h/s.H)*b.finalGrad.strides 0+(h%s.H)*b.finalGrad.strides 1+
     k*b.finalGrad.strides 2+v*b.finalGrad.strides 3]

structure IntegerDomain (s : Shape) (b : InputBuffers) : Prop where
  args : ∀ n∈[s.B,s.H,s.T,s.DK,s.DV,s.BK,s.BV,s.NK,s.NV,s.BH],signed32 n
  grid : s.NV < 2^31 ∧ s.NK ≤ 65535 ∧ s.BH ≤ 65535
  expressions : ∀ iv<s.NV,∀ ik<s.NK,∀ h<s.BH,∀ lk<s.BK,∀ lv<s.BV,
    ∀ e∈integerRoots s b iv ik h lk lv,e.Safe
  pointers : ∀ iv<s.NV,∀ ik<s.NK,∀ h<s.BH,∀ lk<s.BK,∀ lv<s.BV,
    ∀ time : Int,-1 ≤ time → time ≤ s.T →
      ∀ off∈pointerDisplacements s b iv ik h lk lv time,signed64 (4*off)

theorem all_source_i32_exact (s : Shape) (b : InputBuffers) (domain : IntegerDomain s b)
    (iv ik h lk lv : Nat) (hv : iv<s.NV) (hk : ik<s.NK) (hh : h<s.BH)
    (hlk : lk<s.BK) (hlv : lv<s.BV) (e : IExpr)
    (he : e∈integerRoots s b iv ik h lk lv) : e.machine.toInt=e.value :=
  e.machine_exact (domain.expressions iv hv ik hk h hh lk hlk lv hlv e he)

theorem reverse_time (T r : Nat) (hr : r<T) :
    (T : Int)-1-r=(T-1-r : Nat) := by omega

theorem dead_reverse_time (T : Nat) : (T : Int)-1-T = -1 := by omega

theorem zero_reverse_rebase : (0 : Int)-1 = -1 := rfl

#axiomsClean IExpr.machine_exact
#axiomsClean all_source_i32_exact
#axiomsClean reverse_time
#axiomsClean dead_reverse_time
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
