import bench.repairs.tritonbench_g.fused_recurrent_retention.AddressCheck
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.SourceBounds
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Admission

theorem radius_bounds (limits : Nat → Nat) (env : Nat → Int)
    (bound : ∀ i,(env i).natAbs≤limits i) (e : SExpr) :
    (e.instantiate env).value.natAbs≤e.radius limits := by
  induction e with
  | const x => exact Nat.le_refl _
  | var i => exact bound i
  | add a b ha hb => exact (Int.natAbs_add_le _ _).trans (Nat.add_le_add ha hb)
  | sub a b ha hb => exact (Int.natAbs_sub_le _ _).trans (Nat.add_le_add ha hb)
  | mul a b ha hb => simpa only [SExpr.instantiate,IExpr.value,Int.natAbs_mul,SExpr.radius] using Nat.mul_le_mul ha hb

theorem signed_of_radius (x : Int) (r : Nat) (bound : x.natAbs≤r) (small : r<2^bits) :
    -(2^bits)≤x ∧ x<2^bits := by
  have hp := Int.le_natAbs (a:=x)
  have hn := Int.le_natAbs (a:=-x)
  simp only [Int.natAbs_neg] at hn
  have hh : (r : Int)<2^bits := by exact_mod_cast small
  have hb : (x.natAbs : Int)≤r := Int.ofNat_le.mpr bound
  constructor <;> omega

theorem checked_expr (limits : Nat → Nat) (env : Nat → Int)
    (bound : ∀ i,(env i).natAbs≤limits i) (e : SExpr) :
    e.check limits=Bool.true → (e.instantiate env).Safe := by
  induction e with
  | const x =>
      intro ok
      simp only [SExpr.check,SExpr.radius,Bool.and_true,decide_eq_true_eq] at ok
      exact signed_of_radius x x.natAbs (Nat.le_refl _) ok
  | var i =>
      intro ok
      simp only [SExpr.check,SExpr.radius,Bool.and_true,decide_eq_true_eq] at ok
      exact signed_of_radius _ _ (bound i) ok
  | add a b ha hb =>
      intro ok
      change (decide ((SExpr.add a b).radius limits<2^31) && (a.check limits && b.check limits))=Bool.true at ok
      simp only [Bool.and_eq_true,decide_eq_true_eq] at ok
      exact ⟨ha ok.2.1,hb ok.2.2,signed_of_radius _ _ (radius_bounds limits env bound (.add a b)) ok.1⟩
  | sub a b ha hb =>
      intro ok
      change (decide ((SExpr.sub a b).radius limits<2^31) && (a.check limits && b.check limits))=Bool.true at ok
      simp only [Bool.and_eq_true,decide_eq_true_eq] at ok
      exact ⟨ha ok.2.1,hb ok.2.2,signed_of_radius _ _ (radius_bounds limits env bound (.sub a b)) ok.1⟩
  | mul a b ha hb =>
      intro ok
      change (decide ((SExpr.mul a b).radius limits<2^31) && (a.check limits && b.check limits))=Bool.true at ok
      simp only [Bool.and_eq_true,decide_eq_true_eq] at ok
      exact ⟨ha ok.2.1,hb ok.2.2,signed_of_radius _ _ (radius_bounds limits env bound (.mul a b)) ok.1⟩

theorem env_bounds (s : Shape) (iv ik h lk lv r : Nat) (hb : 0<s.B) (hh : 0<s.H)
    (hiv : iv<s.NV) (hik : ik<s.NK) (hhidx : h<s.BH)
    (hlk : lk<s.BK) (hlv : lv<s.BV) (hr : r≤ s.T) :
    ∀ i,(env s iv ik h lk lv r i).natAbs≤limits s i := by
  have hbatch : h/s.H<s.B := (Nat.div_lt_iff_lt_mul hh).mpr hhidx
  have hhead : h%s.H<s.H := Nat.mod_lt _ hh
  intro i
  rcases i with _ | (_ | (_ | (_ | (_ | (_ | (_ | (_ | i))))))) <;> simp only [env,limits,Int.natAbs_natCast] <;> omega

theorem instantiate_roots (phase : SourcePhase) (seed extra : Bool) (s : Shape) (b : InputBuffers)
    (iv ik h lk lv r : Nat) :
    (roots phase seed extra s b).map (SExpr.instantiate (env s iv ik h lk lv r))=
      SourceAddress.roots phase seed extra s b iv ik h lk lv := by
  cases phase <;> cases seed <;> cases extra <;> rfl

theorem instantiate_formed (phase : SourcePhase) (seed extra : Bool) (s : Shape) (b : InputBuffers)
    (iv ik h lk lv r : Nat) :
    (formed phase seed extra s b).map (fun e=>(e.instantiate (env s iv ik h lk lv r)).value)=
      SourceAddress.formed phase seed extra s b iv ik h lk lv r := by
  cases phase <;> cases seed <;> cases extra <;>
    simp [formed,SourceAddress.formed,SExpr.instantiate,IExpr.value,env,sNat,
      Nat.cast_add,Nat.cast_mul,Nat.cast_ofNat]

theorem nat_signed32 (n : Nat) (hn : n<2^31) : signed32 n :=
  signed_of_radius n n (by simp) hn

theorem checked_bounds (phase : SourcePhase) (seed extra : Bool) (s : Shape) (b : InputBuffers)
    (ok : addressCheck phase seed extra s b=Bool.true) :
    SourceAddress.Bounds phase seed extra s b := by
  simp only [addressCheck,Bool.and_eq_true,decide_eq_true_eq] at ok
  rcases ok with ⟨⟨⟨⟨⟨⟨positive,args⟩,qv⟩,cot⟩,grid⟩,exprs⟩,ptrs⟩
  refine ⟨?_,?_,?_,grid,?_,?_⟩
  · intro n hn
    exact nat_signed32 n (by simpa using List.all_eq_true.mp args n hn)
  · intro i
    have hi := List.all_eq_true.mp qv i.val (List.mem_range.mpr i.isLt)
    have hm : i.val%3=i.val := Nat.mod_eq_of_lt i.isLt
    simp only [hm,decide_eq_true_eq] at hi
    exact ⟨nat_signed32 _ hi.1,nat_signed32 _ hi.2⟩
  · intro hp i
    subst phase
    simp only [ite_true] at cot
    have hi := List.all_eq_true.mp cot i (List.mem_finRange i)
    simp only [decide_eq_true_eq] at hi
    exact ⟨nat_signed32 _ hi.1,fun he=>nat_signed32 _ (hi.2 he)⟩
  · intro iv hv ik hk h hh lk hlk lv hlv e he
    rw [←instantiate_roots phase seed extra s b iv ik h lk lv 0] at he
    obtain ⟨x,hx,rfl⟩ := List.mem_map.mp he
    exact checked_expr (limits s) _
      (env_bounds s iv ik h lk lv 0 positive.1 positive.2 hv hk hh hlk hlv (Nat.zero_le _)) x
      (List.all_eq_true.mp exprs x hx)
  · intro iv hv ik hk h hh lk hlk lv hlv r hr offset ho
    rw [←instantiate_formed phase seed extra s b iv ik h lk lv r] at ho
    obtain ⟨x,hx,rfl⟩ := List.mem_map.mp ho
    have hb := radius_bounds (limits s) _
      (env_bounds s iv ik h lk lv r positive.1 positive.2 hv hk hh hlk hlv hr) x
    have hs : 4*x.radius (limits s)<2^63 := by simpa using List.all_eq_true.mp ptrs x hx
    apply signed_of_radius _ (4*x.radius (limits s)) _ hs
    simpa only [Int.natAbs_mul,Int.natAbs_natCast] using Nat.mul_le_mul_left 4 hb

#axiomsClean checked_bounds
#axiomsClean checked_expr
#axiomsClean env_bounds
#axiomsClean instantiate_roots
#axiomsClean instantiate_formed
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Admission
