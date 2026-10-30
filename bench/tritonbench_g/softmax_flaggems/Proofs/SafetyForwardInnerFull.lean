import bench.tritonbench_g.softmax_flaggems.Proofs.SafetyForwardInnerSource

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety.ForwardInner
open VeriTile Triton FP.Structural Safety
open SoftmaxFlaggemsFiniteSeed FPReference FPLoop FPForwardInner
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Actual safety for all four loops, including unmasked aligned chunks and
reverse output traversal. The numerical loop invariants are reused unchanged. -/
theorem source (A : Algebra α) (s : State α) (X Y : RegionName)
    (rows N B : Nat) (hB : 0<B) (hBN : B<N) (hrow : s.pids 0<rows)
    (bounds : RegionName → Nat)
    (hx : ∀ k,k<rows*N → k<bounds X) (hy : ∀ k,k<rows*N → k<bounds Y) :
    TraceSafe A bounds (Source.forward_inner Y X rows N B Bool.false) s := by
  have hN : 0<N := lt_trans hB hBN
  let row := s.pids 0
  let seed : Acc α B := ((fun _ => literal A (-340282346638528859811704183484516925440.0)),fun _ => literal A 0)
  let f₀ := accumulate A s X Bool.false row N B
  let first := (walk f₀ (previous N B) B 0 seed).2
  obtain ⟨mid₀,he₀,hi₀⟩ := range_model A "start_n" (rangeBody (code Y X rows N B) 5)
    f₀ (fun _ v st => Ready s X Y row N B v st) (previous N B) B
    (fun cur v st _ hs => accumulate_step A s st X Y Bool.false row rows N B cur v hs)
    0 seed (initial A s X Y N B) (initial_ready A s X Y N B)
  let f₁ := accumulate A s X Bool.true row N B
  let v := (walk f₁ N B (previous N B) first).2
  obtain ⟨mid₁,he₁,hi₁⟩ := range_model A "start_n" (rangeBody (code Y X rows N B) 6)
    f₁ (fun _ v st => Ready s X Y row N B v st) N B
    (fun cur v st _ hs => accumulate_step A s st X Y Bool.true row rows N B cur v hs)
    (previous N B) first mid₀ hi₀
  obtain ⟨_,ready⟩ := after_accumulate A s mid₁ X Y row rows N B hB v hi₁
  let mx := maxReduced A B v
  let z := sumReduced A B v
  let out₀ := (walk (emit A X Y Bool.true row N B mx z) B B 0 s).2
  obtain ⟨mid₂,he₂,hi₂⟩ := range_model A "start_n" (rangeBody (code Y X rows N B) 11)
    (emit A X Y Bool.true row N B mx z)
    (fun _ base st => WriteReady base X Y row N B mx z st) B B
    (fun cur base st _ hs => emit_step A base st X Y Bool.true row rows N B cur mx z hs)
    0 s (writeInitial A mid₁ N B v) ready
  have safe₀ := range_safe A bounds "start_n" (rangeBody (code Y X rows N B) 5)
    (fun cur v st => cur%B=0 ∧ Ready s X Y row N B v st) f₀ (previous N B) B
    (fun cur v st hc hs => accumulate_safe A s st X Y Bool.false row rows N B cur v hs.2 hrow bounds hx
      (fun i _ => first_bound N B cur hN hB hs.1 hc i))
    (fun cur v st _ hs => by
      obtain ⟨t,he,ht⟩ := accumulate_step A s st X Y Bool.false row rows N B cur v hs.2
      exact ⟨t,he,by simpa [Nat.add_mod] using hs.1,ht⟩)
    0 seed (initial A s X Y N B) ⟨Nat.zero_mod _,initial_ready A s X Y N B⟩
  have safe₁ := range_safe A bounds "start_n" (rangeBody (code Y X rows N B) 6)
    (fun _ v st => Ready s X Y row N B v st) f₁ N B
    (fun cur v st _ hs => accumulate_safe A s st X Y Bool.true row rows N B cur v hs hrow bounds hx
      (fun i hi => by simpa [active] using hi))
    (fun cur v st _ hs => accumulate_step A s st X Y Bool.true row rows N B cur v hs)
    (previous N B) first mid₀ hi₀
  have safe₂ := range_safe A bounds "start_n" (rangeBody (code Y X rows N B) 11)
    (fun _ base st => WriteReady base X Y row N B mx z st)
    (emit A X Y Bool.true row N B mx z) B B
    (fun cur base st _ hs => emit_safe A base st X Y Bool.true row rows N B cur mx z hs hrow bounds hx hy
      (fun i hi => by simpa [active] using hi))
    (fun cur base st _ hs => emit_step A base st X Y Bool.true row rows N B cur mx z hs)
    0 s (writeInitial A mid₁ N B v) ready
  have safe₃ := range_safe A bounds "start_n" (rangeBody (code Y X rows N B) 12)
    (fun cur base st => B≤cur ∧ WriteReady base X Y row N B mx z st)
    (emit A X Y Bool.false row N B mx z) N B
    (fun cur base st _ hs => emit_safe A base st X Y Bool.false row rows N B cur mx z hs.2 hrow bounds hx hy
      (fun i _ => reverse_bound N B cur hN hB hBN hs.1 i))
    (fun cur base st _ hs => by
      obtain ⟨t,he,ht⟩ := emit_step A base st X Y Bool.false row rows N B cur mx z hs.2
      exact ⟨t,he,Nat.le_trans hs.1 (Nat.le_add_right cur B),ht⟩)
    B out₀ mid₂ ⟨le_refl _,hi₂⟩
  apply (initial_safety A s X Y rows N B bounds).mpr
  refine ⟨safe₀,?_⟩
  intro t ht
  have : t=mid₀ := Option.some.inj (ht.symm.trans he₀)
  subst t
  apply (between_safety A s mid₀ X Y row rows N B first hi₀ bounds).mpr
  refine ⟨safe₁,?_⟩
  intro t ht
  have : t=mid₁ := Option.some.inj (ht.symm.trans he₁)
  subst t
  apply (after_safety A s mid₁ X Y row rows N B hB v hi₁ bounds).mpr
  refine ⟨safe₂,?_⟩
  intro t ht
  have : t=mid₂ := Option.some.inj (ht.symm.trans he₂)
  subst t
  exact (last_safety A mid₂ X Y rows N B bounds).mpr safe₃

#axiomsClean source
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.SourceSafety.ForwardInner
