import bench.tritonbench_g.triton_conv2d_fwd.Proofs.Coordinates
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.Ownership
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.ISA
namespace VeriTile.Bench.Repairs.Conv2d

structure SourceSafe (s : Shape) (inputWords weightWords : Nat) : Prop where
  parameters : ∀ z∈scalarParameters s,Signed32 z
  coordinateFormed : ∀ g r f h w c,g<s.groups → r<pSpatial s → f<pFeatures s →
    h<s.kh → w<s.kw → c<s.padded → ∀ z∈coordinates s g r f h w c,Signed32 z
  pointers : ∀ g r f h w c,g<s.groups → r<pSpatial s → f<pFeatures s →
    h<s.kh → w<s.kw → c<s.padded → Signed32 (inputPointer s g r h w c) ∧
      Signed32 (weightPointer s g f h w c) ∧ Signed32 (outputPointer s ⟨g,r,f⟩)
  inputFormed : ∀ g r h w c,g<s.groups → r<pSpatial s → h<s.kh → w<s.kw → c<s.padded →
    ∀ z∈inputIntermediates s g r h w c,Signed32 z
  weightFormed : ∀ g f h w c,g<s.groups → f<pFeatures s → h<s.kh → w<s.kw → c<s.padded →
    ∀ z∈weightIntermediates s g f h w c,Signed32 z
  outputFormed : ∀ x,x.group<s.groups → x.spatial<pSpatial s → x.feature<pFeatures s →
    ∀ z∈outputIntermediates s x,Signed32 z
  inputMemory : ∀ g r h w c,g<s.groups → batch s r<s.n → c<s.cg →
    0 ≤ ih s r h → ih s r h<s.h → 0 ≤ iw s r w → iw s r w<s.w →
      0 ≤ inputPointer s g r h w c ∧ (inputPointer s g r h w c).toNat< inputWords
  weightMemory : ∀ g f h w c,g<s.groups → f<s.og → c<s.cg → h<s.kh → w<s.kw →
    weightPointer s g f h w c<weightWords
  outputMemory : ∀ x,s.Active x → outputPointer s x<s.size
  covers : ∀ address,address<s.size → ∃ x,s.Active x ∧ outputPointer s x=address
  unique : ∀ p q i j,i∈lanes → j∈lanes → s.Active (cellOf p i) → s.Active (cellOf q j) →
    outputPointer s (cellOf p i)=outputPointer s (cellOf q j) → p=q ∧ i=j

theorem source_safe (s : Shape) (v : Valid s) (domain : envelope s<2^31)
    (inputWords weightWords : Nat) (input : s.n≠0 → inputTop s< inputWords)
    (weight : weightTop s<weightWords) : SourceSafe s inputWords weightWords := by
  refine ⟨?_,?_,pointers_signed32 s v domain,?_,?_,?_,?_,?_,address_bound s v,address_covers s v,?_⟩
  · intro z hz
    have hi : (z:Int) ≤ envelope s := by exact_mod_cast parameters_bound s z hz
    exact signed32_of_bound s z domain (by omega) hi
  · intro g r f h w c hg hr hf hh hw hc z hz
    have hi : (z:Int) ≤ envelope s := by exact_mod_cast coordinates_bound s v g r f h w c hg hr hf hh hw hc z hz
    exact signed32_of_bound s z domain (by omega) hi
  · intro g r h w c hg hr hh hw hc z hz
    have b := input_intermediates_bound s v g r h w c hg hr hh hw hc z hz
    exact signed32_of_bound s z domain b.1 b.2
  · intro g f h w c hg hf hh hw hc z hz
    have hi : (z:Int) ≤ envelope s := by
      exact_mod_cast weight_intermediates_bound s v g f h w c z hg hf hh hw hc hz
    exact signed32_of_bound s z domain (by omega) hi
  · intro x hg hr hf z hz
    have hi : (z:Int) ≤ envelope s := by exact_mod_cast output_intermediates_bound s v x z hg hr hf hz
    exact signed32_of_bound s z domain (by omega) hi
  · intro g r h w c hg hn hc hy0 hy hx0 hx
    have b := input_in_bounds s g r h w c hg hn hc hy0 hy hx0 hx
    exact ⟨b.1,lt_of_lt_of_le b.2 (input (by omega))⟩
  · intro g f h w c hg hf hc hh hw
    exact lt_of_lt_of_le (weight_in_bounds s g f h w c hg hf hc hh hw) weight
  · intro p q i j hi hj hai haj eq
    exact source_owner_unique p q i j hi hj (address_injective s v _ _ hai haj eq)

theorem scaled_pointer_signed64 (z : Int) (h : Signed32 z) (byte : Fin 4) :
    -(2^63:Int) ≤ 4*z+byte.val ∧ 4*z+byte.val<2^63 := by
  unfold Signed32 at h
  have b := byte.isLt
  omega

#axiomsClean source_safe
#axiomsClean scaled_pointer_signed64
end VeriTile.Bench.Repairs.Conv2d
