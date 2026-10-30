import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.IO

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface.Binding
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section
attribute [local instance] Classical.propDecidable
variable {α : Type} [Inhabited α]

structure Agrees (c : Shape) (x y : Input α) : Prop where
  q : ∀ z, z<c.Z → ∀ t, t<c.T → ∀ e, e<c.K → x.q z t e=y.q z t e
  k : ∀ z, z<c.Z → ∀ t, t<c.T → ∀ e, e<c.K → x.k z t e=y.k z t e
  v : ∀ z, z<c.Z → ∀ t, t<c.T → ∀ f, f<c.V → x.v z t f=y.v z t f
  w : ∀ z, z<c.Z → ∀ t, t<c.T → ∀ e, e<c.K → x.w z t e=y.w z t e
  u : ∀ z, z<c.Z → ∀ e, e<c.K → x.u z e=y.u z e
  h0 : c.initial=Bool.true → ∀ z, z<c.Z → ∀ e, e<c.K → ∀ f, f<c.V → x.h0 z e f=y.h0 z e f

theorem initial (M : Algebra α) (c : Shape) (hc : Sizes c) (scale : α)
    (xs : Inputs α c) (s : State α) (live : Launches c)
    (binding : (programIO M c scale).InputBinding xs s) :
    Agrees c (inputs c (Source.words s.mem)) (logicalInputs c xs) := by
  constructor
  · intro z hz t ht e he
    have h := binding (0 : Fin 6) (⟨z,hz⟩,⟨t,ht⟩,⟨e,he⟩,()) ⟨live,Or.inl (by change (0 : Nat)≠5; decide)⟩
    simpa [programIO,inputPorts,inputRead,Source.words,Source.regionName,logicalInputs,inputs,hz,ht,he,mapValue] using h
  · intro z hz t ht e he
    have h := binding (1 : Fin 6) (⟨z,hz⟩,⟨t,ht⟩,⟨e,he⟩,()) ⟨live,Or.inl (by change (1 : Nat)≠5; decide)⟩
    simpa [programIO,inputPorts,inputRead,Source.words,Source.regionName,logicalInputs,inputs,hz,ht,he,mapValue] using h
  · intro z hz t ht f hf
    have h := binding (2 : Fin 6) (⟨z,hz⟩,⟨t,ht⟩,⟨f,hf⟩,()) ⟨live,Or.inl (by change (2 : Nat)≠5; decide)⟩
    simpa [programIO,inputPorts,inputRead,Source.words,Source.regionName,logicalInputs,inputs,hz,ht,hf,mapValue] using h
  · intro z hz t ht e he
    have h := binding (3 : Fin 6) (⟨z,hz⟩,⟨t,ht⟩,⟨e,he⟩,()) ⟨live,Or.inl (by change (3 : Nat)≠5; decide)⟩
    simpa [programIO,inputPorts,inputRead,Source.words,Source.regionName,logicalInputs,inputs,hz,ht,he,mapValue] using h
  · intro z hz e he
    have hm := Nat.mod_lt z hc.H
    have h := binding (4 : Fin 6) (⟨z%c.H,hm⟩,⟨e,he⟩,()) ⟨live,Or.inl (by change (4 : Nat)≠5; decide)⟩
    simpa [programIO,inputPorts,inputRead,Source.words,Source.regionName,logicalInputs,inputs,biasAddress,hm,he,mapValue] using h
  · intro initial z hz e he f hf
    have h := binding (5 : Fin 6) (⟨z,hz⟩,⟨e,he⟩,⟨f,hf⟩,()) ⟨live,Or.inr initial⟩
    simpa [programIO,inputPorts,inputRead,Source.words,Source.regionName,logicalInputs,inputs,hz,he,hf,mapValue] using h

theorem seed (a : Ops α) (c : Shape) (x y : Input α) (h : Agrees c x y)
    (z e f : Nat) (hz : z<c.Z) : scalarSeed a c x z e f=scalarSeed a c y z e f := by
  unfold scalarSeed
  by_cases hi : c.initial=Bool.true
  · rw [if_pos hi,if_pos hi]
    by_cases active : e<c.K ∧ f<c.V
    · rw [if_pos active,if_pos active,h.h0 hi z hz e active.1 f active.2]
    · rw [if_neg active,if_neg active]
  · rw [if_neg hi,if_neg hi]

theorem outer (a : Ops α) (c : Shape) (x y : Input α) (h : Agrees c x y)
    (z n e f : Nat) (hz : z<c.Z) (hn : n<c.T) :
    scalarOuter a c x z n e f=scalarOuter a c y z n e f := by
  unfold scalarOuter
  have hk : (if e<c.K then x.k z (visit c n) e else a.zero)=
      (if e<c.K then y.k z (visit c n) e else a.zero) := by
    split_ifs with he
    · exact h.k z hz _ (visit_lt c hn) e he
    · rfl
  have hv : (if f<c.V then x.v z (visit c n) f else a.zero)=
      (if f<c.V then y.v z (visit c n) f else a.zero) := by
    split_ifs with hf
    · exact h.v z hz _ (visit_lt c hn) f hf
    · rfl
  rw [hk,hv]

theorem state (a : Ops α) (c : Shape) (x y : Input α) (h : Agrees c x y)
    (z e f n : Nat) (hz : z<c.Z) (hn : n≤c.T) :
    scalarState a c x z e f n=scalarState a c y z e f n := by
  induction n with
  | zero => exact seed a c x y h z e f hz
  | succ n ih =>
    rw [scalarState,scalarState,ih (by omega),outer a c x y h z n e f hz (by omega)]
    by_cases he : e<c.K
    · rw [if_pos he,if_pos he,h.w z hz _ (visit_lt c (by omega)) e he]
    · rw [if_neg he,if_neg he]

theorem term (a : Ops α) (c : Shape) (x y : Input α) (h : Agrees c x y)
    (scale : α) (z n e f : Nat) (hz : z<c.Z) (hn : n<c.T) :
    scalarTerm a c x scale z n e f=scalarTerm a c y scale z n e f := by
  unfold scalarTerm
  rw [state a c x y h z e f n hz (by omega),outer a c x y h z n e f hz hn]
  by_cases he : e<c.K
  · simp only [if_pos he,h.u z hz e he,h.q z hz _ (visit_lt c hn) e he]
  · simp only [if_neg he]

theorem partial_eq (a : Ops α) (c : Shape) (plan : Plans c) (x y : Input α) (h : Agrees c x y)
    (scale : α) (z ik n f : Nat) (hz : z<c.Z) (hn : n<c.T) :
    scalarPartial a c plan x scale z ik n f=scalarPartial a c plan y scale z ik n f := by
  unfold scalarPartial
  congr 1
  funext j
  exact term a c x y h scale z n _ f hz hn

#axiomsClean partial_eq
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface.Binding
