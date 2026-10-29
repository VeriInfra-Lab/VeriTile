import VeriTile.Triton.Float.StepR
import VeriTile.Meta.StatementAudit

/-! Relational loop support for a masked-pointer normalization proof. A pointer
register used only under a mask may differ on inactive lanes. The relation
retains all memory, launch coordinates and every other register. This file
alone does not prove a convolution source/model equivalence. -/
namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.MaskedPointerSimulation
open VeriTile.Triton

structure AgreeExcept (ignored : RegName) (s t : BlockState) : Prop where
  memory : s.mem = t.mem
  pids : s.pids = t.pids
  numPids : s.numPids = t.numPids
  undef : s.undef = t.undef
  registers : ∀ dtype shape name, name ≠ ignored → s.regs dtype shape name = t.regs dtype shape name

theorem AgreeExcept.refl (ignored : RegName) (s : BlockState) : AgreeExcept ignored s s :=
  ⟨rfl,rfl,rfl,rfl,fun _ _ _ _ => rfl⟩

theorem AgreeExcept.setReg {ignored : RegName} {s t : BlockState}
    (h : AgreeExcept ignored s t) (name : RegName) (dtype : TileDType)
    (shape : TileShape) (value : Tile dtype shape) :
    AgreeExcept ignored (s.setReg name dtype shape value) (t.setReg name dtype shape value) := by
  refine ⟨h.memory,h.pids,h.numPids,h.undef,?_⟩
  intro d sh n hn
  by_cases same : n = name
  · simp only [BlockState.setReg,same,↓reduceDIte]
  · simpa only [BlockState.setReg,same,↓reduceDIte] using h.registers d sh n hn

theorem AgreeExcept.writeIgnored {ignored : RegName} {s t : BlockState}
    (h : AgreeExcept ignored s t) (dtype : TileDType) (shape : TileShape)
    (sourceValue modelValue : Tile dtype shape) :
    AgreeExcept ignored (s.setReg ignored dtype shape sourceValue)
      (t.setReg ignored dtype shape modelValue) := by
  refine ⟨h.memory,h.pids,h.numPids,h.undef,?_⟩
  intro d sh n hn
  simpa only [BlockState.setReg,hn,↓reduceDIte] using h.registers d sh n hn

/-- Lift a local, checked source/model step simulation through a range.
The model's successful run is transported; no equality of the ignored pointer
register is asserted. The caller must discharge the actual body simulation. -/
theorem range (R : RoundingModel) (relation : BlockState → BlockState → Prop)
    (index : RegName) (source model : List Stmt)
    (index_preserves : ∀ s t, relation s t → ∀ n,
      relation (s.setReg index .nat [] (Tile.scalar n)) (t.setReg index .nat [] (Tile.scalar n)))
    (body : ∀ s t, relation s t → ∀ t', stepStmtsR R model t = some t' →
      ∃ s', stepStmtsR R source s = some s' ∧ relation s' t')
    (cur stop stride : Nat) (s t t' : BlockState) (h : relation s t)
    (run : stepForRangeAuxR R index cur stop stride model t = some t') :
    ∃ s', stepForRangeAuxR R index cur stop stride source s = some s' ∧ relation s' t' := by
  by_cases hz : stride = 0
  · simp only [stepForRangeAuxR,hz,↓reduceIte] at run ⊢
    cases run
    exact ⟨s,rfl,h⟩
  · by_cases hc : cur < stop
    · rw [stepForRangeAuxR] at run ⊢
      simp only [hz,hc,↓reduceIte] at run ⊢
      cases hm : stepStmtsR R model (t.setReg index .nat [] (Tile.scalar cur)) with
      | none => simp only [hm] at run; contradiction
      | some u =>
        obtain ⟨v,hv,hrel⟩ := body _ _ (index_preserves s t h cur) u hm
        simp only [hm] at run
        simp only [hv]
        exact range R relation index source model index_preserves body
          (cur+stride) stop stride v u t' hrel run
    · simp only [stepForRangeAuxR,hz,hc,↓reduceIte] at run ⊢
      cases run
      exact ⟨s,rfl,h⟩
termination_by stop - cur
decreasing_by omega

/-- Equal active addresses and the IO contract's zero undef pin suffice for
masked loads. Inactive pointer values themselves need not agree. -/
theorem masked_load (R : RoundingModel) {shape : TileShape}
    (sourcePointer modelPointer : Op .ptr shape) (sourceMask modelMask : Op .bool shape)
    (s t : BlockState) (sp tp : Tile .ptr shape) (mask : Tile .bool shape)
    (hsp : evalOpR R sourcePointer s = some sp)
    (htp : evalOpR R modelPointer t = some tp)
    (hsm : evalOpR R sourceMask s = some mask)
    (htm : evalOpR R modelMask t = some mask)
    (memory : s.mem = t.mem)
    (source_zero : ∀ r o, s.undef r o = 0)
    (model_zero : ∀ r o, t.undef r o = 0)
    (active : ∀ i, mask.data i = Bool.true → sp.data i = tp.data i) :
    evalOpR R (.load .real (.ptr sourcePointer) (.mask sourceMask)) s =
      evalOpR R (.load .real (.ptr modelPointer) (.mask modelMask)) t := by
  unfold evalOpR
  simp only [hsp,htp,hsm,htm,Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  ext i
  by_cases hm : mask.data i = Bool.true
  · simp only [hm,↓reduceIte,active i hm,BlockState.readMemValue_real]
    simp only [BlockState.readMem,memory]
  · simp only [hm,↓reduceIte,source_zero,model_zero,Bool.false_eq_true]

theorem active_signed_offset (base sh sw : Nat) (h w : Int)
    (hh : 0 ≤ h) (hw : 0 ≤ w) :
    base + ((sh : Int)*h).toNat - (-((sh : Int)*h)).toNat +
        ((sw : Int)*w).toNat - (-((sw : Int)*w)).toNat =
      base + ((sh : Int)*h).toNat + ((sw : Int)*w).toNat := by
  have hneg : (-((sh : Int)*h)).toNat = 0 :=
    Int.toNat_eq_zero.mpr (neg_nonpos.mpr (mul_nonneg (Int.natCast_nonneg _) hh))
  have wneg : (-((sw : Int)*w)).toNat = 0 :=
    Int.toNat_eq_zero.mpr (neg_nonpos.mpr (mul_nonneg (Int.natCast_nonneg _) hw))
  simp only [hneg,wneg,Nat.sub_zero]

#axiomsClean masked_load
#axiomsClean active_signed_offset
#axiomsClean AgreeExcept.setReg
#axiomsClean AgreeExcept.writeIgnored
#axiomsClean range
end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.MaskedPointerSimulation
