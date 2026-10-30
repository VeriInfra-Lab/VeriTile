import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.RealLocality

namespace VeriTile.Bench.Repairs.ContextAttnFwd.RealInterface
open VeriTile.Triton
noncomputable section

@[simp] theorem decode_encode (p : Layout) (g : Grid) (e : Event p g) :
    decode p g (encode p g e)=e := Equiv.symm_apply_apply _ _

end
end VeriTile.Bench.Repairs.ContextAttnFwd.RealInterface

namespace VeriTile.Bench.Repairs.ContextAttnFwd.RealBindings
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
open RealInterface RealLocality
noncomputable section

theorem agreement (p : Layout) (r : Regions) (extent : RegionBounds) (g : Grid) (scale : ℝ)
    (v : Values r extent) (s : BlockState) (bound : (io p r extent g scale).InputBinding s v) :
    Agree r extent (arrays r extent v) (Program.inputs s r) := by
  constructor <;> intro off h
  · simpa only [arrays,dif_pos h] using (bound (0 : Fin 6) ⟨off,h⟩ trivial).symm
  · simpa only [arrays,dif_pos h] using (bound (1 : Fin 6) ⟨off,h⟩ trivial).symm
  · simpa only [arrays,dif_pos h] using (bound (2 : Fin 6) ⟨off,h⟩ trivial).symm
  · simpa only [arrays,dif_pos h] using (bound (3 : Fin 6) ⟨off,h⟩ trivial).symm
  · simpa only [arrays,dif_pos h] using (bound (4 : Fin 6) ⟨off,h⟩ trivial).symm
  · simpa only [arrays,dif_pos h] using (bound (5 : Fin 6) ⟨off,h⟩ trivial).symm

theorem metadata (p : Layout) (r : Regions) (extent : RegionBounds) (g : Grid) (scale : ℝ)
    (v : Values r extent) (s : BlockState) (domain : Domain p r extent g v)
    (bound : (io p r extent g scale).InputBinding s v) (idx : GridIndex g) :
    Metadata p (launch idx) (arrays r extent v) (Program.inputs s r) :=
  RealLocality.metadata p r extent (launch idx) _ _ (agreement p r extent g scale v s bound) (domain.1 idx)

theorem bounds (p : Layout) (r : Regions) (extent : RegionBounds) (g : Grid) (scale : ℝ)
    (v : Values r extent) (s : BlockState) (domain : Domain p r extent g v)
    (bound : (io p r extent g scale).InputBinding s v) (idx : GridIndex g) :
    AddressBounds.Valid p r (launch idx) (Program.inputs s r) extent :=
  (metadata p r extent g scale v s domain bound idx).bounds p r _ _ _ extent (domain.1 idx)

theorem capacity (p : Layout) (r : Regions) (extent : RegionBounds) (g : Grid) (scale : ℝ)
    (v : Values r extent) (s : BlockState) (domain : Domain p r extent g v)
    (bound : (io p r extent g scale).InputBinding s v) (caps : RegionBounds)
    (windows : (io p r extent g scale).InBounds caps s v) (idx : GridIndex g) :
    AddressBounds.Valid p r (launch idx) (Program.inputs s r) caps := by
  have hb := bounds p r extent g scale v s domain bound idx
  have hm := metadata p r extent g scale v s domain bound idx
  have input (role : Fin 6) (off : Nat) (h : off < extent (inputRegion r role)) :
      off < caps (inputRegion r role) := windows.1 role ⟨off,h⟩ trivial
  have inside {a : Int} {n : Nat} (nonnegative : 0 ≤ a) (cap : a.toNat < n) : AddressBounds.inside a n := by
    constructor <;> omega
  constructor
  · exact input 3 _ hb.start
  · exact input 5 _ hb.prompt
  · exact input 4 _ hb.sequence
  · intro i d hi
    exact inside (hb.query i d hi).1 (input 0 _ (AddressBounds.inside_toNat (hb.query i d hi)))
  · intro c hc j d hi
    exact inside (hb.key c hc j d hi).1 (input 1 _ (AddressBounds.inside_toNat (hb.key c hc j d hi)))
  · intro c hc j d hi
    exact inside (hb.value c hc j d hi).1 (input 2 _ (AddressBounds.inside_toNat (hb.value c hc j d hi)))
  · intro i d hi
    apply inside (hb.output i d hi).1
    have ha : active p g (arrays r extent v) (idx,i,d,()) := by
      change RealIO.active p (launch idx) (arrays r extent v) _
      rw [hm.active p _ _ _]
      exact hi
    have hout := windows.2.1 (0 : Fin 1) (encode p g (idx,i,d,())) (by simpa only [io,decode_encode] using ha)
    change address p g (arrays r extent v) (decode p g (encode p g (idx,i,d,()))) < caps r.out at hout
    rw [decode_encode] at hout
    change RealIO.write p (launch idx) (arrays r extent v) _ < _ at hout
    rw [hm.address p _ _ _] at hout
    exact hout

#axiomsClean capacity
end
end VeriTile.Bench.Repairs.ContextAttnFwd.RealBindings
