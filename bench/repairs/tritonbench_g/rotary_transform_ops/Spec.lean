import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.Host
namespace VeriTile.Bench.Repairs.RotaryTransformOps
/-- Independent rotation at an arbitrary logical column. Input coordinates
and cached coefficients are initial values. Outside the rotary prefix the
caller input is returned unchanged. -/
noncomputable def realFormula (c : Config) (m : Memory ℝ) (batch row head col : Nat) : ℝ :=
  if col<2*c.shape.half then
    let pair := c.shape.pairOf col
    let second := c.shape.second col
    let token := c.start batch+row
    let a := m .input (c.shape.address token head (c.shape.dim pair false))
    let b := m .input (c.shape.address token head (c.shape.dim pair true))
    let cache := row+c.shift batch<c.shape.coefficientRows
    let co := if cache then m .cosine ((row+c.shift batch)*c.shape.half+pair) else 1
    let sn := if cache then m .sine ((row+c.shift batch)*c.shape.half+pair) else 0
    let signed := if c.shape.conjugate then -sn else sn
    if second then a*signed+b*co else a*co-b*signed
  else m .input (c.shape.address (c.start batch+row) head col)

def nativeFormula (c : Config) (p : Native.Instructions) (m : Memory Native.Word)
    (batch row head col : Nat) : Native.Word :=
  if col<2*c.shape.half then
    let pair := c.shape.pairOf col
    let second := c.shape.second col
    let token := c.start batch+row
    let a := m .input (c.shape.address token head (c.shape.dim pair false))
    let b := m .input (c.shape.address token head (c.shape.dim pair true))
    let cache := row+c.shift batch<c.shape.coefficientRows
    let co := if cache then m .cosine ((row+c.shift batch)*c.shape.half+pair) else Native.one
    let sn := if cache then m .sine ((row+c.shift batch)*c.shape.half+pair) else Native.zero
    let signed := if c.shape.conjugate then p.sub0 sn else sn
    if second then
      if c.shape.interleaved then p.fma co b (p.mul signed a) else p.fma signed a (p.mul co b)
    else p.fma co a (p.neg (p.mul signed b))
  else m .input (c.shape.address (c.start batch+row) head col)

theorem real_pair_formula (c : Config) (m : Memory ℝ) (w : Work) (hw : c.Active w) (o : Bool) :
    component (pairValues c realOps w m) o=realFormula c m w.batch w.row w.head (c.shape.dim w.pair o) := by
  have bound := dim_bound c.shape w.pair hw.2.2.2 o
  obtain ⟨pair,second⟩ := pair_inverse c.shape w.pair hw.2.2.2 o
  simp only [realFormula,bound,↓reduceIte,pair,second,pairValues,Config.cell,Config.coeff,Config.hasCoeff]
  cases hi : c.shape.interleaved <;> cases hc : c.shape.conjugate <;> cases o <;>
    simp [sourcePair,realOps,component,hi,hc] <;> ring

theorem native_pair_formula (c : Config) (p : Native.Instructions) (m : Memory Native.Word)
    (w : Work) (hw : c.Active w) (o : Bool) :
    Native.pairWord c p w m o=nativeFormula c p m w.batch w.row w.head (c.shape.dim w.pair o) := by
  have bound := dim_bound c.shape w.pair hw.2.2.2 o
  obtain ⟨pair,second⟩ := pair_inverse c.shape w.pair hw.2.2.2 o
  simp only [nativeFormula,bound,↓reduceIte,pair,second,Native.pairWord,Native.graph,
    Config.cell,Config.coeff,Config.hasCoeff]

#specNonCircular realFormula avoiding [sourcePair,pairValues,source,run,Host.mathematical]
#specNonCircular nativeFormula avoiding [Native.graph,Native.pairWord,Native.selected,Physical.native,Host.selected]
#axiomsClean real_pair_formula
#axiomsClean native_pair_formula
end VeriTile.Bench.Repairs.RotaryTransformOps
