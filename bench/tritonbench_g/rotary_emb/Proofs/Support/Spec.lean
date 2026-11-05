import bench.tritonbench_g.rotary_emb.Proofs.Vector

namespace VeriTile.Bench.Repairs.RotaryEmb

/-- Interleaved rotary formula, independently indexed by an arbitrary tensor
column. The cosine and sine row strides are independent. -/
@[standalone_spec] noncomputable def formula (s : Shape) (m : Memory ℝ)
    (key : Bool) (token head dim : Nat) : ℝ :=
  let x0 := m (dataRegion key) ((token*s.heads key+head)*s.width+(dim/2)*2)
  let x1 := m (dataRegion key) ((token*s.heads key+head)*s.width+(dim/2)*2+1)
  let c := m .cos (token*s.cosWidth+dim)
  let z := m .sin (token*s.sinWidth+dim)
  if dim%2=0 then x0*c-x1*z else x0*z+x1*c

theorem pair_formula (s : Shape) (m : Memory ℝ) (w : Work) (odd : Bool) :
    component (pairValues s realOps w m) odd=
      formula s m w.key w.token w.head (2*w.pair+if odd then 1 else 0) := by
  cases odd <;>
    simp [component,pairValues,sourcePair,realOps,formula,Shape.cell,Shape.address,
      Shape.coefficient,Nat.add_div,Nat.add_mod,Nat.mul_comm,Nat.mul_add,add_assoc]

theorem column_decomposition (d : Nat) : 2*(d/2)+(if decide (d%2=1) then 1 else 0)=d := by
  by_cases h : d%2=1 <;> simp [h] <;> omega

theorem column_active (s : Shape) (valid : s.Valid) (key : Bool) (n h d : Nat)
    (hn : n<s.tokens) (hh : h<s.heads key) (hd : d<s.rotary) :
    s.Active ⟨key,n,h,d/2⟩ := by
  refine ⟨hn,hh,?_⟩
  have e := even_width s valid
  change d/2<s.pairs
  omega

theorem source_formula (s : Shape) (valid : s.Valid) (m : Memory ℝ)
    (key : Bool) (n h d : Nat) (hn : n<s.tokens) (hh : h<s.heads key) (hd : d<s.rotary) :
    source s realOps m (dataRegion key) (s.address key n h d)=formula s m key n h d := by
  rw [source_eq_run s valid]
  have result := run_readback s valid realOps m ⟨key,n,h,d/2⟩
    (column_active s valid key n h d hn hh hd) (decide (d%2=1))
  rw [pair_formula] at result
  change run s realOps m (dataRegion key)
    (s.address key n h (2*(d/2)+if decide (d%2=1) then 1 else 0))=
    formula s m key n h (2*(d/2)+if decide (d%2=1) then 1 else 0) at result
  rw [column_decomposition] at result
  exact result

#specNonCircular formula avoiding [sourcePair,pairValues,step,execute,source,run]
#axiomsClean source_formula
end VeriTile.Bench.Repairs.RotaryEmb
