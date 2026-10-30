import bench.tritonbench_g.fused_rotary_embedding.Proofs.Geometry

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding

theorem radix_bound (a b n m : Nat) (ha : a<n) (hb : b<m) : a*m+b<n*m := by
  calc
    _ < a*m+m := Nat.add_lt_add_left hb _
    _ = (a+1)*m := by ring
    _ ≤ n*m := Nat.mul_le_mul_right _ (by omega)

theorem input_bound (s : Shape) (hs : s.Valid) (w : Work) (hw : s.Active w) (o : Bool) :
    s.input w o<s.tokens*s.heads w.kind*s.width :=
  radix_bound _ _ _ _ (radix_bound _ _ _ _ hw.1 hw.2.1) (dim_bound s hs w hw o)

theorem coefficient_bound (s : Shape) (hs : s.Valid) (w : Work) (hw : s.Active w) (sine : Bool) :
    s.coeff sine w.token w.pair<s.tokens*(if sine then s.sinWidth else s.cosWidth) := by
  apply radix_bound _ _ _ _ hw.1
  have p := hw.2.2
  cases sine <;> simp only [Bool.false_eq_true,↓reduceIte]
  · exact lt_of_lt_of_le p hs.cos
  · exact lt_of_lt_of_le p hs.sin

theorem cache_bound (s : Shape) (hs : s.Valid) (key : Bool) (b h slot d : Nat)
    (hb : b<s.blocks) (hh : h<s.headsK) (hl : slot<s.blockSize) (hd : d<s.width) :
    s.cache key b h slot d<s.blocks*s.headsK*s.blockSize*s.width := by
  have bh := radix_bound _ _ _ _ hb hh
  unfold Shape.cache
  split
  · have dw : s.width/s.split*s.split=s.width := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hs.divides)
    have dq : d/s.split<s.width/s.split := (Nat.div_lt_iff_lt_mul hs.split).mpr (by simpa [dw] using hd)
    have h1 := radix_bound _ _ _ _ bh dq
    have h2 := radix_bound _ _ _ _ h1 hl
    have h3 := radix_bound _ _ _ _ h2 (Nat.mod_lt d hs.split)
    have size : s.blocks*s.headsK*(s.width/s.split)*s.blockSize*s.split=
        s.blocks*s.headsK*s.blockSize*s.width := by
      calc
        _ = s.blocks*s.headsK*s.blockSize*(s.width/s.split*s.split) := by ring
        _ = _ := by rw [dw]
    simpa only [size] using h3
  · exact radix_bound _ _ _ _ (radix_bound _ _ _ _ bh hl) hd

theorem output_bound (c : Config) (hc : c.Valid) (w : Work) (hw : c.shape.Active w) (o : Bool) :
    c.output w o < if w.kind=.q then c.shape.tokens*c.shape.headsQ*c.shape.width
      else c.shape.blocks*c.shape.headsK*c.shape.blockSize*c.shape.width := by
  by_cases hq : w.kind=.q
  · simp only [Config.output,hq,ite_true]
    simpa [Shape.heads,hq] using input_bound c.shape hc.shape w hw o
  · simp only [Config.output,hq,ite_false]
    exact cache_bound c.shape hc.shape _ _ _ _ _ (hc.blockBound _ hw.1)
      (by simpa [Shape.heads,hq] using hw.2.1) (Nat.mod_lt _ hc.shape.blockSize)
      (dim_bound c.shape hc.shape w hw o)

theorem table_bound (c : Config) (hc : c.Valid) (t : Nat) (ht : t<c.shape.tokens) :
    t*c.shape.tableWidth+c.past t/c.shape.blockSize<c.shape.tokens*c.shape.tableWidth :=
  radix_bound _ _ _ _ ht (hc.lengthTable t ht)

/-- Literal canonical source stride decomposition, including the 4D x=D
route and 5D split-x route. -/
def sourceCacheAddress (s : Shape) (b h slot d : Nat) : Nat :=
  if s.newLayout then
    b*(s.headsK*(s.width/s.split)*s.blockSize*s.split) +
    h*((s.width/s.split)*s.blockSize*s.split) + slot*s.split +
    (d/s.split)*(s.blockSize*s.split)+d%s.split
  else b*(s.headsK*s.blockSize*s.width)+h*(s.blockSize*s.width)+slot*s.width+
    (d/s.split)*0+d%s.split

theorem cache_source_address (s : Shape) (b h slot d : Nat) (hd : d<s.width)
    (ordinary : s.newLayout=false → s.split=s.width) :
    sourceCacheAddress s b h slot d=s.cache true b h slot d := by
  cases layout : s.newLayout
  · have split := ordinary layout
    simp [sourceCacheAddress,Shape.cache,layout,split,Nat.mod_eq_of_lt hd]
    ring
  · simp [sourceCacheAddress,Shape.cache,layout]
    ring

theorem active_byte_int64 (offset extent : Nat) (ho : offset<extent)
    (he : extent≤(2^63-1)/4) (i : Fin 4) : 4*offset+i.val<2^63 := by omega

#axiomsClean cache_source_address
#axiomsClean output_bound
#axiomsClean table_bound
#axiomsClean active_byte_int64
end VeriTile.Bench.Repairs.FusedRotaryEmbedding
