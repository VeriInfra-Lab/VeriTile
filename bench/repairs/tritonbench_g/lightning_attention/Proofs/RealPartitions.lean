import bench.repairs.tritonbench_g.lightning_attention.Proofs.RealInputs
namespace VeriTile.Bench.Repairs.LightningAttention.Real
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealMath (block_sum)
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

theorem subblocks (N NC CB width start : Nat) (hw : NC*CB=width) (f : Nat → ℝ) :
    (∑ s : Fin NC, ∑ l : Fin CB, if start+s.val*CB+l.val<N then f (start+s.val*CB+l.val) else 0)=
      ∑ t : Fin width, if start+t.val<N then f (start+t.val) else 0 := by
  subst width
  simpa only [Nat.add_assoc] using block_sum (fun t => if start+t<N then f (start+t) else 0) NC CB

theorem reverse_row (NT NC CB n s : Nat) (hn : n<NT) (hs : s<NC) :
    NT*(NC*CB)-(n*NC+s+1)*CB=(NT-1-n)*(NC*CB)+(NC-1-s)*CB := by
  have he : NT=(NT-1-n)+n+1 := by omega
  have hm := congrArg (fun x => x*NC) he
  simp only [Nat.add_mul,Nat.one_mul] at hm
  calc
    _ = (NT*NC-(n*NC+s+1))*CB := by rw [Nat.sub_mul,Nat.mul_assoc]
    _ = ((NT-1-n)*NC+(NC-1-s))*CB := by congr 1; omega
    _ = _ := by ring

theorem query_subdot (p : Parameters) (head row : Nat) (m : Memory ℝ)
    (i : TileIndex [p.E,p.D]) :
    dot dotPrimitive (subVT algebra p head row m) (subK algebra p head row m) i=
      ∑ t : Fin p.CB, if row+t.val<p.N then value p head m (row+t.val) i.1.val*key p head m (row+t.val) i.2.1.val else 0 := by
  rw [dot_eq]
  apply Finset.sum_congr rfl
  intro t _
  by_cases ht : row+t.val<p.N <;>
    simp [subVT,subK,ht,value,key,Nat.add_assoc]

theorem key_value_subdot (p : Parameters) (head row : Nat) (m : Memory ℝ)
    (i : TileIndex [p.D,p.E]) :
    dot dotPrimitive (subQT algebra p head row m) (subG algebra p head row m) i=
      ∑ t : Fin p.CB, if row+t.val<p.N then query p head m (row+t.val) i.1.val*incoming p head m (row+t.val) i.2.1.val else 0 := by
  rw [dot_eq]
  apply Finset.sum_congr rfl
  intro t _
  by_cases ht : row+t.val<p.N <;>
    simp [subQT,subG,ht,query,incoming,Nat.add_assoc]

theorem query_delta (p : Parameters) (htile : p.NC*p.CB=p.BT)
    (head n : Nat) (m : Memory ℝ) (i : TileIndex [p.E,p.D]) :
    queryBlockDelta algebra dotPrimitive p head n m i=
      blockSum p (fun t => value p head m t i.1.val*key p head m t i.2.1.val) n := by
  unfold queryBlockDelta
  change foldRange (fun sub acc j => acc j+dot dotPrimitive (subVT algebra p head ((n*p.NC+sub)*p.CB) m)
    (subK algebra p head ((n*p.NC+sub)*p.CB) m) j) 0 p.NC 1 (fun _ => 0) i=_
  rw [fold_sum,zero_add]
  simp_rw [query_subdot]
  rw [← Fin.sum_univ_eq_sum_range]
  simp only [Nat.add_mul,Nat.mul_assoc,htile]
  have h := subblocks p.N p.NC p.CB p.BT (n*p.BT) htile (fun t => value p head m t i.1.val*key p head m t i.2.1.val)
  simpa only [htile,blockSum] using h

theorem key_value_delta (p : Parameters) (htile : p.NC*p.CB=p.BT)
    (head n : Nat) (hn : n<p.NT) (m : Memory ℝ) (i : TileIndex [p.D,p.E]) :
    keyValueBlockDelta algebra dotPrimitive p head n m i=
      blockSum p (fun t => query p head m t i.1.val*incoming p head m t i.2.1.val) (p.NT-1-n) := by
  unfold keyValueBlockDelta
  change foldRange (fun sub acc j => acc j+dot dotPrimitive (subQT algebra p head (p.NT*p.BT-(n*p.NC+sub+1)*p.CB) m)
    (subG algebra p head (p.NT*p.BT-(n*p.NC+sub+1)*p.CB) m) j) 0 p.NC 1 (fun _ => 0) i=_
  rw [fold_sum,zero_add]
  simp_rw [key_value_subdot]
  have he : ∀ s∈Finset.range p.NC,p.NT*p.BT-(n*p.NC+s+1)*p.CB=
      (p.NT-1-n)*p.BT+(p.NC-1-s)*p.CB := by
    intro s hs
    rw [← htile]
    exact reverse_row p.NT p.NC p.CB n s hn (Finset.mem_range.mp hs)
  rw [Finset.sum_congr rfl (fun s hs => by rw [he s hs])]
  rw [Finset.sum_range_reflect (fun s => ∑ t : Fin p.CB, if (p.NT-1-n)*p.BT+s*p.CB+t.val<p.N then query p head m ((p.NT-1-n)*p.BT+s*p.CB+t.val) i.1.val*incoming p head m ((p.NT-1-n)*p.BT+s*p.CB+t.val) i.2.1.val else 0) p.NC]
  rw [← Fin.sum_univ_eq_sum_range]
  have h := subblocks p.N p.NC p.CB p.BT ((p.NT-1-n)*p.BT) htile (fun t => query p head m t i.1.val*incoming p head m t i.2.1.val)
  simpa only [htile,blockSum] using h

theorem query_carry_blocks (p : Parameters) (htile : p.NC*p.CB=p.BT)
    (head n : Nat) (m : Memory ℝ) (i : TileIndex [p.E,p.D]) :
    queryCarry algebra dotPrimitive p head m n i=
      ∑ c∈Finset.range n,blockSum p (fun t => value p head m t i.1.val*key p head m t i.2.1.val) c := by
  induction n with
  | zero => simp [queryCarry]
  | succ n ih => simp only [queryCarry,add_eq];rw [ih,query_delta p htile,Finset.sum_range_succ]

theorem query_carry (p : Parameters) (hp : 0<p.BT) (htile : p.NC*p.CB=p.BT)
    (head n : Nat) (m : Memory ℝ) (i : TileIndex [p.E,p.D]) :
    queryCarry algebra dotPrimitive p head m n i=
      ∑ t : Fin p.N, if t.val/p.BT<n then value p head m t i.1.val*key p head m t i.2.1.val else 0 := by
  rw [query_carry_blocks p htile,← Fin.sum_univ_eq_sum_range]
  unfold blockSum
  rw [ChunkLinearAttn.Real.blocks_prefix p.N n p.BT (fun t => value p head m t i.1.val*key p head m t i.2.1.val)]
  apply Finset.sum_congr rfl
  intro t _
  simp only [Nat.div_lt_iff_lt_mul hp]

theorem key_value_carry_blocks (p : Parameters) (htile : p.NC*p.CB=p.BT)
    (head n : Nat) (hn : n≤p.NT) (m : Memory ℝ) (i : TileIndex [p.D,p.E]) :
    keyValueCarry algebra dotPrimitive p head m n i=
      ∑ c∈Finset.range n,blockSum p (fun t => query p head m t i.1.val*incoming p head m t i.2.1.val) (p.NT-1-c) := by
  induction n with
  | zero => simp [keyValueCarry]
  | succ n ih =>
    simp only [keyValueCarry,add_eq]
    rw [ih (by omega),key_value_delta p htile head n (by omega),Finset.sum_range_succ]

theorem key_value_carry (p : Parameters) (hp : 0<p.BT) (htile : p.NC*p.CB=p.BT)
    (hcover : p.N≤p.NT*p.BT) (head n : Nat) (hn : n≤p.NT)
    (m : Memory ℝ) (i : TileIndex [p.D,p.E]) :
    keyValueCarry algebra dotPrimitive p head m n i=
      ∑ t : Fin p.N, if p.NT-n≤t.val/p.BT then query p head m t i.1.val*incoming p head m t i.2.1.val else 0 := by
  rw [key_value_carry_blocks p htile head n hn]
  rw [ChunkLinearAttn.Real.reverse_prefix p.NT n hn]
  rw [← Fin.sum_univ_eq_sum_range]
  unfold blockSum
  exact ChunkLinearAttn.Real.blocks_select p.N p.NT p.BT hp hcover
    (fun c => p.NT-n≤c) (fun t => query p head m t i.1.val*incoming p head m t i.2.1.val)

#axiomsClean query_delta
#axiomsClean key_value_delta
#axiomsClean query_carry
#axiomsClean key_value_carry
end
end VeriTile.Bench.Repairs.LightningAttention.Real
