import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.GridComposition

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

/-- Every logical O or DV coordinate has a program and an active lane,
including the final time and value tiles when either dimension is ragged. -/
theorem output_coverage (shape : Shape) (L : Nat) (positive : 0<L) (valueWidth : 0<shape.BV)
    (bh t v : Nat) (hb : bh<shape.B*shape.H) (ht : t<shape.T) (hv : v<shape.V) :
    ∃ (p : Program shape L) (r : Fin L) (f : Fin shape.BV),
      (outputRectangle shape p.tile L).active ⟨r,f,()⟩ ∧
      (outputRectangle shape p.tile L).address ⟨r,f,()⟩=dense shape.T shape.V bh t v := by
  obtain ⟨tc,tr,te⟩ := tile_owner shape.T L t positive ht
  obtain ⟨vc,vr,ve⟩ := tile_owner shape.V shape.BV v valueWidth hv
  let p : Program shape L := ⟨⟨bh,hb⟩,⟨t/L,tc⟩,⟨v/shape.BV,vc⟩⟩
  refine ⟨p,⟨t%L,tr⟩,⟨v%shape.BV,vr⟩,?_,?_⟩
  · simpa only [outputRectangle,Program.tile,Rectangle.active,p,te,ve] using And.intro ht hv
  · simp only [outputRectangle,Program.tile,Rectangle.address,p,Nat.mul_zero,Nat.add_zero,te,ve,dense]
    ring

/-- DQ/DK retain each value-shard scratch slice; no shard reduction is
part of this address coverage statement. -/
theorem query_coverage (shape : Shape) (keyFits : shape.K≤ shape.BK)
    (sh bh t e : Nat) (hs : sh<blocks shape.V shape.BV) (hb : bh<shape.B*shape.H)
    (ht : t<shape.T) (he : e<shape.K) :
    ∃ (p : Program shape 64) (r : Fin 64) (f : Fin shape.BK),
      p.value.val=sh ∧ (queryRectangle shape p.tile).active ⟨r,f,()⟩ ∧
      (queryRectangle shape p.tile).address ⟨r,f,()⟩=
        dense shape.T shape.K (bh+shape.B*shape.H*sh) t e := by
  obtain ⟨tc,tr,te⟩ := tile_owner shape.T 64 t (by decide) ht
  let p : Program shape 64 := ⟨⟨bh,hb⟩,⟨t/64,tc⟩,⟨sh,hs⟩⟩
  refine ⟨p,⟨t%64,tr⟩,⟨e,lt_of_lt_of_le he keyFits⟩,rfl,?_,?_⟩
  · simpa only [queryRectangle,Program.tile,Rectangle.active,p,Nat.zero_mul,Nat.zero_add,te]
      using And.intro ht he
  · simp only [queryRectangle,Program.tile,Rectangle.address,p,Nat.zero_mul,Nat.zero_add,te,dense]
    ring

theorem normalizer_coverage (shape : Shape) (valueWidth : 0<shape.BV) (valuePositive : 0<shape.V)
    (bh t : Nat) (hb : bh<shape.B*shape.H) (ht : t<shape.T) :
    ∃ (p : Program shape 128) (r : Fin 128),
      normalizerWrites shape p.tile "z" (bh*shape.T+t) := by
  obtain ⟨tc,tr,te⟩ := tile_owner shape.T 128 t (by decide) ht
  have vc := (tile_owner shape.V shape.BV 0 valueWidth valuePositive).1
  let p : Program shape 128 := ⟨⟨bh,hb⟩,⟨t/128,tc⟩,⟨0,by simpa using vc⟩⟩
  refine ⟨p,⟨t%128,tr⟩,rfl,rfl,⟨t%128,tr⟩,?_,?_⟩
  · simpa only [p,Program.tile,te] using ht
  · simp only [normalizerAddress,Program.tile,p,Nat.mul_zero,Nat.add_zero]
    omega

theorem programs_empty (shape : Shape) (L : Nat) (positive : 0<L)
    (empty : shape.B=0 ∨ shape.H=0 ∨ shape.T=0) : programs shape L=[] := by
  rcases empty with hb|hh|ht
  · simp [programs,hb]
  · simp [programs,hh]
  · have count : blocks shape.T L=0 := by
      simp only [blocks,ht,Nat.zero_add]
      exact Nat.div_eq_of_lt (by omega)
    simp [programs,count]

/-- Empty accepted B/H/T domains launch no program and preserve the entire
state, independently of scalar primitives or native compiler descriptors. -/
theorem empty_launch (shape : Shape) (L : Nat) (positive : 0<L)
    (empty : shape.B=0 ∨ shape.H=0 ∨ shape.T=0)
    (launch : Program shape L → State α → Option (State α)) (s : State α) :
    runPrograms launch (programs shape L) s=some s := by
  rw [programs_empty shape L positive empty]
  rfl

#axiomsClean output_coverage
#axiomsClean query_coverage
#axiomsClean normalizer_coverage
#axiomsClean programs_empty
#axiomsClean empty_launch
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
