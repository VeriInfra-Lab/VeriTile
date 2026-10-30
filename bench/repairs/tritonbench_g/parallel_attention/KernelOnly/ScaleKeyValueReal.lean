import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.ScaleQueryReal

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Math
open VeriTile.Bench.Repairs.ParallelAttention
variable {σ : ℝ}
noncomputable section
open scoped BigOperators

def kvStart (s : Shape) (tile : Tile) (diagonal : Bool) (j : Nat) : Nat :=
  if diagonal then tile.chunk*64+j*32 else (blocks s.T 32-1-j)*32

def kvValid (s : Shape) (tile : Tile) (r j u : Nat) (diagonal : Bool) : Bool :=
  decide (kvStart s tile diagonal j+u < s.T) && (!diagonal || decide (r ≤ j*32+u))

def kvRaw (a : Ops α) (p : Plans) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r j u : Nat) (diagonal : Bool) : α :=
  a.mul (dot a p [2,0,tile.bh,tile.chunk,tile.value,r,j,u] s.BK
    (fun f => let k := load a s.T s.K x.k tile.bh (tile.chunk*64+r) (tile.key*s.BK+f)
              if diagonal then k else a.cast k)
    (fun f => load a s.T s.K x.q tile.bh (kvStart s tile diagonal j+u) (tile.key*s.BK+f)))
    (a.scale s.K s.useScale)

def kvScore (a : Ops α) (p : Plans) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r j u : Nat) (diagonal : Bool) : α :=
  if kvValid s tile r j u diagonal then kvRaw a p s x tile r j u diagonal else a.zero

def kvSquare (a : Ops α) (p : Plans) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r j u : Nat) (diagonal : Bool) : α :=
  if diagonal then
    if kvValid s tile r j u diagonal then
      a.mul (kvRaw a p s x tile r j u diagonal) (kvRaw a p s x tile r j u diagonal)
    else a.zero
  else a.mul (kvScore a p s x tile r j u diagonal) (kvScore a p s x tile r j u diagonal)

def kvDS (a : Ops α) (p : Plans) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r j u : Nat) (diagonal : Bool) : α :=
  let raw := dot a p [2,1,tile.bh,tile.chunk,tile.value,r,j,u] s.BV
    (fun f => load a s.T s.V x.v tile.bh (tile.chunk*64+r) (tile.value*s.BV+f))
    (fun f => a.cast (load a s.T s.V x.do_ tile.bh (kvStart s tile diagonal j+u) (tile.value*s.BV+f)))
  let z := dzLoad a s x tile.bh (kvStart s tile diagonal j+u)
  if diagonal then
    a.mul (if kvValid s tile r j u diagonal then (if tile.value=0 then a.add raw z else raw) else a.zero)
      (a.scale s.K s.useScale)
  else
    if kvValid s tile r j u diagonal then
      if tile.value=0 then a.add (a.mul raw (a.scale s.K s.useScale)) (a.mul z (a.scale s.K s.useScale))
      else a.mul raw (a.scale s.K s.useScale)
    else a.zero

def keyProduct (s : Shape) (x : TensorInputs ℝ) (tile : Tile) (r e t : Nat) : ℝ :=
  if t < s.T ∧ tile.chunk*64+r ≤ t then
    2*σ*partialCotangent s x tile.bh tile.value t (tile.chunk*64+r)*
      score s.K (σ) (headInputs x tile.bh) t (tile.chunk*64+r)*x.q tile.bh t e
  else 0

def valueProduct (s : Shape) (x : TensorInputs ℝ) (tile : Tile) (r v t : Nat) : ℝ :=
  if t < s.T ∧ tile.chunk*64+r ≤ t then
    (score s.K (σ) (headInputs x tile.bh) t (tile.chunk*64+r))^2*
      x.do_ tile.bh t (tile.value*s.BV+v)
  else 0

theorem kvRaw_real (p : Plans) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r j u : Nat) (diagonal : Bool) (hk : tile.key=0)
    (fits : s.K ≤ s.BK) (hn : tile.chunk*64+r < s.T)
    (ht : kvStart s tile diagonal j+u < s.T) :
    kvRaw (scaleOps σ) p s x tile r j u diagonal=
      score s.K (σ) (headInputs x tile.bh)
        (kvStart s tile diagonal j+u) (tile.chunk*64+r) := by
  have h := key_query_dot (σ := σ) p [2,0,tile.bh,tile.chunk,tile.value,r,j,u]
    s x tile.bh (kvStart s tile diagonal j+u) (tile.chunk*64+r) ht hn fits
  cases diagonal <;> simpa [kvRaw,hk] using h

theorem reverse_partial_dot (p : Plans) (site : List Nat) (s : Shape) (x : TensorInputs ℝ)
    (bh shard t n : Nat) (ht : t < s.T) (hn : n < s.T) :
    dot (scaleOps σ) p site s.BV
      (fun f => load (scaleOps σ) s.T s.V x.v bh n (shard*s.BV+f))
      (fun f => (scaleOps σ).cast (load (scaleOps σ) s.T s.V x.do_ bh t (shard*s.BV+f)))=
      ∑ f∈Finset.range s.BV,if shard*s.BV+f < s.V then
        x.do_ bh t (shard*s.BV+f)*x.v bh n (shard*s.BV+f) else 0 := by
  rw [dot_real (σ := σ)]
  apply Finset.sum_congr rfl
  intro f hf
  by_cases h : shard*s.BV+f < s.V <;> simp [load,ht,hn,h,mul_comm]

theorem kvDS_real (p : Plans) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r j u : Nat) (diagonal : Bool) (hn : tile.chunk*64+r < s.T) :
    kvDS (scaleOps σ) p s x tile r j u diagonal=
      if kvValid s tile r j u diagonal then
        σ*partialCotangent s x tile.bh tile.value
          (kvStart s tile diagonal j+u) (tile.chunk*64+r)
      else 0 := by
  by_cases h : kvValid s tile r j u diagonal=true
  · have ht : kvStart s tile diagonal j+u < s.T := by
      have h' := h
      simp only [kvValid,Bool.and_eq_true,decide_eq_true_eq] at h'
      exact h'.1
    have d := reverse_partial_dot (σ := σ) p [2,1,tile.bh,tile.chunk,tile.value,r,j,u]
      s x tile.bh tile.value (kvStart s tile diagonal j+u) (tile.chunk*64+r) ht hn
    simp only [real_cast (σ := σ)] at d
    simp only [kvDS,real_cast (σ := σ),d]
    cases diagonal <;> by_cases hv : tile.value=0 <;>
      simp [h,partialCotangent,hv,dzLoad,ht] <;> ring
  · cases diagonal <;> simp [kvDS,h]

theorem kvValid_causal (s : Shape) (tile : Tile) (r j u : Nat) (diagonal : Bool)
    (hr : r < 64) (hu : u < 32)
    (future : diagonal=false → j < blocks s.T 32-(tile.chunk+1)*2) :
    (kvValid s tile r j u diagonal=true) ↔
      kvStart s tile diagonal j+u < s.T ∧ tile.chunk*64+r ≤ kvStart s tile diagonal j+u := by
  cases diagonal with
  | true => simp [kvValid,kvStart,Nat.add_assoc]
  | false =>
    have hj := future rfl
    have before : tile.chunk*64+r ≤ (blocks s.T 32-1-j)*32+u := by omega
    simp [kvValid,kvStart,before]

theorem kv_key_product (p : Plans) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r e j u : Nat) (diagonal : Bool) (hk : tile.key=0)
    (fits : s.K ≤ s.BK) (hn : tile.chunk*64+r < s.T) (he : e < s.K)
    (hr : r < 64) (hu : u < 32)
    (future : diagonal=false → j < blocks s.T 32-(tile.chunk+1)*2) :
    (scaleOps σ).cast ((scaleOps σ).mul ((scaleOps σ).mul (scaleOps σ).two (kvDS (scaleOps σ) p s x tile r j u diagonal))
        (kvScore (scaleOps σ) p s x tile r j u diagonal)) *
      load (scaleOps σ) s.T s.K x.q tile.bh (kvStart s tile diagonal j+u) (tile.key*s.BK+e)=
      keyProduct (σ := σ) s x tile r e (kvStart s tile diagonal j+u) := by
  have hc := kvValid_causal s tile r j u diagonal hr hu future
  by_cases h : kvValid s tile r j u diagonal=true
  · have hb := hc.mp h
    rw [kvDS_real (σ := σ) p s x tile r j u diagonal hn]
    simp only [kvScore,h,if_true,kvRaw_real (σ := σ) p s x tile r j u diagonal hk fits hn hb.1,
      real_cast (σ := σ),real_mul (σ := σ),real_two (σ := σ),keyProduct,if_pos hb,hk,Nat.zero_mul,Nat.zero_add]
    simp only [load,hb.1,he,and_self,if_true]
    ring
  · have hb : ¬(kvStart s tile diagonal j+u < s.T ∧ tile.chunk*64+r ≤ kvStart s tile diagonal j+u) := by
      exact fun hb => h (hc.mpr hb)
    simp [kvScore,h,keyProduct,hb]

theorem kv_value_product (p : Plans) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r v j u : Nat) (diagonal : Bool) (hk : tile.key=0)
    (fits : s.K ≤ s.BK) (hn : tile.chunk*64+r < s.T) (hv : tile.value*s.BV+v < s.V)
    (hr : r < 64) (hu : u < 32)
    (future : diagonal=false → j < blocks s.T 32-(tile.chunk+1)*2) :
    (scaleOps σ).cast (kvSquare (scaleOps σ) p s x tile r j u diagonal) *
      (scaleOps σ).cast (load (scaleOps σ) s.T s.V x.do_ tile.bh (kvStart s tile diagonal j+u) (tile.value*s.BV+v))=
      valueProduct (σ := σ) s x tile r v (kvStart s tile diagonal j+u) := by
  have hc := kvValid_causal s tile r j u diagonal hr hu future
  by_cases h : kvValid s tile r j u diagonal=true
  · have hb := hc.mp h
    have raw := kvRaw_real (σ := σ) p s x tile r j u diagonal hk fits hn hb.1
    cases diagonal <;> simp [kvSquare,kvScore,h,raw,valueProduct,hb,load,hb.1,hv,pow_two]
  · have hb : ¬(kvStart s tile diagonal j+u < s.T ∧ tile.chunk*64+r ≤ kvStart s tile diagonal j+u) := by
      exact fun hb => h (hc.mpr hb)
    cases diagonal <;> simp [kvSquare,kvScore,h,valueProduct,hb]

theorem keyValueStep_real (p : Plans) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r e v j : Nat) (diagonal : Bool) (carry : ℝ×ℝ) (hk : tile.key=0)
    (fits : s.K ≤ s.BK) (hn : tile.chunk*64+r < s.T) (he : e < s.K)
    (hv : tile.value*s.BV+v < s.V) (hr : r < 64)
    (future : diagonal=false → j < blocks s.T 32-(tile.chunk+1)*2) :
    keyValueStep (scaleOps σ) p s x tile r e v diagonal j carry=
      (carry.1+∑ u∈Finset.range 32,keyProduct (σ := σ) s x tile r e (kvStart s tile diagonal j+u),
       carry.2+∑ u∈Finset.range 32,valueProduct (σ := σ) s x tile r v (kvStart s tile diagonal j+u)) := by
  change (carry.1+dot (scaleOps σ) p [2,3,tile.bh,tile.chunk,tile.value,r,j,e] 32
      (fun u => (scaleOps σ).cast ((scaleOps σ).mul ((scaleOps σ).mul (scaleOps σ).two (kvDS (scaleOps σ) p s x tile r j u diagonal))
        (kvScore (scaleOps σ) p s x tile r j u diagonal)))
      (fun u => load (scaleOps σ) s.T s.K x.q tile.bh (kvStart s tile diagonal j+u) (tile.key*s.BK+e)),
    carry.2+dot (scaleOps σ) p [2,2,tile.bh,tile.chunk,tile.value,r,j,v] 32
      (fun u => (scaleOps σ).cast (kvSquare (scaleOps σ) p s x tile r j u diagonal))
      (fun u => (scaleOps σ).cast (load (scaleOps σ) s.T s.V x.do_ tile.bh (kvStart s tile diagonal j+u) (tile.value*s.BV+v))))=_
  rw [dot_real (σ := σ),dot_real (σ := σ)]
  apply Prod.ext <;> dsimp only <;> congr 1 <;> apply Finset.sum_congr rfl <;> intro u hu
  · exact kv_key_product (σ := σ) p s x tile r e j u diagonal hk fits hn he hr (Finset.mem_range.mp hu) future
  · exact kv_value_product (σ := σ) p s x tile r v j u diagonal hk fits hn hv hr (Finset.mem_range.mp hu) future

#axiomsClean kvRaw_real
#axiomsClean kvDS_real
#axiomsClean kvValid_causal
#axiomsClean kv_key_product
#axiomsClean kv_value_product
#axiomsClean keyValueStep_real
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Math
