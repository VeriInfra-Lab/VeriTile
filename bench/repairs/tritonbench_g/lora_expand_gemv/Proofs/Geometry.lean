import bench.repairs.tritonbench_g.lora_expand_gemv.Kernels
namespace VeriTile.Bench.Repairs.LoraExpandGemv
open VeriTile.Triton FP.Structural FPValue
noncomputable section
open Classical

/-- Exact original tensor strides are retained, including singleton dimensions.
The only new operation is a dense logical copy of the one-dimensional indices. -/
structure Geometry where
  (B L N K : Nat)
  (xm xk l0 wn wk ym yn : Nat)
  (indexStart indexStride indexCapacity : Nat)
  (xCapacity wCapacity yCapacity : Nat)
  add : Bool

def Geometry.block (g : Geometry) := TritonBenchG.LoraExpandGemv.Host.nextPowerOfTwo g.K
def Geometry.even (g : Geometry) := decide (g.K % g.block=0)
def Geometry.count (g : Geometry) := g.N*g.B

def Geometry.parameters (g : Geometry) : Branches.Parameters where
  input := "X"
  weight := "W"
  output := "Y"
  indices := ⟨"I"⟩
  inner := g.K
  outputSize := g.N
  splits := g.N
  inputBatchStride := g.xm
  inputInnerStride := g.xk
  loraStride := g.l0
  weightRowStride := g.wn
  weightInnerStride := g.wk
  outputBatchStride := g.ym
  outputStride := g.yn
  blockRows := 256
  blockInner := g.block
  evenK := g.even
  addInputs := g.add
  castType := Bool.true

def xoff (g : Geometry) (b k : Nat) := b*g.xm+k*g.xk
def woff (g : Geometry) (l n k : Nat) := l*g.l0+n*g.wn+k*g.wk
def yoff (g : Geometry) (b n : Nat) := b*g.ym+n

def logicalIndex (g : Geometry) (raw : Nat → Int) (b : Nat) := raw (g.indexStart+b*g.indexStride)

def denseIndices (g : Geometry) (raw : Nat → Int) : Nat → Int := logicalIndex g raw

theorem normalization (g : Geometry) (raw : Nat → Int) (b : Nat) :
    denseIndices g raw b=raw (g.indexStart+b*g.indexStride) := rfl

/-- Bounds are explicit tensor/ABI/resource conditions, never numerical output
premises. yRows/yCols are exactly what contiguity needs at non-singleton sizes.
Inactive addresses need representable integer arithmetic, not backing storage. -/
structure Valid (g : Geometry) (raw : Nat → Int) : Prop where
  inner : 0<g.K
  tile : g.K≤g.block ∧ g.block≤4096
  even : g.even=Bool.true → g.block=g.K
  yRows : g.N≤g.ym ∨ g.B≤1
  yCols : g.yn=1 ∨ g.N≤1
  index : ∀ b,b<g.B → g.indexStart+b*g.indexStride<g.indexCapacity
  ids : ∀ b,b<g.B → logicalIndex g raw b=-1 ∨
    (0≤logicalIndex g raw b ∧ logicalIndex g raw b<(g.L : Int))
  x : ∀ b k,b<g.B → k<g.K → xoff g b k<g.xCapacity
  w : ∀ l n k,l<g.L → n<g.N → k<g.K → woff g l n k<g.wCapacity
  y : ∀ b n,b<g.B → n<g.N → yoff g b n<g.yCapacity
  grid : g.B<2^31 ∧ g.N<2^31
  cdiv : g.N+g.N<2^31
  index32 : g.L<2^31
  xArithmetic : g.B*g.xm+g.block*g.xk<2^31
  wArithmetic : g.L*g.l0+g.N*g.wn+256*g.wn+g.block*g.wk<2^31
  yArithmetic : g.B*g.ym+g.N+256*g.yn<2^31

theorem length_one (g : Geometry) (hn : 0<g.N) : Branches.length g.parameters=1 := by
  change (g.N+g.N-1)/g.N=1
  have h : g.N+g.N-1=(g.N-1)+1*g.N := by omega
  rw [h,Nat.add_mul_div_right _ _ hn,Nat.div_eq_of_lt (by omega)]

theorem coordinates (g : Geometry) (q : Nat) (h : q<g.count) :
    q%g.N<g.N ∧ q/g.N<g.B := by
  have hn : 0<g.N := by unfold Geometry.count at h; nlinarith
  exact ⟨Nat.mod_lt _ hn,(Nat.div_lt_iff_lt_mul hn).2 (by simpa [Geometry.count,Nat.mul_comm] using h)⟩

theorem coverage (g : Geometry) (b n : Nat) (hb : b<g.B) (hn : n<g.N) :
    b*g.N+n<g.count ∧ (b*g.N+n)/g.N=b ∧ (b*g.N+n)%g.N=n := by
  have hp : 0<g.N := by omega
  constructor
  · unfold Geometry.count; nlinarith
  constructor
  · rw [Nat.add_comm,Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hn]; omega
  · simp [Nat.add_mod,Nat.mod_eq_of_lt hn]

theorem y_injective (g : Geometry) (raw : Nat → Int) (v : Valid g raw)
    (b n b' n' : Nat) (hb : b<g.B) (hn : n<g.N) (hb' : b'<g.B) (hn' : n'<g.N)
    (eq : yoff g b n=yoff g b' n') : b=b' ∧ n=n' := by
  unfold yoff at eq
  rcases v.yRows with rows|single
  · have hr : 0<g.ym := by omega
    have he : (b*g.ym+n)/g.ym=(b'*g.ym+n')/g.ym := congrArg (fun z => z/g.ym) eq
    have h1 : (b*g.ym+n)/g.ym=b := by rw [Nat.add_comm,Nat.add_mul_div_right _ _ hr,Nat.div_eq_of_lt (by omega)]; omega
    have h2 : (b'*g.ym+n')/g.ym=b' := by rw [Nat.add_comm,Nat.add_mul_div_right _ _ hr,Nat.div_eq_of_lt (by omega)]; omega
    rw [h1,h2] at he
    exact ⟨he,by simpa [he] using eq⟩
  · have h0 : b=0 ∧ b'=0 := by omega
    exact ⟨by omega,by simpa [h0.1,h0.2] using eq⟩

theorem logical_y_address (g : Geometry) (raw : Nat → Int) (v : Valid g raw)
    (b n : Nat) (hn : n<g.N) : b*g.ym+n*g.yn=yoff g b n := by
  rcases v.yCols with h|h
  · simp [h,yoff]
  · have hz : n=0 := by omega
    simp [hz,yoff]

#axiomsClean normalization
#axiomsClean length_one
#axiomsClean coordinates
#axiomsClean coverage
#axiomsClean y_injective
#axiomsClean logical_y_address
end
end VeriTile.Bench.Repairs.LoraExpandGemv
