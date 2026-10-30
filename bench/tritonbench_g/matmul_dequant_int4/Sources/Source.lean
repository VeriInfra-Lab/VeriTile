import bench.tritonbench_g.matmul_dequant_int4.Sources.Bounds
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Source
set_option maxHeartbeats 5000000
structure FusedConfig where
  tile : Tile
  warps : Nat
  stages : Nat
  deriving DecidableEq, Repr
structure DequantConfig where
  bk : Nat
  bn : Nat
  warps : Nat
  stages : Nat
  deriving DecidableEq, Repr

def fusedConfigs : List FusedConfig := [
⟨⟨64,256,32,8⟩,4,4⟩,
⟨⟨128,128,32,8⟩,4,4⟩,
⟨⟨128,64,32,8⟩,4,4⟩,
⟨⟨64,128,32,8⟩,4,4⟩,
⟨⟨128,32,32,8⟩,4,4⟩,
⟨⟨64,64,32,8⟩,4,4⟩,
⟨⟨64,128,32,8⟩,8,2⟩,
⟨⟨64,64,64,8⟩,8,3⟩,
⟨⟨64,256,64,8⟩,4,4⟩,
⟨⟨128,128,64,8⟩,4,4⟩,
⟨⟨128,64,64,8⟩,4,4⟩,
⟨⟨64,128,64,8⟩,4,4⟩,
⟨⟨128,32,64,8⟩,4,4⟩,
⟨⟨64,64,64,8⟩,4,4⟩,
⟨⟨64,128,64,8⟩,8,2⟩,
⟨⟨32,32,128,8⟩,4,2⟩,
⟨⟨64,64,128,8⟩,8,3⟩]

def dequantConfigs : List DequantConfig := [
⟨64,32,4,4⟩,
⟨32,32,4,4⟩,
⟨32,32,2,5⟩,
⟨64,32,2,5⟩,
⟨32,64,4,4⟩,
⟨32,32,4,4⟩,
⟨32,32,2,5⟩,
⟨32,64,2,5⟩]

theorem fused_valid (c : FusedConfig) (h : c∈fusedConfigs) : c.tile.Valid := by
  simp only [fusedConfigs,List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h
  all_goals subst c;constructor <;> decide +kernel

theorem fused_divisors (c : FusedConfig) (h : c∈fusedConfigs) :
    c.tile.bm∣128 ∧ c.tile.bn∣256 ∧ c.tile.bk∣128 := by
  simp only [fusedConfigs,List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h
  all_goals subst c;constructor <;> first | decide +kernel | constructor <;> decide +kernel

theorem dequant_valid (c : DequantConfig) (h : c∈dequantConfigs) :
    0<c.bk ∧ 0<c.bn ∧ c.bk∣64 ∧ c.bn∣64 := by
  simp only [dequantConfigs,List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with h|h|h|h|h|h|h|h
  all_goals subst c;decide +kernel

structure FusedRawScope (s : Shape) (noGroups : Bool) : Prop where
  shape : s.Valid
  k128 : 128∣s.k
  n256 : 256∣s.n
  group128 : 128∣s.group
  noGroupsEq : noGroups=true ↔ s.group=s.k

theorem all_fused_domains (s : Shape) (ng : Bool) (h : FusedRawScope s ng)
    (c : FusedConfig) (hc : c∈fusedConfigs) : FusedDomain s c.tile ng :=
  ⟨h.shape,fused_valid c hc,dvd_trans (fused_divisors c hc).2.2 h.k128,
    dvd_trans (fused_divisors c hc).2.1 h.n256,
    dvd_trans (fused_divisors c hc).2.2 h.group128,h.noGroupsEq⟩

def paddedK (s : Shape) := ceil s.k 64*64
def paddedN (s : Shape) := ceil s.n 64*64
structure DequantGuard (s : Shape) : Prop where
  output : paddedK s*s.n+paddedN s<2^31
  weight : (paddedK s/8+1)*s.n+paddedN s<2^31
  scale : (paddedK s/s.group+1)*s.n+paddedN s<2^31
  zero : (paddedK s/s.group+1)*(s.n/8)+paddedN s/8<2^31
  grid : paddedK s*paddedN s<2^31
  k : s.k<2^31
  n : s.n<2^31
  group : s.group<2^31
  packed : s.k*s.n<2^31

theorem ceil_le_multiple (k b upper : Nat) (hb : 0<b) (hk : k≤upper) (hu : b∣upper) :
    ceil k b*b≤upper := by
  obtain ⟨q,hq⟩ := hu
  have hd : ceil k b≤q := by
    unfold ceil
    apply (Nat.div_le_iff_le_mul hb).mpr
    have hqb : k≤q*b := by simpa [hq,Nat.mul_comm] using hk
    omega
  rw [hq]
  nlinarith

theorem dequant_envelopes (s : Shape) (c : DequantConfig) (hc : c∈dequantConfigs) :
    ceil s.k c.bk*c.bk≤paddedK s ∧ ceil s.n c.bn*c.bn≤paddedN s := by
  obtain ⟨hk,hn,hkd,hnd⟩ := dequant_valid c hc
  exact ⟨ceil_le_multiple s.k c.bk _ hk (ceil_cover _ _ (by decide)) (dvd_mul_of_dvd_right hkd _),
    ceil_le_multiple s.n c.bn _ hn (ceil_cover _ _ (by decide)) (dvd_mul_of_dvd_right hnd _)⟩

theorem dequant_formed_addresses (s : Shape) (guard : DequantGuard s) (k n : Nat)
    (hk : k<paddedK s) (hn : n<paddedN s) :
    qAddress s k n<2^31 ∧ sAddress s k n<2^31 ∧ zAddress s k n<2^31 ∧ k*s.n+n<2^31 := by
  have hq := Nat.div_le_div_right (show k≤paddedK s by omega) (c:=8)
  have hg := Nat.div_le_div_right (show k≤paddedK s by omega) (c:=s.group)
  have hnn := Nat.div_le_div_right (show n≤paddedN s by omega) (c:=8)
  have qm := Nat.mul_le_mul_right s.n hq
  have sm := Nat.mul_le_mul_right s.n hg
  have zm := Nat.mul_le_mul_right (s.n/8) hg
  have om := Nat.mul_le_mul_right s.n (show k≤paddedK s by omega)
  have hQ := guard.weight;have hS := guard.scale;have hZ := guard.zero;have hO := guard.output
  simp only [Nat.add_mul,Nat.one_mul] at hQ hS hZ
  dsimp [qAddress,sAddress,zAddress]
  exact ⟨by omega,by omega,by omega,by omega⟩

#axiomsClean fused_valid
#axiomsClean fused_divisors
#axiomsClean dequant_valid
#axiomsClean all_fused_domains
#axiomsClean dequant_envelopes
#axiomsClean dequant_formed_addresses
end VeriTile.Bench.Repairs.MatmulDequantInt4.Source
