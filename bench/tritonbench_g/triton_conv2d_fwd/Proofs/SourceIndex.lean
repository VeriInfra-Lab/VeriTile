import bench.tritonbench_g.triton_conv2d_fwd.Proofs.Grid
namespace VeriTile.Bench.Repairs.Conv2d

def pSpatial (s : Shape) := blocks s.spatial 128*128
def pFeatures (s : Shape) := blocks s.outputs 32*32
def pBatch (s : Shape) := pSpatial s/s.plane+1
def hAbs (s : Shape) := s.kh+s.ph+s.sh*s.oh
def wAbs (s : Shape) := s.kw+s.pw+s.sw*s.ow
def inputEnvelope (s : Shape) := pBatch s*s.sn+s.groups*s.cg*s.sc+s.padded*s.sc+hAbs s*s.sx+wAbs s*s.sy
def weightEnvelope (s : Shape) := (pFeatures s+s.groups*s.og)*s.so+s.padded*s.tc+s.kh*s.tx+s.kw*s.ty
def outputEnvelope (s : Shape) := pBatch s*s.outputs*s.oh*s.ow+
  (s.groups*s.og+pFeatures s)*s.oh*s.ow+s.oh*s.ow+s.ow
/-- The same conservative integer expression checked by the Python wrapper. -/
def envelope (s : Shape) :=
  s.n+s.channels+s.h+s.w+s.outputs+s.oh+s.ow+s.kh+s.kw+s.sh+s.sw+s.ph+s.pw+
  s.groups+s.cg+s.og+128+s.h+2*s.ph+s.w+2*s.pw+
  pSpatial s+pFeatures s+s.padded+pBatch s+hAbs s+wAbs s+
  s.sn+s.sc+s.sx+s.sy+s.so+s.tc+s.tx+s.ty+
  inputEnvelope s+weightEnvelope s+outputEnvelope s

def Signed32 (z : Int) : Prop := -(2^31) ≤ z ∧ z<2^31

theorem signed32_of_bound (s : Shape) (z : Int) (domain : envelope s<2^31)
    (lo : -(envelope s:Int) ≤ z) (hi : z ≤ envelope s) : Signed32 z := by
  unfold Signed32
  have dom : (envelope s:Int)<2^31 := by exact_mod_cast domain
  constructor <;> omega

theorem source_offsets (s : Shape) (v : Valid s) (pid : Program) (lane : Lane)
    (hp : pid∈programs s) (hl : lane∈lanes) :
    (cellOf pid lane).spatial<pSpatial s ∧ (cellOf pid lane).feature<pFeatures s ∧
    (cellOf pid lane).group<s.groups := by
  have hp' : pid.1<blocks s.spatial 128 ∧ pid.2.1<blocks s.outputs 32 ∧ pid.2.2<s.groups := by
    rcases pid with ⟨a,b,c⟩;simpa only [programs,List.mem_product,List.mem_range] using hp
  have hl' : lane.1<128 ∧ lane.2<32 := by
    rcases lane with ⟨a,b⟩;simpa only [lanes,List.mem_product,List.mem_range] using hl
  simp only [cellOf,pSpatial,pFeatures]
  omega

theorem input_signed_bounds (s : Shape) (v : Valid s) (g r h w c : Nat)
    (hg : g<s.groups) (hr : r<pSpatial s) (hh : h<s.kh) (hw : w<s.kw) (hc : c<s.padded) :
    -(inputEnvelope s:Int) ≤ inputPointer s g r h w c ∧
      inputPointer s g r h w c ≤ inputEnvelope s := by
  have br : batch s r ≤ pBatch s := by
    rw [batch_eq]
    unfold pBatch
    have h : r/s.plane ≤ pSpatial s/s.plane := Nat.div_le_div_right (Nat.le_of_lt hr)
    omega
  have oh : outH s r ≤ s.oh := Nat.le_of_lt (Nat.mod_lt _ v.oh)
  have ow : outW s r ≤ s.ow := Nat.le_of_lt (Nat.mod_lt _ v.ow)
  have hy : -(hAbs s:Int) ≤ ih s r h ∧ ih s r h ≤ hAbs s := by
    have mul : (s.sh:Int)*outH s r ≤ s.sh*s.oh := by exact_mod_cast Nat.mul_le_mul_left s.sh oh
    have hb : (h:Int) ≤ s.kh := by exact_mod_cast Nat.le_of_lt hh
    unfold ih hAbs
    push_cast
    constructor
    · have z : (0:Int) ≤ (h:Int)+s.kh+s.sh*s.oh+s.sh*outH s r := by positivity
      linarith
    · omega
  have hx : -(wAbs s:Int) ≤ iw s r w ∧ iw s r w ≤ wAbs s := by
    have mul : (s.sw:Int)*outW s r ≤ s.sw*s.ow := by exact_mod_cast Nat.mul_le_mul_left s.sw ow
    have hb : (w:Int) ≤ s.kw := by exact_mod_cast Nat.le_of_lt hw
    unfold iw wAbs
    push_cast
    constructor
    · have z : (0:Int) ≤ (w:Int)+s.kw+s.sw*s.ow+s.sw*outW s r := by positivity
      linarith
    · omega
  have b : (s.sn:Int)*batch s r ≤ s.sn*pBatch s := by exact_mod_cast Nat.mul_le_mul_left s.sn br
  have g' : (g:Int)*(s.cg*s.sc) ≤ s.groups*(s.cg*s.sc) := by
    exact_mod_cast Nat.mul_le_mul_right (s.cg*s.sc) (Nat.le_of_lt hg)
  have c' : (s.sc:Int)*c ≤ s.sc*s.padded := by exact_mod_cast Nat.mul_le_mul_left s.sc (Nat.le_of_lt hc)
  have ylo := mul_le_mul_of_nonneg_left hy.1 (Int.natCast_nonneg s.sx)
  have yhi := mul_le_mul_of_nonneg_left hy.2 (Int.natCast_nonneg s.sx)
  have xlo := mul_le_mul_of_nonneg_left hx.1 (Int.natCast_nonneg s.sy)
  have xhi := mul_le_mul_of_nonneg_left hx.2 (Int.natCast_nonneg s.sy)
  have pos : (0:Int) ≤ s.sn*batch s r+s.sc*g*s.cg+s.sc*c+
      pBatch s*s.sn+s.groups*s.cg*s.sc+s.padded*s.sc := by positivity
  unfold inputPointer inputEnvelope
  push_cast at *
  constructor <;> nlinarith

theorem weight_bound (s : Shape) (g f h w c : Nat)
    (hg : g<s.groups) (hf : f<pFeatures s) (hh : h<s.kh) (hw : w<s.kw) (hc : c<s.padded) :
    weightPointer s g f h w c ≤ weightEnvelope s := by
  have gf := Nat.mul_le_mul_right s.og (Nat.le_of_lt hg)
  have fo := Nat.mul_le_mul_left s.so (show f+g*s.og ≤ pFeatures s+s.groups*s.og by omega)
  have cc := Nat.mul_le_mul_left s.tc (Nat.le_of_lt hc)
  have hv := Nat.mul_le_mul_left s.tx (Nat.le_of_lt hh)
  have wv := Nat.mul_le_mul_left s.ty (Nat.le_of_lt hw)
  unfold weightPointer weightEnvelope
  nlinarith

theorem output_bound_padded (s : Shape) (v : Valid s) (x : Cell)
    (hg : x.group<s.groups) (hr : x.spatial<pSpatial s) (hf : x.feature<pFeatures s) :
    outputPointer s x ≤ outputEnvelope s := by
  have br : batch s x.spatial ≤ pBatch s := by
    rw [batch_eq];unfold pBatch
    have h : x.spatial/s.plane ≤ pSpatial s/s.plane := Nat.div_le_div_right (Nat.le_of_lt hr)
    omega
  have fb : x.group*s.og+x.feature ≤ s.groups*s.og+pFeatures s := by
    nlinarith [hg,hf]
  have hb := Nat.le_of_lt (Nat.mod_lt (x.spatial/s.ow) v.oh)
  have wb := Nat.le_of_lt (Nat.mod_lt x.spatial v.ow)
  have b := Nat.mul_le_mul_right (s.outputs*s.oh*s.ow) br
  have f := Nat.mul_le_mul_right (s.oh*s.ow) fb
  have h := Nat.mul_le_mul_right s.ow hb
  unfold outputPointer outputEnvelope outH outW
  nlinarith

theorem pointers_signed32 (s : Shape) (v : Valid s) (domain : envelope s<2^31)
    (g r f h w c : Nat) (hg : g<s.groups) (hr : r<pSpatial s) (hf : f<pFeatures s)
    (hh : h<s.kh) (hw : w<s.kw) (hc : c<s.padded) :
    Signed32 (inputPointer s g r h w c) ∧ Signed32 (weightPointer s g f h w c) ∧
      Signed32 (outputPointer s ⟨g,r,f⟩) := by
  have ie : inputEnvelope s ≤ envelope s := by unfold envelope;omega
  have we : weightEnvelope s ≤ envelope s := by unfold envelope;omega
  have oe : outputEnvelope s ≤ envelope s := by unfold envelope;omega
  have i := input_signed_bounds s v g r h w c hg hr hh hw hc
  have w' := weight_bound s g f h w c hg hf hh hw hc
  have o := output_bound_padded s v ⟨g,r,f⟩ hg hr hf
  have ie' : (inputEnvelope s:Int) ≤ envelope s := by exact_mod_cast ie
  have we' : (weightPointer s g f h w c:Int) ≤ envelope s := by exact_mod_cast le_trans w' we
  have oe' : (outputPointer s ⟨g,r,f⟩:Int) ≤ envelope s := by exact_mod_cast le_trans o oe
  refine ⟨signed32_of_bound s _ domain (by omega) (by omega),?_,?_⟩
  · exact signed32_of_bound s _ domain (by omega) we'
  · exact signed32_of_bound s _ domain (by omega) oe'

theorem output_mask (s : Shape) (v : Valid s) (x : Cell) (hg : x.group<s.groups) :
    s.Active x ↔ batch s x.spatial<s.n ∧ x.feature<s.og ∧
      outH s x.spatial<s.oh ∧ outW s x.spatial<s.ow := by
  have b : batch s x.spatial<s.n ↔ x.spatial<s.spatial := by
    rw [batch_eq,Nat.div_lt_iff_lt_mul (plane_pos s v)]
    rfl
  have h := Nat.mod_lt (x.spatial/s.ow) v.oh
  have w := Nat.mod_lt x.spatial v.ow
  simp only [Shape.Active,b,outH,outW]
  tauto

def inputTop (s : Shape) := (s.n-1)*s.sn+(s.channels-1)*s.sc+(s.h-1)*s.sx+(s.w-1)*s.sy
def weightTop (s : Shape) := (s.outputs-1)*s.so+(s.cg-1)*s.tc+(s.kh-1)*s.tx+(s.kw-1)*s.ty

theorem input_in_bounds (s : Shape) (g r h w c : Nat) (hg : g<s.groups)
    (hn : batch s r<s.n) (hc : c<s.cg) (hy0 : 0 ≤ ih s r h) (hy : ih s r h<s.h)
    (hx0 : 0 ≤ iw s r w) (hx : iw s r w<s.w) :
    0 ≤ inputPointer s g r h w c ∧ (inputPointer s g r h w c).toNat< inputTop s+1 := by
  have fc : g*s.cg+c<s.channels := by unfold Shape.channels;nlinarith
  have yn : (ih s r h).toNat<s.h := by omega
  have xn : (iw s r w).toNat<s.w := by omega
  have nb := Nat.mul_le_mul_right s.sn (show batch s r ≤ s.n-1 by omega)
  have cb := Nat.mul_le_mul_right s.sc (show g*s.cg+c ≤ s.channels-1 by omega)
  have yb := Nat.mul_le_mul_right s.sx (show (ih s r h).toNat ≤ s.h-1 by omega)
  have xb := Nat.mul_le_mul_right s.sy (show (iw s r w).toNat ≤ s.w-1 by omega)
  have eq : inputPointer s g r h w c=
      (s.sn*batch s r+s.sc*(g*s.cg+c)+s.sx*(ih s r h).toNat+s.sy*(iw s r w).toNat:Nat) := by
    unfold inputPointer
    push_cast
    rw [Int.toNat_of_nonneg hy0,Int.toNat_of_nonneg hx0]
    ring
  rw [eq]
  constructor
  · positivity
  · rw [Int.toNat_natCast]
    unfold inputTop
    nlinarith

theorem weight_in_bounds (s : Shape) (g f h w c : Nat) (hg : g<s.groups)
    (hf : f<s.og) (hc : c<s.cg) (hh : h<s.kh) (hw : w<s.kw) :
    weightPointer s g f h w c<weightTop s+1 := by
  have fo : g*s.og+f<s.outputs := by unfold Shape.outputs;nlinarith
  have ob := Nat.mul_le_mul_right s.so (show g*s.og+f ≤ s.outputs-1 by omega)
  have cb := Nat.mul_le_mul_right s.tc (show c ≤ s.cg-1 by omega)
  have hb := Nat.mul_le_mul_right s.tx (show h ≤ s.kh-1 by omega)
  have wb := Nat.mul_le_mul_right s.ty (show w ≤ s.kw-1 by omega)
  unfold weightPointer weightTop
  nlinarith

#axiomsClean pointers_signed32
#axiomsClean output_mask
#axiomsClean input_in_bounds
#axiomsClean weight_in_bounds
#axiomsClean source_offsets
#axiomsClean input_signed_bounds
end VeriTile.Bench.Repairs.Conv2d
