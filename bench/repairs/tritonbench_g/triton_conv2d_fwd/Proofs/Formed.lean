import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.SourceIndex
namespace VeriTile.Bench.Repairs.Conv2d
set_option maxHeartbeats 2400000

/-- Scalar source parameters and conservative padded maxima checked before
launch. All these positive values fit the selected i32 JIT signature. -/
def scalarParameters (s : Shape) : List Nat :=
  [s.n,s.channels,s.h,s.w,s.outputs,s.oh,s.ow,s.kh,s.kw,s.sh,s.sw,s.ph,s.pw,
   s.groups,s.cg,s.og,128,32,s.sn,s.sc,s.sx,s.sy,s.so,s.tc,s.tx,s.ty,
   pSpatial s,pFeatures s,s.padded,pBatch s,hAbs s,wAbs s]

theorem parameters_bound (s : Shape) (z : Nat) (hz : z∈scalarParameters s) : z ≤ envelope s := by
  simp only [scalarParameters,List.mem_cons,List.mem_singleton,List.not_mem_nil,or_false] at hz
  rcases hz with h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h <;>
    subst z <;> unfold envelope <;> omega

/-- Every addition, subtraction and multiplication in the signed input
pointer construction, including the masked negative spatial offsets. -/
def inputIntermediates (s : Shape) (g r h w c : Nat) : List Int :=
  let b : Int := s.sn*batch s r
  let q : Int := s.sc*g*s.cg
  let k : Int := s.sc*c
  let y : Int := s.sx*ih s r h
  let x : Int := s.sy*iw s r w
  [b,s.sc*g,q,b+q,k,(h:Int)-s.ph,(w:Int)-s.pw,
   s.sh*outH s r,s.sw*outW s r,ih s r h,iw s r w,y,x,
   b+q+k,b+q+k+y,b+q+k+y+x]

theorem input_intermediates_bound (s : Shape) (v : Valid s) (g r h w c : Nat)
    (hg : g<s.groups) (hr : r<pSpatial s) (hh : h<s.kh) (hw : w<s.kw) (hc : c<s.padded)
    (z : Int) (hz : z∈inputIntermediates s g r h w c) :
    -(envelope s:Int) ≤ z ∧ z ≤ envelope s := by
  have br : batch s r ≤ pBatch s := by
    rw [batch_eq];unfold pBatch
    have h : r/s.plane ≤ pSpatial s/s.plane := Nat.div_le_div_right (Nat.le_of_lt hr)
    omega
  have hb : (h:Int) ≤ s.kh := by exact_mod_cast Nat.le_of_lt hh
  have wb : (w:Int) ≤ s.kw := by exact_mod_cast Nat.le_of_lt hw
  have byy : (s.sh:Int)*outH s r ≤ s.sh*s.oh := by
    exact_mod_cast Nat.mul_le_mul_left s.sh (Nat.le_of_lt (Nat.mod_lt (r/s.ow) v.oh))
  have bxx : (s.sw:Int)*outW s r ≤ s.sw*s.ow := by
    exact_mod_cast Nat.mul_le_mul_left s.sw (Nat.le_of_lt (Nat.mod_lt r v.ow))
  have hy : -(hAbs s:Int) ≤ ih s r h ∧ ih s r h ≤ hAbs s := by
    unfold hAbs ih;push_cast
    have pos : (0:Int) ≤ (h:Int)+s.kh+s.sh*s.oh+s.sh*outH s r := by positivity
    constructor <;> nlinarith
  have hx : -(wAbs s:Int) ≤ iw s r w ∧ iw s r w ≤ wAbs s := by
    unfold wAbs iw;push_cast
    have pos : (0:Int) ≤ (w:Int)+s.kw+s.sw*s.ow+s.sw*outW s r := by positivity
    constructor <;> nlinarith
  have bb : (s.sn:Int)*batch s r ≤ pBatch s*s.sn := by
    exact_mod_cast (by nlinarith [Nat.mul_le_mul_left s.sn br] : s.sn*batch s r ≤ pBatch s*s.sn)
  have gg : (s.sc:Int)*g*s.cg ≤ s.groups*s.cg*s.sc := by
    exact_mod_cast (by nlinarith [Nat.mul_le_mul_right (s.sc*s.cg) (Nat.le_of_lt hg)] : s.sc*g*s.cg ≤ s.groups*s.cg*s.sc)
  have cg : (s.sc:Int)*g ≤ s.groups*s.cg*s.sc := by
    have one : (1:Int) ≤ s.cg := by exact_mod_cast v.cg
    have small : (s.sc:Int)*g ≤ s.sc*g*s.cg := by
      simpa using mul_le_mul_of_nonneg_left one (show (0:Int) ≤ s.sc*g by positivity)
    exact small.trans gg
  have cc : (s.sc:Int)*c ≤ s.padded*s.sc := by
    exact_mod_cast (by nlinarith [Nat.mul_le_mul_left s.sc (Nat.le_of_lt hc)] : s.sc*c ≤ s.padded*s.sc)
  have yl := mul_le_mul_of_nonneg_left hy.1 (Int.natCast_nonneg s.sx)
  have yu := mul_le_mul_of_nonneg_left hy.2 (Int.natCast_nonneg s.sx)
  have xl := mul_le_mul_of_nonneg_left hx.1 (Int.natCast_nonneg s.sy)
  have xu := mul_le_mul_of_nonneg_left hx.2 (Int.natCast_nonneg s.sy)
  have ie : (inputEnvelope s:Int) ≤ envelope s := by exact_mod_cast (show inputEnvelope s ≤ envelope s by unfold envelope;omega)
  have he : (hAbs s:Int) ≤ envelope s := by exact_mod_cast parameters_bound s _ (by simp [scalarParameters])
  have we : (wAbs s:Int) ≤ envelope s := by exact_mod_cast parameters_bound s _ (by simp [scalarParameters])
  have ha : (s.kh:Int)+s.ph+s.sh*s.oh ≤ envelope s := by simpa [hAbs] using he
  have wa : (s.kw:Int)+s.pw+s.sw*s.ow ≤ envelope s := by simpa [wAbs] using we
  have nb : (0:Int) ≤ s.sn*batch s r := by positivity
  have ng : (0:Int) ≤ s.sc*g*s.cg := by positivity
  have nc : (0:Int) ≤ s.sc*c := by positivity
  have b0 : (0:Int) ≤ pBatch s*s.sn := by positivity
  have g0 : (0:Int) ≤ s.groups*s.cg*s.sc := by positivity
  have c0 : (0:Int) ≤ s.padded*s.sc := by positivity
  have y0 : (0:Int) ≤ hAbs s*s.sx := by positivity
  have x0 : (0:Int) ≤ wAbs s*s.sy := by positivity
  have h0 : (0:Int) ≤ s.sh*outH s r := by positivity
  have w0 : (0:Int) ≤ s.sw*outW s r := by positivity
  have cg0 : (0:Int) ≤ s.sc*g := by positivity
  unfold inputEnvelope at ie
  push_cast at ie
  have phb : (s.ph:Int) ≤ envelope s := by exact_mod_cast parameters_bound s _ (by simp [scalarParameters])
  have pwb : (s.pw:Int) ≤ envelope s := by exact_mod_cast parameters_bound s _ (by simp [scalarParameters])
  have khb : (s.kh:Int) ≤ envelope s := by exact_mod_cast parameters_bound s _ (by simp [scalarParameters])
  have kwb : (s.kw:Int) ≤ envelope s := by exact_mod_cast parameters_bound s _ (by simp [scalarParameters])
  have hn := Int.natCast_nonneg h
  have wn := Int.natCast_nonneg w
  have phn := Int.natCast_nonneg s.ph
  have pwn := Int.natCast_nonneg s.pw
  have khn := Int.natCast_nonneg s.kh
  have kwn := Int.natCast_nonneg s.kw
  ring_nf at bb gg cc cg yl yu xl xu ie nb ng nc b0 g0 c0 y0 x0
  simp only [inputIntermediates,List.mem_cons,List.mem_singleton,List.not_mem_nil,or_false] at hz
  rcases hz with h|h|h|h|h|h|h|h|h|h|h|h|h|h|h|h <;> subst z <;> constructor <;>
    linarith only [bb,gg,cc,cg,yl,yu,xl,xu,ie,he,we,ha,wa,nb,ng,nc,b0,g0,c0,y0,x0,
      h0,w0,cg0,hb,wb,byy,bxx,hy.1,hy.2,hx.1,hx.2,phb,pwb,khb,kwb,hn,wn,phn,pwn,khn,kwn]

/-- The remaining pointer operations are nonnegative. Their partial sums
are bounded by the full pointer's padded upper envelope. -/
def weightIntermediates (s : Shape) (g f h w c : Nat) : List Nat :=
  let b := s.so*f+s.so*g*s.og
  [s.so*f,s.so*g,s.so*g*s.og,b,s.tc*c,s.tx*h,s.ty*w,
   b+s.tc*c,b+s.tc*c+s.tx*h,weightPointer s g f h w c]
def outputIntermediates (s : Shape) (x : Cell) : List Nat :=
  let a := batch s x.spatial*s.outputs*s.oh*s.ow
  let b := (x.group*s.og+x.feature)*s.oh*s.ow
  let h := outH s x.spatial*s.ow
  [a,b,h,outW s x.spatial,a+b,a+b+h,outputPointer s x]

theorem weight_intermediates_bound (s : Shape) (v : Valid s) (g f h w c z : Nat)
    (hg : g<s.groups) (hf : f<pFeatures s) (hh : h<s.kh) (hw : w<s.kw) (hc : c<s.padded)
    (hz : z∈weightIntermediates s g f h w c) : z ≤ envelope s := by
  have bound := weight_bound s g f h w c hg hf hh hw hc
  have le : weightEnvelope s ≤ envelope s := by unfold envelope;omega
  have small : s.so*g ≤ s.so*g*s.og := by
    simpa using Nat.mul_le_mul_left (s.so*g) (show 1 ≤ s.og by exact v.og)
  unfold weightPointer at bound
  simp only [weightIntermediates,List.mem_cons,List.mem_singleton,List.not_mem_nil,or_false] at hz
  rcases hz with h|h|h|h|h|h|h|h|h|h <;> subst z <;> try unfold weightPointer
  all_goals omega

theorem output_intermediates_bound (s : Shape) (v : Valid s) (x : Cell) (z : Nat)
    (hg : x.group<s.groups) (hr : x.spatial<pSpatial s) (hf : x.feature<pFeatures s)
    (hz : z∈outputIntermediates s x) : z ≤ envelope s := by
  have bound := output_bound_padded s v x hg hr hf
  have le : outputEnvelope s ≤ envelope s := by unfold envelope;omega
  unfold outputPointer at bound
  simp only [outputIntermediates,List.mem_cons,List.mem_singleton,List.not_mem_nil,or_false] at hz
  rcases hz with h|h|h|h|h|h|h <;> subst z <;> try unfold outputPointer
  all_goals omega

#axiomsClean parameters_bound
#axiomsClean input_intermediates_bound
#axiomsClean weight_intermediates_bound
#axiomsClean output_intermediates_bound
end VeriTile.Bench.Repairs.Conv2d
