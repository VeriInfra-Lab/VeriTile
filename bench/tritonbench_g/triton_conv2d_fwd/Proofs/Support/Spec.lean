import bench.tritonbench_g.triton_conv2d_fwd.Kernels
namespace VeriTile.Bench.Repairs.Conv2d
open scoped BigOperators
/-- Initial logical array interpretation, using a complete nonnegative tensor
coordinate address only after padding bounds are checked. -/
def inputArray {α} (s : Shape) (zero : α) (m : Memory α)
    (n c : Nat) (y x : Int) : α :=
  if n<s.n ∧ c<s.channels ∧ 0 ≤ y ∧ y<s.h ∧ 0 ≤ x ∧ x<s.w then
    m .input (s.sn*n+s.sc*c+s.sx*y.toNat+s.sy*x.toNat) else zero
def weightArray {α} (s : Shape) (m : Memory α) (o c h w : Nat) : α :=
  m .weight (s.so*o+s.tc*c+s.tx*h+s.ty*w)
/-- Independent grouped cross-correlation; dilation is one and there is no
bias argument. No tiling, execution, compiler, or output premise appears. -/
noncomputable def mathematical (s : Shape) (m : Memory ℝ) (x : Cell) : ℝ :=
  ∑ c ∈ Finset.range s.cg, ∑ h ∈ Finset.range s.kh, ∑ w ∈ Finset.range s.kw,
    inputArray s 0 m (x.spatial/(s.oh*s.ow)) (x.group*s.cg+c)
      (((x.spatial/s.ow%s.oh:Nat):Int)*s.sh-s.ph+h)
      (((x.spatial%s.ow:Nat):Int)*s.sw-s.pw+w) *
    weightArray s m (x.group*s.og+x.feature) c h w

def refA {α β} (s : Shape) (p : ISA α β) (m : Memory α)
    (g r h w c : Nat) : α :=
  if c<s.cg then inputArray s p.zero m (r/(s.oh*s.ow)) (g*s.cg+c)
    (((r/s.ow%s.oh:Nat):Int)*s.sh-s.ph+h) (((r%s.ow:Nat):Int)*s.sw-s.pw+w) else p.zero
def refB {α β} (s : Shape) (p : ISA α β) (m : Memory α)
    (g f h w c : Nat) : α :=
  if c<s.cg ∧ f<s.og then weightArray s m (g*s.og+f) c h w else p.zero
abbrev Tap := Nat×Nat×Nat
/-- Ordered source tap order is h, w, channel blocks. The instruction-sized
step retains every padded operation even for NaNs, signed zero and infinity. -/
def taps (s : Shape) : List Tap :=
  (List.range s.kh).flatMap fun h =>
    (List.range s.kw).flatMap fun w => (List.range (blocks s.cg 32)).map fun c => (h,w,c)
def referenceStep {α β} (s : Shape) (p : ISA α β) (flags : Flags) (m : Memory α)
    (g r f : Nat) (carry : Frag α) (tap : Tap) : Frag α :=
  dot32 p flags (fun i k => refA s p m g (r+i.val) tap.1 tap.2.1 (32*tap.2.2+k.val))
    (fun k j => refB s p m g (f+j.val) tap.1 tap.2.1 (32*tap.2.2+k.val)) carry
def ordered {α β} (s : Shape) (p : ISA α β) (flags : Flags) (m : Memory α) (x : Cell) : α :=
  ((taps s).foldl (referenceStep s p flags m x.group (x.spatial/64*64) (x.feature/32*32))
    (fun _ _ => p.zero)) ⟨x.spatial%64,Nat.mod_lt _ (by decide)⟩
      ⟨x.feature%32,Nat.mod_lt _ (by decide)⟩

noncomputable def realISA : ISA ℝ ℝ where
  zero := 0
  fma := fun a b c => a*b+c
  halfRN := id
  wgmma16 := fun a b c i j => (∑ k : Fin 16,a i k*b k j)+c i j
  wgmma8 := fun a b c i j => (∑ k : Fin 8,a i k*b k j)+c i j

#specNonCircular mathematical avoiding [source,sourceValue,sourceStep,aLoad,bLoad,inputPointer,outputPointer]
#specNonCircular ordered avoiding [source,sourceValue,sourceStep,aLoad,bLoad,inputPointer,outputPointer]
end VeriTile.Bench.Repairs.Conv2d
