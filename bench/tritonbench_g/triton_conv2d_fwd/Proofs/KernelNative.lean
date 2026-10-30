import bench.tritonbench_g.triton_conv2d_fwd.Proofs.KernelNumeric
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.Software
namespace VeriTile.Bench.Repairs.Conv2d.KernelNative
open Native
noncomputable section

/-- One output lane of each literal SM90 instruction. The vectors are exactly
that lane's operand row and column; there is no configurable reduction length,
CTA/kernel callback, or completed output port. TF32 input treatment belongs to
the K8 instruction, since these profiles emit no separate TF32 conversion. -/
structure Instructions where
  fmaRN : Word → Word → Word → Word
  halfRN : Word → Half
  wgmma16 : (Fin 16 → Half) → (Fin 16 → Half) → Word → Word
  wgmma8 : (Fin 8 → Word) → (Fin 8 → Word) → Word → Word

def embed (h : Half) : Word := fun i=>if hi:i.val<2 then h ⟨i.val,hi⟩ else 0
def half (w : Word) : Half := fun i=>w ⟨i.val,by have h:=i.isLt;omega⟩
@[simp] theorem half_embed (h : Half) : half (embed h)=h := by
  funext i
  simp [half,embed,i.isLt]

def fmas (p : Instructions) (a b : Fin 32 → Word) : Nat → Word → Word
  | 0,carry => carry
  | k+1,carry => p.fmaRN (a ⟨k%32,Nat.mod_lt _ (by decide)⟩)
      (b ⟨k%32,Nat.mod_lt _ (by decide)⟩) (fmas p a b k carry)

def mma16 (p : Instructions) (a b : Fin 32 → Word) (chunk : Nat) (carry : Word) : Word :=
  p.wgmma16 (fun k=>half (a ⟨(chunk*16+k.val)%32,Nat.mod_lt _ (by decide)⟩))
    (fun k=>half (b ⟨(chunk*16+k.val)%32,Nat.mod_lt _ (by decide)⟩)) carry

def mma8 (p : Instructions) (a b : Fin 32 → Word) (chunk : Nat) (carry : Word) : Word :=
  p.wgmma8 (fun k=>a ⟨(chunk*8+k.val)%32,Nat.mod_lt _ (by decide)⟩)
    (fun k=>b ⟨(chunk*8+k.val)%32,Nat.mod_lt _ (by decide)⟩) carry

/-- Explicit fixed K32 graph: 32 scalar FMAs, four K8 WGMMA lane steps, or
two K16 steps. The source's FP16 branch has already converted both operands.
The old accumulator enters the first native instruction as observed in PTX. -/
def lower32 (p : Instructions) (f : Flags) (a b : Fin 32 → Word) (carry : Word) : Word :=
  if f.fp16 then mma16 p a b 1 (mma16 p a b 0 carry)
  else if f.tf32 then mma8 p a b 3 (mma8 p a b 2 (mma8 p a b 1 (mma8 p a b 0 carry)))
  else fmas p a b 32 carry

def operations (p : Instructions) (f : Flags) : KernelNumeric.Operations Word :=
  ⟨zero,fun w=>embed (p.halfRN w),lower32 p f⟩

/-- Production default closes the scalar numerical operation to the existing
IEEE RN-even FMA specification. Conversion and tensor-instruction primitives
remain explicit only in the two non-default paths. -/
def software (p : Instructions) : Instructions := {p with fmaRN:=Software.fmaRN}

structure LocalLaws (actual target : Instructions) : Prop where
  fmaRN : ∀ a b c,actual.fmaRN a b c=target.fmaRN a b c
  halfRN : ∀ a,actual.halfRN a=target.halfRN a
  wgmma16 : ∀ a b c,actual.wgmma16 a b c=target.wgmma16 a b c
  wgmma8 : ∀ a b c,actual.wgmma8 a b c=target.wgmma8 a b c

theorem instructions_eq (a b : Instructions) (h : LocalLaws a b) : a=b := by
  cases a;cases b
  simp only [Instructions.mk.injEq]
  exact ⟨funext (fun a=>funext (fun b=>funext (h.fmaRN a b))),funext h.halfRN,
    funext (fun a=>funext (fun b=>funext (h.wgmma16 a b))),
    funext (fun a=>funext (fun b=>funext (h.wgmma8 a b)))⟩

#axiomsClean half_embed
#axiomsClean instructions_eq
#specNonCircular lower32 avoiding [KernelSource.kernel,KernelGrid.launch,KernelGrid.value]
end
end VeriTile.Bench.Repairs.Conv2d.KernelNative
