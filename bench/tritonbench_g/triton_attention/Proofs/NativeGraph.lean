import bench.tritonbench_g.triton_attention.Proofs.NativeNumerical
import bench.tritonbench_g.triton_attention.Proofs.FPBackend

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI
namespace Native
open VeriTile Triton FP.Structural
namespace Prior
export VeriTile.Bench.Repairs.TritonAttention.Whole001.Native
  (Width Ops Instructions LocalConformance ops dot local_conformance_eq)
end Prior
abbrev Word := BitVec 32
abbrev Width := Prior.Width

/-- A finite native graph. Every arithmetic port is a scalar instruction or
one component of a fixed K8 WGMMA, including its actual live-seed bit. -/
inductive Graph where
  | word (x : Word)
  | literal (x : ℝ)
  | add (a b : Graph)
  | sub (a b : Graph)
  | mul (a b : Graph)
  | fma (a b c : Graph)
  | neg (a : Graph)
  | max (a b : Graph)
  | exp2 (a : Graph)
  | div (a b : Graph)
  | narrow (a : Graph)
  | widen (a : Graph)
  | mma8 (w : Width) (a b : Fin 8 → Graph) (live : Bool) (seed : Graph)
  | shuffle (mask : Nat) (a : Graph)

def Graph.eval (I : Prior.Instructions) (literal : ℝ → Word) : Graph → Word
  | .word w => w
  | .literal x => literal x
  | .add a b => I.add (a.eval I literal) (b.eval I literal)
  | .sub a b => I.sub (a.eval I literal) (b.eval I literal)
  | .mul a b => I.mul (a.eval I literal) (b.eval I literal)
  | .fma a b c => I.fma (a.eval I literal) (b.eval I literal) (c.eval I literal)
  | .neg a => I.neg (a.eval I literal)
  | .max a b => I.max (a.eval I literal) (b.eval I literal)
  | .exp2 a => I.exp2 (a.eval I literal)
  | .div a b => I.div (a.eval I literal) (b.eval I literal)
  | .narrow a => I.narrow (a.eval I literal)
  | .widen a => I.widen (a.eval I literal)
  | .mma8 w a b live seed => I.mma8 w (fun i => (a i).eval I literal)
      (fun i => (b i).eval I literal) live (seed.eval I literal)
  | .shuffle _ a => a.eval I literal

def ops : Prior.Ops Graph where
  zero := .word 0
  one := .word 0x3f800000
  negInf := .word 0xff800000
  log2e := .word 0x3fb8aa3b
  add := .add
  sub := .sub
  mul := .mul
  fma := .fma
  neg := .neg
  max := .max
  exp2 := .exp2
  div := .div
  narrow := .narrow
  widen := .widen
  mma8 := .mma8

/-- Dot lowering is an explicit sequence of fixed K8 instructions. It is not
a variable-length numerical callback. All source K extents are divisible by 8. -/
def dot (w : Width) (k : Nat) (a b : Nat → Graph) (live : Bool) (seed : Graph) : Graph :=
  Prior.dot ops w k a b live seed

theorem local_transport (actual reference : Prior.Instructions)
    (literal : ℝ → Word) (h : Prior.LocalConformance actual reference) (g : Graph) :
    g.eval actual literal = g.eval reference literal := by
  rw [Prior.local_conformance_eq actual reference h]

theorem dot_eval (I : Prior.Instructions) (literal : ℝ → Word) (w : Width) (k : Nat)
    (a b : Nat → Graph) (live : Bool) (seed : Graph) :
    (dot w k a b live seed).eval I literal =
      Prior.dot (Prior.ops I) w k (fun i => (a i).eval I literal)
        (fun i => (b i).eval I literal) live (seed.eval I literal) := by
  unfold dot Prior.dot
  generalize List.range (k/8) = chunks
  induction chunks generalizing seed with
  | nil => rfl
  | cons c cs ih =>
    simp only [List.foldl_cons]
    rw [ih]
    rfl

#axiomsClean local_transport
#axiomsClean dot_eval
#specNonCircular Prior.LocalConformance avoiding [Graph.eval,dot]
end Native
end VeriTile.Bench.Repairs.TritonAttentionScalarABI
