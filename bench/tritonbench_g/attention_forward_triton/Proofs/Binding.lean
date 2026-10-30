import bench.tritonbench_g.attention_forward_triton.Proofs.Physical
namespace VeriTile.Bench.Repairs.AttentionForwardTriton.Whole
open Source
inductive Profile where | baseline | independent | permuted | unaligned | alignedFull
  deriving DecidableEq

def argumentClass (n : Nat) : Nat := if n=1 then 1 else if n%16=0 then 16 else 0

def pointerClasses : Profile → List Nat
  | .baseline=>[16, 16, 16, 16, 16, 16]
  | .independent=>[16, 16, 0, 16, 16, 16]
  | .permuted=>[16, 0, 16, 16, 16, 16]
  | .unaligned=>[0, 0, 0, 16, 16, 16]
  | .alignedFull=>[16, 16, 16, 16, 16, 16]

def integerClasses : Profile → List Nat
  | .baseline=>[16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 1, 1, 0]
  | .independent=>[16, 16, 16, 0, 16, 16, 16, 1, 16, 16, 16, 0, 16, 16, 16, 1, 0, 0, 0]
  | .permuted=>[16, 16, 16, 1, 16, 16, 16, 0, 16, 16, 1, 0, 16, 16, 16, 1, 0, 0, 0]
  | .unaligned=>[16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 0, 0, 0]
  | .alignedFull=>[16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 0, 0, 16]

def dynamicValues (p : Parameters) : List Nat :=
  [p.q.batch,p.q.head,p.q.row,p.q.dim,p.k.batch,p.k.head,p.k.row,p.k.dim,
   p.v.batch,p.v.head,p.v.row,p.v.dim,p.o.batch,p.o.head,p.o.row,p.o.dim,
   p.batches,p.heads,p.length]

/-- Exact non-numerical JIT signature/specialization classes from five saved
Triton 3.7.1 SM80 compilations at 8 warps, 2 stages and default fusion. This
restricts native statements; it is not a universal compiler correctness proof.
Empty shapes do not launch and need no compiled specialization. -/
def CompilerBinding (p : Parameters) (storage : TensorStorage p)
    (pl : Placement p storage) (profile : Profile) : Prop :=
  p.batches=0 ∨ p.heads=0 ∨ p.length=0 ∨
    (([Buffer.query,.key,.value,.queryScale,.keyScale,.output].map
      (fun r=>if pl.base r%16=0 then 16 else 0))=pointerClasses profile ∧
     (dynamicValues p).map argumentClass=integerClasses profile)

theorem saved_resource_fit : (131076:Nat)≤163840 := by decide
#specNonCircular CompilerBinding avoiding [invoke,runPrograms,VeriTile.Triton.FP.Structural.exec]
end VeriTile.Bench.Repairs.AttentionForwardTriton.Whole
