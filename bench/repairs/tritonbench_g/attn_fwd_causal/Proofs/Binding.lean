import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.Physical
import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.Native
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Whole
open Source
inductive Profile where | short | tail | prefix | unaligned | alignedFull
  deriving DecidableEq

def argumentClass (n : Nat) : Nat := if n=1 then 1 else if n%16=0 then 16 else 0

def pointerClasses : Profile → List Nat
  | .short=>[16, 16, 16, 16, 16, 16]
  | .tail=>[16, 16, 16, 16, 16, 16]
  | .prefix=>[16, 16, 16, 16, 16, 16]
  | .unaligned=>[0, 0, 0, 16, 16, 16]
  | .alignedFull=>[16, 16, 16, 16, 16, 16]

def integerClasses : Profile → List Nat
  | .short=>[16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 1, 1, 1]
  | .tail=>[16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 0, 0, 0]
  | .prefix=>[16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 0, 0, 0]
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
def SavedCompilerBinding (p : Parameters) (storage : TensorStorage p)
    (pl : Placement p storage) (profile : Profile) : Prop :=
  p.batches=0 ∨ p.heads=0 ∨ p.length=0 ∨
    (([Buffer.query,.key,.value,.queryScale,.keyScale,.output].map
      (fun r=>if pl.base r%16=0 then 16 else 0))=pointerClasses profile ∧
     (dynamicValues p).map argumentClass=integerClasses profile)

/-- Decoded local operand/routing metadata; no floating values or aggregate
numerical equalities occur in these compiler observations. -/
structure Routes where
  qkChunk : Fin 16 → Fin 8 → Nat
  pvChunk : Fin 4 → Fin 16 → Nat
  rowReduction : Fin 4 → Fin 16 → Nat
  xor2 : Fin 4 → Nat
  xor1 : Fin 4 → Nat
  featureOwner : Fin 128 → Nat
  keyOwner : Fin 64 → Nat

def LocalRouting (r : Routes) : Prop :=
  (∀ block lane,r.qkChunk block lane=block.val*8+lane.val) ∧
  (∀ block lane,r.pvChunk block lane=block.val*16+lane.val) ∧
  (∀ owner lane,r.rowReduction owner lane=(Native.localIndex owner lane).val) ∧
  (∀ owner,r.xor2 owner=Nat.xor owner.val 2) ∧
  (∀ owner,r.xor1 owner=Nat.xor owner.val 1) ∧
  (∀ feature,r.featureOwner feature=(Native.physicalOwner feature.val).val) ∧
  (∀ key,r.keyOwner key=(Native.physicalOwner key.val).val)

/-- Ordinary compiler boundary for every admitted canonical source shape.
The old saved-profile fingerprint remains available separately as evidence.
Only local integer routes and launch/resource metadata are supplied here. -/
structure CompilerBinding (_p : Parameters) (_storage : TensorStorage _p)
    (_pl : Placement _p _storage) (_profile : Profile) where
  routes : Routes
  routing : LocalRouting routes
  warps : Nat
  stages : Nat
  sharedBytes : Nat
  selected : warps=8 ∧ stages=2 ∧ sharedBytes≤163840

#specNonCircular LocalRouting avoiding [Native.Graph.eval,VeriTile.Triton.FP.Structural.exec]
#specNonCircular CompilerBinding avoiding [Native.Graph.eval,invoke,runPrograms,VeriTile.Triton.FP.Structural.exec]

theorem saved_resource_fit : (131076:Nat)≤163840 := by decide
#specNonCircular SavedCompilerBinding avoiding [invoke,runPrograms,VeriTile.Triton.FP.Structural.exec]
end VeriTile.Bench.Repairs.AttnFwdCausal.Whole
