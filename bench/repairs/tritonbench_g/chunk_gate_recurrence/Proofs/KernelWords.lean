import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelCarrier

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelWords
open VeriTile.Triton FP.Structural KernelProgram KernelCarrier
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

def inputs (a : KernelReference.Inputs Node) : KernelReference.Inputs Word where
  source := fun t i => (a.source t i).value
  gate := fun t => (a.gate t).value
  seed := fun i => (a.seed i).value
  incoming := fun t i => (a.incoming t i).value
  saved := fun t i => (a.saved t i).value
def update (isa : Instructions) (swap : Bool) (carry gate incoming : Word) : Word :=
  if swap then isa.fmaRN gate carry incoming else isa.fmaRN carry gate incoming
def forward (isa : Instructions) (swap : Bool) (a : KernelReference.Inputs Word) (seed : Bool) : Nat → Lane → Word
  | 0 => fun i => if seed then a.seed i else 0
  | n+1 => fun i => update isa swap (forward isa swap a seed n i) (a.gate n) (a.source n i)
def reverse (isa : Instructions) (swap : Bool) (a : KernelReference.Inputs Word) (N : Nat) : Nat → Lane → Word
  | 0 => fun _ => 0
  | n+1 => fun i => update isa swap (reverse isa swap a N n i) (a.gate (N-1-n)) (a.incoming (N-1-n) i)
def index (i : Fin 1024) : Lane :=
  TileShape.reshapeIndex (input:=[64,16]) (output:=[64*16]) (by decide) (i,PUnit.unit)
def contraction (isa : Instructions) (plan : Reduction.ProductPlan 1024) (carry saved : Lane → Word) : Word :=
  Reduction.product (ops isa) plan (fun i => (carry (index i),saved (index i)))

theorem forward_lowering (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (a : KernelReference.Inputs Node) (seed : Bool) (n : Nat) :
    KernelReference.forward (algebra isa plan swap) a seed n=
      (fun i => atom (forward isa swap (inputs a) seed n i)) := by
  induction n with
  | zero => cases seed <;> rfl
  | succ n ih =>
    funext i
    simp only [KernelReference.forward,ih,KernelCarrier.step]
    rfl
theorem reverse_lowering (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (a : KernelReference.Inputs Node) (N n : Nat) :
    KernelReference.reverse (algebra isa plan swap) a N n=
      (fun i => atom (reverse isa swap (inputs a) N n i)) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    funext i
    simp only [KernelReference.reverse,ih,KernelCarrier.step]
    rfl
theorem contraction_lowering (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (carry saved : Lane → Node) :
    KernelReference.contraction (algebra isa plan swap) carry saved=
      atom (contraction isa plan (fun i => (carry i).value) (fun i => (saved i).value)) := by
  simp only [KernelReference.contraction,KernelReference.flattened,KernelReference.product,
    algebra,reduce,dif_pos rfl]
  rfl

#axiomsClean forward_lowering
#axiomsClean reverse_lowering
#axiomsClean contraction_lowering
#specNonCircular forward avoiding [FP.Structural.exec,FP.Structural.run,KernelReference.forward]
#specNonCircular reverse avoiding [FP.Structural.exec,FP.Structural.run,KernelReference.reverse]
#specNonCircular contraction avoiding [FP.Structural.exec,FP.Structural.run,KernelReference.contraction]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelWords
