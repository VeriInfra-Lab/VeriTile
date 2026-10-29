import bench.tritonbench_g.matmul_kernel.Proofs.FPTypedSource
namespace VeriTile.Bench.TritonBenchG.MatmulKernel.FPValue
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section

def pidM {α : Type} (_ : Parameters) (s : State α) := s.pids 0
def pidN {α : Type} (_ : Parameters) (s : State α) := s.pids 1
def row {α : Type} (p : Parameters) (s : State α) (i : Fin (BM p)) := pidM p s*BM p+i.val
def col {α : Type} (p : Parameters) (s : State α) (j : Fin (BN p)) := pidN p s*BN p+j.val
def aRow {α : Type} (p : Parameters) (s : State α) (i : Fin (BM p)) := row p s i % 4096
def bCol {α : Type} (p : Parameters) (s : State α) (j : Fin (BN p)) := col p s j % 4096

def aOffset {α : Type} (p : Parameters) (s : State α) (k : Nat) (i : AIndex p) :=
  aRow p s i.1*4096+i.2.1.val+k*BK p
def bOffset {α : Type} (p : Parameters) (s : State α) (k : Nat) (i : BIndex p) :=
  i.1.val*4096+bCol p s i.2.1+k*(BK p*4096)
def cOffset {α : Type} (p : Parameters) (s : State α) (i : CIndex p) :=
  4096*row p s i.1+col p s i.2.1

def active {α : Type} (p : Parameters) (_ : State α) (_ : CIndex p) : Prop := True
instance {α : Type} (p : Parameters) (s : State α) : DecidablePred (active p s) := fun _ => inferInstanceAs (Decidable True)
def read {α : Type} [Inhabited α] (_ : Storage) (c : Cell α) : α := c.read .fp16
def stored {α : Type} (_ : Storage) (x : α) : Cell α := .mk .fp16 x

/-- All BLOCK_K lanes load memory on every trip, including any host-tail lanes.
No fictitious zero padding or finite-value restriction is introduced. -/
def aValue {α : Type} [Inhabited α] (_ : Algebra α) (p : Parameters) (st : Storage)
    (s : State α) (k : Nat) (i : AIndex p) : α := read st (s.mem p.A (aOffset p s k i))
def bValue {α : Type} [Inhabited α] (_ : Algebra α) (p : Parameters) (st : Storage)
    (s : State α) (k : Nat) (i : BIndex p) : α := read st (s.mem p.B (bOffset p s k i))
def update {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) (k : Nat) (acc : CIndex p → α) : CIndex p → α :=
  D.apply (some .fp32) (dtype st) (dtype st) .real {} (batch := [])
    (aValue M p st s k) (bValue M p st s k) acc
def accumulation {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) : CIndex p → α :=
  foldRange (update M D p st s) 0 p.numKBlocks 1 (fun _ => M.literal (some .fp32) .real 0)
def outputValue {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) (i : CIndex p) : α :=
  M.cast none .real .fp16 (accumulation M D p st s i)
end
end VeriTile.Bench.TritonBenchG.MatmulKernel.FPValue
