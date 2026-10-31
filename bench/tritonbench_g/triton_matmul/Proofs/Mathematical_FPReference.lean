import bench.tritonbench_g.triton_matmul.Sources.Numerics
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section

def blocksM (p : Parameters) := (p.M+BM p-1)/BM p
def blocksN (p : Parameters) := (p.N+BN p-1)/BN p
def perGroup (p : Parameters) := p.GROUP_SIZE_M * blocksN p
def firstM {α : Type} (p : Parameters) (s : State α) := s.pids 0 / perGroup p * p.GROUP_SIZE_M
def groupSize {α : Type} (p : Parameters) (s : State α) := min (blocksM p-firstM p s) p.GROUP_SIZE_M
def pidM {α : Type} (p : Parameters) (s : State α) := firstM p s + s.pids 0 % groupSize p s
def pidN {α : Type} (p : Parameters) (s : State α) := (s.pids 0 % perGroup p) / groupSize p s
def row {α : Type} (p : Parameters) (s : State α) (i : Fin (BM p)) := pidM p s * BM p+i.val
def col {α : Type} (p : Parameters) (s : State α) (j : Fin (BN p)) := pidN p s * BN p+j.val
def aRow {α : Type} (p : Parameters) (s : State α) (i : Fin (BM p)) := if row p s i<p.M then row p s i else 0
def bCol {α : Type} (p : Parameters) (s : State α) (j : Fin (BN p)) := if col p s j<p.N then col p s j else 0

def aOffset {α : Type} (p : Parameters) (s : State α) (k : Nat) (i : AIndex p) :=
  aRow p s i.1 * p.stride_am + i.2.1.val*p.stride_ak + k*(BK p*p.stride_ak)
def bOffset {α : Type} (p : Parameters) (s : State α) (k : Nat) (i : BIndex p) :=
  i.1.val*p.stride_bk + bCol p s i.2.1*p.stride_bn + k*(BK p*p.stride_bk)
def cOffset {α : Type} (p : Parameters) (s : State α) (i : CIndex p) :=
  p.stride_cm*row p s i.1 + p.stride_cn*col p s i.2.1

def active {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Prop :=
  row p s i.1<p.M ∧ col p s i.2.1<p.N
instance {α : Type} (p : Parameters) (s : State α) : DecidablePred (active p s) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))

def read {α : Type} [Inhabited α] (st : Storage) (c : Cell α) : α :=
  match st with | .half => c.read .fp16 | .e4m3 => c.read .f8e4

def stored {α : Type} (st : Storage) (x : α) : Cell α :=
  match st with | .half => .mk .fp16 x | .e4m3 => .mk .f8e4 x

/-- Source other=0.0 is converted to the real storage dtype; every padded lane
is retained. Row/column clamping is not the tail mask. -/
def aValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (st : Storage)
    (s : State α) (k : Nat) (i : AIndex p) : α :=
  if i.2.1.val < p.K-k*BK p then read st (s.mem p.A (aOffset p s k i))
  else M.cast none .real (dtype st) (M.literal none .real 0.0)
def bValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (st : Storage)
    (s : State α) (k : Nat) (i : BIndex p) : α :=
  if i.1.val < p.K-k*BK p then read st (s.mem p.B (bOffset p s k i))
  else M.cast none .real (dtype st) (M.literal none .real 0.0)

def update {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) (k : Nat) (acc : CIndex p → α) : CIndex p → α :=
  D.apply (some .fp32) (dtype st) (dtype st) .real {} (batch := [])
    (aValue M p st s k) (bValue M p st s k) acc

def accumulation {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) : CIndex p → α :=
  foldRange (update M D p st s) 0 p.numKBlocks 1
    (fun _ => M.literal (some .fp32) .real 0)

def outputValue {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) (i : CIndex p) : α :=
  M.cast none (dtype st) (dtype st) (M.cast none .real (dtype st) (accumulation M D p st s i))

end
end VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
