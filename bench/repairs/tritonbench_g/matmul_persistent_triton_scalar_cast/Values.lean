import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Model

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 7000000

abbrev AIndex (p : Parameters) := TileIndex [128,p.problem.profile.blockK]
abbrev BIndex (p : Parameters) := TileIndex [p.problem.profile.blockK,256]
abbrev CIndex := TileIndex [128,256]

def aOffset (p : Parameters) (tile block : Nat) (i : AIndex p) : Nat :=
  clamp p.problem.M (p.problem.row tile i.1.val)*p.am+
    clamp p.problem.K (p.problem.k block i.2.1.val)*p.ak

def bOffset (p : Parameters) (tile block : Nat) (i : BIndex p) : Nat :=
  clamp p.problem.K (p.problem.k block i.1.val)*p.bk+
    clamp p.problem.N (p.problem.column tile i.2.1.val)*p.bn

def cOffset (p : Parameters) (tile : Nat) (i : CIndex) : Nat :=
  p.cm*clamp p.problem.M (p.problem.row tile i.1.val)+
    p.cn*clamp p.problem.N (p.problem.column tile i.2.1.val)

def active (p : Parameters) (tile : Nat) (i : CIndex) : Prop :=
  p.problem.row tile i.1.val<p.problem.M ∧ p.problem.column tile i.2.1.val<p.problem.N

instance (p : Parameters) (tile : Nat) (i : CIndex) : Decidable (active p tile i) :=
  inferInstanceAs (Decidable (_ ∧ _))

variable {α : Type} [Inhabited α]

def readCell : Storage → Cell α → α
  | .fp16,c => c.read .fp16
  | .f8e4,c => c.read .f8e4

def pack : Storage → α → Cell α
  | .fp16,x => .mk .fp16 x
  | .f8e4,x => .mk .f8e4 x

@[simp] theorem read_pack (profile : Storage) (value : α) : readCell profile (pack profile value)=value := by
  cases profile <;> simp [readCell,pack]

def aValues (P : Source.Arithmetic α α) (p : Parameters) (s : State α)
    (tile block : Nat) (i : AIndex p) : α :=
  if p.problem.k block i.2.1.val<p.problem.K then
    readCell p.problem.profile (s.mem "A" (aOffset p tile block i)) else P.cast p.problem.profile P.accumulatorZero

def bValues (P : Source.Arithmetic α α) (p : Parameters) (s : State α)
    (tile block : Nat) (i : BIndex p) : α :=
  if p.problem.k block i.1.val<p.problem.K then
    readCell p.problem.profile (s.mem "B" (bOffset p tile block i)) else P.cast p.problem.profile P.accumulatorZero

def accumulation (P : Source.Arithmetic α α) (p : Parameters) (s : State α)
    (tile : Nat) : Nat → Source.Acc α
  | 0 => fun _ _ => P.accumulatorZero
  | block+1 => P.dot p.problem.profile.blockK
      (fun r k => aValues P p s tile block (r,k,PUnit.unit))
      (fun k c => bValues P p s tile block (k,c,PUnit.unit))
      (accumulation P p s tile block)

structure InnerInvariant (P : Source.Arithmetic α α) (p : Parameters)
    (initial : State α) (tile block : Nat) (current : State α) : Prop where
  memory : current.mem=initial.mem
  offCM : current.regs .int [128] "offs_cm" = some (fun i => (p.problem.row tile i.1.val : Int))
  offCN : current.regs .int [256] "offs_cn" = some (fun i => (p.problem.column tile i.1.val : Int))
  offAM : current.regs .int [128] "offs_am" = some (fun i => (clamp p.problem.M (p.problem.row tile i.1.val) : Int))
  offBN : current.regs .int [256] "offs_bn" = some (fun i => (clamp p.problem.N (p.problem.column tile i.1.val) : Int))
  offK : current.regs .int [p.problem.profile.blockK] "offs_k_for_mask" = some (fun i => (i.1.val : Int))
  acc : current.regs .real [128,256] "accumulator" = some (fun i => accumulation P p initial tile block i.1 i.2.1)

structure Strides (p : Parameters) (b : Buffers) : Prop where
  am : p.am=b.A.rowStride
  ak : p.ak=b.A.columnStride
  bk : p.bk=b.B.rowStride
  bn : p.bn=b.B.columnStride
  cm : p.cm=b.C.rowStride
  cn : p.cn=b.C.columnStride

structure Inputs (p : Parameters) (b : Buffers) (s : State α) (entry : Memory α) : Prop where
  a : ∀ i,readCell p.problem.profile (s.mem "A" i)=entry (b.A.storage,b.A.offset+i)
  b : ∀ i,readCell p.problem.profile (s.mem "B" i)=entry (b.B.storage,b.B.offset+i)

theorem accumulation_source (P : Source.Arithmetic α α) (p : Parameters) (b : Buffers)
    (s : State α) (entry : Memory α) (strides : Strides p b) (inputs : Inputs p b s entry)
    (zero : P.cast p.problem.profile P.accumulatorZero=P.inputZero)
    (tile count : Nat) : accumulation P p s tile count=Source.accumulator P p.problem b entry tile count := by
  induction count with
  | zero => rfl
  | succ count ih =>
    simp only [accumulation,Source.accumulator,ih]
    congr 1
    · funext row k
      simp [aValues,aOffset,Source.loadA,inputs.a,strides.am,strides.ak,Tensor.address,Nat.add_assoc,zero]
    · funext k column
      simp [bValues,bOffset,Source.loadB,inputs.b,strides.bk,strides.bn,Tensor.address,Nat.add_assoc,zero]

#axiomsClean accumulation_source
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
