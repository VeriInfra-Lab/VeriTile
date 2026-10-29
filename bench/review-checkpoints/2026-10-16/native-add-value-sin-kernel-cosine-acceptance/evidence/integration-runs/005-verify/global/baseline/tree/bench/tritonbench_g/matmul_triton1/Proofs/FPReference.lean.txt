import bench.tritonbench_g.matmul_triton1.Proofs.FPSource
namespace VeriTile.Bench.TritonBenchG.MatmulTriton1.FPValue
open VeriTile.Triton FP.Structural
noncomputable section

def columnBlocks (p : Parameters) : Nat := (p.columns+p.blockColumns-1)/p.blockColumns
def row {α : Type} (p : Parameters) (s : State α) (i : Fin p.blockRows) : Nat :=
  i.val+s.pids 0/columnBlocks p*p.blockRows
def column {α : Type} (p : Parameters) (s : State α) (j : Fin p.blockColumns) : Nat :=
  j.val+s.pids 0%columnBlocks p*p.blockColumns

def aOffset {α : Type} (p : Parameters) (s : State α) (block : Nat) (i : AIndex p) : Nat :=
  row p s i.1*inner p+i.2.1.val+block*p.blockInner
def bOffset {α : Type} (p : Parameters) (s : State α) (block : Nat) (i : BIndex p) : Nat :=
  i.1.val*p.columns+column p s i.2.1+block*(p.blockInner*p.columns)
def cOffset {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Nat :=
  row p s i.1*p.columns+column p s i.2.1

def aValues {α : Type} [Inhabited α] (p : Parameters) (s : State α) (block : Nat)
    (i : AIndex p) : α := (s.mem p.input (aOffset p s block i)).read .fp32
def bValues {α : Type} [Inhabited α] (p : Parameters) (s : State α) (block : Nat)
    (i : BIndex p) : α := (s.mem p.weight (bOffset p s block i)).read .fp32

/-- Every dot consumes the next complete input tile from the initial memory.
Its result is added separately to the previous fp32 accumulator. -/
def accumulation {α : Type} [Inhabited α] (P : Primitives α) (p : Parameters)
    (initial : State α) : Nat → CIndex p → α
  | 0 => fun _ => P.zero
  | block+1 => fun i => P.add (accumulation P p initial block i)
      (P.dot (some .fp32) {allowTF32:=some Bool.false} (batch:=[])
        (aValues p initial block) (bValues p initial block) i)

structure Invariant {α : Type} [Inhabited α] (P : Primitives α) (p : Parameters)
    (initial : State α) (block : Nat) (current : State α) : Prop where
  memory : current.mem=initial.mem
  aptrs : current.regs .ptr [p.blockRows,p.blockInner] "x_ptrs" =
    some (fun i => (p.input,aOffset p initial block i))
  bptrs : current.regs .ptr [p.blockInner,p.blockColumns] "y_ptrs" =
    some (fun i => (p.weight,bOffset p initial block i))
  cptrs : current.regs .ptr [p.blockRows,p.blockColumns] "z_ptrs" =
    some (fun i => (p.output,cOffset p initial i))
  acc : current.regs .fp32 [p.blockRows,p.blockColumns] "z" =
    some (accumulation P p initial block)

#specNonCircular accumulation avoiding [source, FP.Structural.exec]
end
end VeriTile.Bench.TritonBenchG.MatmulTriton1.FPValue
