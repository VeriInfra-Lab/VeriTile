import bench.tritonbench_g.chunk_delta_fwd.Sources.Kernels
import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPOutput
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
abbrev Memory (α : Type) := RegionName → Nat → Cell α
abbrev PID := Nat × Nat
abbrev HIndex (p : Parameters) := TileIndex [p.BK,p.BV]
abbrev CIndex (p : Parameters) := TileIndex [p.BC,p.BV]
abbrev HValues (p : Parameters) (α : Type) := HIndex p → α
variable {α : Type} [Inhabited α]

def zero (M : Algebra α) : α := M.literal (some .fp32) .real 0
def cast (M : Algebra α) (x : α) := M.cast (some .fp32) .real .real x
def time (p : Parameters) (outer inner lane : Nat) := outer*p.BT+inner*p.BC+lane
def column (p : Parameters) (pid : PID) (lane : Nat) := pid.1*p.BV+lane
def weightBase (p : Parameters) (pid : PID) := pid.2/p.H*p.wBatch+pid.2%p.H*p.wHead
def innerCount (p : Parameters) := (p.BT+p.BC-1)/p.BC

def keyLoad (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α)
    (c b : Nat) (i : TileIndex [p.BK,p.BC]) : α :=
  M.fp32Load (if i.1.val<p.K ∧ time p c b i.2.1.val<p.T then
    (m "key" (pid.2*(p.T*p.K)+i.1.val+time p c b i.2.1.val*p.K)).read .real else zero M)
def weightLoad (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α)
    (c b : Nat) (i : TileIndex [p.BC,p.BK]) : α :=
  M.fp32Load (if time p c b i.1.val<p.T ∧ i.2.1.val<p.K then
    (m "weight" (weightBase p pid+time p c b i.1.val*p.wTime+i.2.1.val*p.wDim)).read .real else zero M)
def valueLoad (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α)
    (c b : Nat) (i : CIndex p) : α :=
  M.fp32Load (if time p c b i.1.val<p.T ∧ column p pid i.2.1.val<p.V then
    (m "value" (pid.2*(p.T*p.V)+time p c b i.1.val*p.V+column p pid i.2.1.val)).read .real else zero M)
def initial (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) : HValues p α :=
  fun i => if p.useSeed then cast M (if i.1.val<p.K ∧ column p pid i.2.1.val<p.V then
    (m "initial" (pid.2*p.K*p.V+i.1.val*p.V+column p pid i.2.1.val)).read .real else zero M) else zero M

def corrected (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (c b : Nat) (frozen : HValues p α) : CIndex p → α :=
  fun i => M.binary (some .fp32) .real .sub (valueLoad M p pid m c b i)
    (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false}
      (batch:=[]) (weightLoad M p pid m c b) (fun j => cast M (frozen j)) i)
def accumulatedStep (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (c b : Nat) (frozen acc : HValues p α) : HValues p α :=
  fun i => M.binary (some .fp32) .real .add (acc i)
    (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false}
      (batch:=[]) (keyLoad M p pid m c b) (fun j => cast M (corrected (α:=α) M D p pid m c b frozen j)) i)
def accumulated (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (c : Nat) (frozen : HValues p α) : HValues p α :=
  foldRange (fun b acc => accumulatedStep M D p pid m c b frozen acc) 0 (innerCount p) 1 (fun _ => zero M)
def next (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (c : Nat) (frozen : HValues p α) : HValues p α :=
  fun i => M.binary (some .fp32) .real .add (frozen i) (accumulated M D p pid m c frozen i)
def states (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) : Nat → HValues p α
  | 0 => initial M p pid m
  | c+1 => next M D p pid m c (states M D p pid m c)

def scatter {ι : Type} (xs : List ι) (r : RegionName) (active : ι → Prop)
    [DecidablePred active] (offset : ι → Nat) (value : ι → Cell α) (m : Memory α) : Memory α :=
  xs.foldl (fun mem i => if active i then fun r' o => if r'=r ∧ o=offset i then value i else mem r' o else mem) m
def hActive (p : Parameters) (pid : PID) (i : HIndex p) := i.1.val<p.K ∧ column p pid i.2.1.val<p.V
instance (p : Parameters) (pid : PID) : DecidablePred (hActive p pid) := fun _ => inferInstanceAs (Decidable (_∧_))
def cActive (p : Parameters) (pid : PID) (c b : Nat) (i : CIndex p) := time p c b i.1.val<p.T ∧ column p pid i.2.1.val<p.V
instance (p : Parameters) (pid : PID) (c b : Nat) : DecidablePred (cActive p pid c b) := fun _ => inferInstanceAs (Decidable (_∧_))
def hOffset (p : Parameters) (pid : PID) (c : Nat) (i : HIndex p) := pid.2*(p.NT*p.K*p.V)+c*p.K*p.V+i.1.val*p.V+column p pid i.2.1.val
def cOffset (p : Parameters) (pid : PID) (c b : Nat) (i : CIndex p) := pid.2*(p.T*p.V)+time p c b i.1.val*p.V+column p pid i.2.1.val
def fOffset (p : Parameters) (pid : PID) (i : HIndex p) := pid.2*p.K*p.V+i.1.val*p.V+column p pid i.2.1.val
def writeHistory (M : Algebra α) (p : Parameters) (pid : PID) (c : Nat) (value : HValues p α) (m : Memory α) :=
  scatter (TileShape.allIndices [p.BK,p.BV]) "history" (hActive p pid) (hOffset p pid c) (fun i => .mk .real (cast M (value i))) m
def writeCorrected (M : Algebra α) (p : Parameters) (pid : PID) (c b : Nat) (value : CIndex p → α) (m : Memory α) :=
  scatter (TileShape.allIndices [p.BC,p.BV]) "corrected" (cActive p pid c b) (cOffset p pid c b) (fun i => .mk .real (cast M (value i))) m
def writeFinal (M : Algebra α) (p : Parameters) (pid : PID) (value : HValues p α) (m : Memory α) :=
  if p.storeFinal then scatter (TileShape.allIndices [p.BK,p.BV]) "final" (hActive p pid) (fOffset p pid) (fun i => .mk .real (cast M (value i))) m else m
def writeInner (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input : Memory α) (c : Nat) (frozen : HValues p α) (m : Memory α) : Memory α :=
  foldRange (fun b mem => writeCorrected M p pid c b (corrected M D p pid input c b frozen) mem) 0 (innerCount p) 1 m
def writeOuter (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input : Memory α) (c : Nat) (m : Memory α) : Memory α :=
  writeInner M D p pid input c (states M D p pid input c)
    (writeHistory M p pid c (states M D p pid input c) m)
def programMemory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input current : Memory α) : Memory α :=
  writeFinal M p pid (states M D p pid input p.NT)
    (foldRange (writeOuter M D p pid input) 0 p.NT 1 current)

def SameInputs (a b : Memory α) : Prop :=
  ∀ r : RegionName, r∈(["key","weight","value","initial"] : List RegionName) → ∀ o, a r o=b r o

#specNonCircular states avoiding [run,FP.Structural.exec]
#specNonCircular programMemory avoiding [run,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd
