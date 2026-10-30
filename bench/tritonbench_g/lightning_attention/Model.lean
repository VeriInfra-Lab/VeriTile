import bench.tritonbench_g.lightning_attention.Kernels
import bench.tritonbench_g.chunk_linear_attn.Model
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
abbrev Memory := ChunkLinearAttn.Memory
abbrev Matrix := ChunkLinearAttn.Matrix
abbrev scatter := @ChunkDeltaFwd.scatter
abbrev scatter_frame := @ChunkDeltaFwd.scatter_frame
abbrev range_invariant := @ChunkDeltaFwd.range_invariant
abbrev foldRange_succ := @ChunkDeltaFwd.foldRange_succ
abbrev foldRange_zero := @ChunkDeltaFwd.foldRange_zero
structure PID where
  (head channel : Nat)
def withDot {α : Type} (M : Algebra α) (D : DotPrimitive α) : Algebra α :=
  { M with
    dot := fun p {_} {_} {_} {_} a b => some (D.apply p .real .real .real {} a b)
    dotWithOptions := fun p o {_} {_} {_} {_} a b => some (D.apply p .real .real .real o a b)
    dotTyped := fun p l r o options {_} {_} {_} {_} a b => some (D.apply p l r o options a b)}
variable {α : Type} [Inhabited α]
def zero (M : Algebra α) : α := M.literal (some .fp32) .real 0
def add (M : Algebra α) (a b : α) : α := M.binary (some .fp32) .real .add a b
def cast (M : Algebra α) (a : α) : α := M.cast (some .fp32) .real .real a
def dot (D : DotPrimitive α) {m k n : Nat} (a : Matrix α m k) (b : Matrix α k n) : Matrix α m n :=
  D.apply (some .fp32) .real .real .real {} (batch:=[]) a b
def read (M : Algebra α) (m : Memory α) (r : RegionName) (offset : Nat) (active : Prop) [Decidable active] : α :=
  cast M (if active then (m r offset).read .real else zero M)
def Inputs (a b : Memory α) : Prop :=
  ∀ r : RegionName, r∈(["query","key","value","incoming"] : List RegionName) → a r=b r
theorem Inputs.refl (m : Memory α) : Inputs m m := by intro r h; rfl
theorem Inputs.trans {a b c : Memory α} (h : Inputs a b) (g : Inputs b c) : Inputs a c :=
  fun r hr => (h r hr).trans (g r hr)
theorem scatter_inputs {ι : Type} (xs : List ι) (r : RegionName)
    (hr : r∉(["query","key","value","incoming"] : List RegionName))
    (active : ι → Prop) [DecidablePred active] (offset : ι → Nat)
    (value : ι → Cell α) (m : Memory α) :
    Inputs (scatter xs r active offset value m) m := by
  intro region hi
  funext o
  apply scatter_frame
  left
  intro he
  exact hr (he ▸ hi)

def fwdQ (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) (block : Nat) : Matrix α p.BT p.D :=
  fun i => read M m "query" (pid.head*p.N*p.D+i.2.1.val+(block*p.BT+i.1.val)*p.D)
    (block*p.BT+i.1.val<p.N)
def fwdK (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) (block : Nat) : Matrix α p.D p.BT :=
  fun i => read M m "key" (pid.head*p.N*p.D+i.1.val+(block*p.BT+i.2.1.val)*p.D)
    (block*p.BT+i.2.1.val<p.N)
def fwdV (M : Algebra α) (p : Parameters) (pid : PID) (m : Memory α) (block : Nat) : Matrix α p.BT p.BM :=
  fun i => read M m "value" (pid.head*p.N*p.E+pid.channel*p.BM+i.2.1.val+(block*p.BT+i.1.val)*p.E)
    (block*p.BT+i.1.val<p.N)
def fwdScore (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID) (m : Memory α) (block : Nat) : Matrix α p.BT p.BT :=
  let score := dot D (fwdQ M p pid m block) (fwdK M p pid m block)
  fun i => if i.2.1.val ≤ i.1.val then score i else zero M
def fwdOutput (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (block : Nat) (carry : Matrix α p.D p.BM) : Matrix α p.BT p.BM :=
  let intra := dot D (fwdScore M D p pid m block) (fwdV M p pid m block)
  let inter := dot D (fwdQ M p pid m block) carry
  fun i => add M (intra i) (inter i)
def fwdNext (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) (block : Nat) (carry : Matrix α p.D p.BM) : Matrix α p.D p.BM :=
  let update := dot D (fwdK M p pid m block) (fwdV M p pid m block)
  fun i => add M (carry i) (update i)
def fwdCarry (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (m : Memory α) : Nat → Matrix α p.D p.BM
  | 0 => fun _ => zero M
  | n+1 => fwdNext M D p pid m n (fwdCarry M D p pid m n)
def fwdActive (p : Parameters) (block : Nat) (i : TileIndex [p.BT,p.BM]) := block*p.BT+i.1.val<p.N
instance (p : Parameters) (block : Nat) : DecidablePred (fwdActive p block) := fun _ => inferInstanceAs (Decidable (_<_))
def fwdOffset (p : Parameters) (pid : PID) (block : Nat) (i : TileIndex [p.BT,p.BM]) :=
  pid.head*p.N*p.E+pid.channel*p.BM+i.2.1.val+(block*p.BT+i.1.val)*p.E
def writeOutput (M : Algebra α) (p : Parameters) (pid : PID) (block : Nat)
    (values : Matrix α p.BT p.BM) (m : Memory α) : Memory α :=
  scatter (TileShape.allIndices [p.BT,p.BM]) "output" (fwdActive p block)
    (fwdOffset p pid block) (fun i => .mk .real (cast M (values i))) m
def fwdMemory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input current : Memory α) : Memory α :=
  foldRange (fun n m => writeOutput M p pid n (fwdOutput M D p pid input n (fwdCarry M D p pid input n)) m)
    0 p.NT 1 current
theorem writeOutput_inputs (M : Algebra α) (p : Parameters) (pid : PID) (block : Nat)
    (values : Matrix α p.BT p.BM) (m : Memory α) : Inputs (writeOutput M p pid block values m) m :=
  scatter_inputs _ _ (by decide) _ _ _ _

#axiomsClean scatter_inputs
#axiomsClean writeOutput_inputs
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
