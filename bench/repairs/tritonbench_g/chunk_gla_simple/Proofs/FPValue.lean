import bench.repairs.tritonbench_g.chunk_gla_simple.Proofs.Typed
import bench.tritonbench_g.parallel_attention.Proofs.FPRange
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.FPValue
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Host
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section

abbrev OIndex (c : Shape) := TileIndex [c.BT,c.BV]
abbrev SIndex (c : Shape) := TileIndex [c.BT,c.BT]
abbrev QIndex (c : Shape) := TileIndex [c.BT,c.BK]
abbrev KIndex (c : Shape) := TileIndex [c.BK,c.BT]
abbrev HIndex (c : Shape) := TileIndex [c.BK,c.BV]
def row {α : Type} (c : Shape) (s : State α) (i : Fin c.BT) := s.pids 1*c.BT+i.val
def column {α : Type} (c : Shape) (s : State α) (i : Fin c.BV) := s.pids 0*c.BV+i.val
def key (c : Shape) (block : Nat) (i : Fin c.BK) := block*c.BK+i.val

def qOffset {α : Type} (c : Shape) (s : State α) (block : Nat) (i : QIndex c) :=
  s.pids 2*(c.T*c.K)+row c s i.1*c.K+key c block i.2.1

def kOffset {α : Type} (c : Shape) (s : State α) (block : Nat) (i : KIndex c) :=
  s.pids 2*(c.T*c.K)+key c block i.1+row c s i.2.1*c.K

def hOffset {α : Type} (c : Shape) (s : State α) (block : Nat) (i : HIndex c) :=
  s.pids 2*(c.NT*c.K*c.V)+s.pids 1*c.K*c.V+key c block i.1*c.V+column c s i.2.1

def oOffset {α : Type} (c : Shape) (s : State α) (i : OIndex c) :=
  s.pids 2*(c.T*c.V)+row c s i.1*c.V+column c s i.2.1

def gOffset {α : Type} (c : Shape) (s : State α) (i : Fin c.BT) := s.pids 2*c.T+row c s i

def active {α : Type} (c : Shape) (s : State α) (i : OIndex c) : Prop :=
  row c s i.1<c.T ∧ column c s i.2.1<c.V
instance {α : Type} (c : Shape) (s : State α) : DecidablePred (active c s) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

def zero {α : Type} (M : Algebra α) := M.literal (some .fp32) .real 0

def qValue {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (s : State α)
    (block : Nat) (i : QIndex c) : α :=
  M.fp32Load (if row c s i.1<c.T ∧ key c block i.2.1<c.K then (s.mem "q" (qOffset c s block i)).read .real else zero M)

def kValue {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (s : State α)
    (block : Nat) (i : KIndex c) : α :=
  M.fp32Load (if key c block i.1<c.K ∧ row c s i.2.1<c.T then (s.mem "k" (kOffset c s block i)).read .real else zero M)

def hValue {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (s : State α)
    (block : Nat) (i : HIndex c) : α :=
  M.fp32Load (if key c block i.1<c.K ∧ column c s i.2.1<c.V then (s.mem "h" (hOffset c s block i)).read .real else zero M)

def vValue {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (s : State α) (i : OIndex c) : α :=
  M.fp32Load (if active c s i then (s.mem "v" (oOffset c s i)).read .real else zero M)

def gValue {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (s : State α) (i : Fin c.BT) : α :=
  M.fp32Load (if row c s i<c.T then (s.mem "g" (gOffset c s i)).read .real else zero M)

abbrev Acc (α : Type) (c : Shape) := (OIndex c → α) × (SIndex c → α)

def update {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (d : Typed.Dots M c)
    (s : State α) (block : Nat) (acc : Acc α c) : Acc α c :=
  (fun i => M.binary (some .fp32) .real .add (acc.1 i)
      (d.qh (qValue M c s block) (hValue M c s block) i),
   fun i => M.binary (some .fp32) .real .add (acc.2 i)
      (d.qk (qValue M c s block) (kValue M c s block) i))

def accumulated {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (d : Typed.Dots M c) (s : State α) :=
  foldRange (update M c d s) 0 ((c.K+c.BK-1)/c.BK) 1 (fun _ => zero M, fun _ => zero M)

def gatedO {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (s : State α) (acc : Acc α c) (i : OIndex c) :=
  M.binary (some .fp32) .real .mul (acc.1 i) (M.unary (some .fp32) .exp (gValue M c s i.1))

def gatedS {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (s : State α) (acc : Acc α c) (i : SIndex c) :=
  if i.2.1.val ≤ i.1.val then M.binary (some .fp32) .real .mul (acc.2 i)
    (M.unary (some .fp32) .exp (M.binary (some .fp32) .real .sub (gValue M c s i.1) (gValue M c s i.2.1)))
  else zero M

/-- Independent recurrence over initial memory. All three dot results, both
carries, padding, exp/sub/mul and explicit source casts remain observable. -/
def outputValue {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (d : Typed.Dots M c)
    (s : State α) (scale : ℝ) (i : OIndex c) :=
  let a := accumulated M c d s
  M.cast (some .fp32) .real .fp32
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .add (gatedO M c s a i)
        (d.sv (fun j => M.cast (some .fp32) .real .real (gatedS M c s a j)) (vValue M c s) i))
      (M.literal (some .fp32) .real scale))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimple.FPValue
