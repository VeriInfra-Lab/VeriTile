import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.PhysicalReturned

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeReduction
open VeriTile Triton FP.Structural
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000

/-- Movement is explicit so a shuffle/shared read is a materialization boundary
for local FMA contraction. It does not perform floating arithmetic. -/
inductive Route where
  | register
  | shuffle (distance : Nat)
  | shared
  deriving DecidableEq, Repr

inductive Tree where
  | zero
  | input (feature : Nat)
  | move (route : Route) (child : Tree)
  | add (left right : Tree)
  deriving Repr

def Tree.leaves : Tree → List Nat
  | .zero => []
  | .input k => [k]
  | .move _ a => a.leaves
  | .add a b => a.leaves ++ b.leaves

def Tree.value {α : Type} (zero : α) (input : Nat → α)
    (move : Route → α → α) (add : α → α → α) : Tree → α
  | .zero => zero
  | .input k => input k
  | .move route a => move route (a.value zero input move add)
  | .add a b => add (a.value zero input move add) (b.value zero input move add)

/-- Descending XOR shuffle distances: the root is the last distance-1 add,
its children end at distance 2, and so on. Operand order is preserved. -/
def butterfly (input : Nat → Tree) : Nat → Nat → Nat → Tree
  | 0, _, base => input base
  | depth+1, stride, base =>
      .add (butterfly input depth (2*stride) base)
        (.move (.shuffle stride) (butterfly input depth (2*stride) (base+stride)))

def registerTree (inputs : List Nat) : Tree :=
  match inputs with
  | [] => .zero
  | k::ks => ks.foldl (fun t j => .add t (.input j)) (.input k)

/-- Exactly the public choices 1/2/4/8 warps and the direct/split BK choices
1..256. Feature bits first occupy lanes, then warps, then registers. -/
def laneBits (e : Fin 9) := min e.val 5
def warpBits (e : Fin 9) (w : Fin 4) := min w.val (e.val-laneBits e)
def registerBits (e : Fin 9) (w : Fin 4) := e.val-laneBits e-warpBits e w
def plan (e : Fin 9) (w : Fin 4) : Tree :=
  let lanes := 2 ^ laneBits e
  let warps := 2 ^ warpBits e w
  let regs := 2 ^ registerBits e w
  let inThread := fun warp lane => registerTree ((List.range regs).map
    (fun reg => reg*(warps*lanes)+warp*lanes+lane))
  let within := fun warp => butterfly (inThread warp) (laneBits e) 1 0
  if warpBits e w = 0 then within 0
  else butterfly (fun warp => .move .shared (within warp)) (warpBits e w) 1 0

/-- A checked structural permutation, for all 36 public reduction profiles.
No sum equality, associativity, neutral-element law or oracle is assumed. -/
theorem coverage (e : Fin 9) (w : Fin 4) :
    (plan e w).leaves.Perm (List.range (2^e.val)) := by
  fin_cases e <;> fin_cases w <;> decide

theorem leaf_bound (e : Fin 9) (w : Fin 4) (k : Nat)
    (h : k ∈ (plan e w).leaves) : k < 2^e.val := by
  exact List.mem_range.mp ((coverage e w).mem_iff.mp h)

theorem no_duplicate_leaf (e : Fin 9) (w : Fin 4) : (plan e w).leaves.Nodup :=
  (coverage e w).nodup_iff.mpr List.nodup_range

theorem all_features (e : Fin 9) (w : Fin 4) (k : Nat) (h : k < 2^e.val) :
    k ∈ (plan e w).leaves := (coverage e w).mem_iff.mpr (List.mem_range.mpr h)

/-- Transport follows solely from scalar-add and movement-node transport,
then induction over the actual ordered tree. -/
theorem transport {α β : Type} (f : α → β) (za : α) (zb : β)
    (xa : Nat → α) (xb : Nat → β) (ma : Route → α → α) (mb : Route → β → β)
    (aa : α → α → α) (ab : β → β → β)
    (hz : f za=zb) (hx : ∀ k, f (xa k)=xb k)
    (hm : ∀ r a, f (ma r a)=mb r (f a))
    (ha : ∀ a b, f (aa a b)=ab (f a) (f b)) (t : Tree) :
    f (t.value za xa ma aa)=t.value zb xb mb ab := by
  induction t with
  | zero => exact hz
  | input k => exact hx k
  | move r a ih => simpa only [Tree.value,hm] using congrArg (mb r) ih
  | add a b ia ib => simp only [Tree.value,ha,ia,ib]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeReduction
