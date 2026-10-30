import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealFormula

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.NativeReduction
open VeriTile Triton
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

/-- The emitted MMA layout uses two query warps and two key warps. Each
lane group owns rows r and r+8; B64 additionally owns r+32 and r+40.
Register adds are a left fold. Warp shuffles use actual lane distances
16,8,4 (query-row bits 4,2,1). B32/64 then combine the two query warps
through shared memory and a distance-1 butterfly. B16 needs one query warp.
The replicated unused query warp is not a second contribution. -/
def width (profile : Fin 3) : Nat := 16 * 2 ^ profile.val

def mmaButterfly (input : Nat → Tree) : Nat → Nat → Nat → Tree
  | 0, _, base => input base
  | depth+1, rowStride, base =>
      .add (mmaButterfly input depth (2*rowStride) base)
        (.move (.shuffle (4*rowStride))
          (mmaButterfly input depth (2*rowStride) (base+rowStride)))

def plan (profile : Fin 3) : Tree :=
  let registers := if profile.val = 2 then [0,8,32,40] else [0,8]
  let inThread := fun warp row => registerTree (registers.map (fun r => warp*16+row+r))
  let within := fun warp => mmaButterfly (inThread warp) 3 1 0
  if profile.val = 0 then within 0
  else .add (.move .shared (within 0))
    (.move (.shuffle 1) (.move .shared (within 1)))

/-- Every logical query row contributes exactly once for every supported
block. This is checked as a permutation of leaf identities, before assigning
any numerical meaning to scalar additions. -/
theorem coverage (profile : Fin 3) :
    (plan profile).leaves.Perm (List.range (width profile)) := by
  fin_cases profile <;> decide

theorem leaf_bound (profile : Fin 3) (row : Nat)
    (h : row ∈ (plan profile).leaves) : row < width profile :=
  List.mem_range.mp ((coverage profile).mem_iff.mp h)

theorem no_duplicate_leaf (profile : Fin 3) : (plan profile).leaves.Nodup :=
  (coverage profile).nodup_iff.mpr List.nodup_range

theorem all_rows (profile : Fin 3) (row : Nat) (h : row < width profile) :
    row ∈ (plan profile).leaves := (coverage profile).mem_iff.mpr (List.mem_range.mpr h)

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
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.NativeReduction
