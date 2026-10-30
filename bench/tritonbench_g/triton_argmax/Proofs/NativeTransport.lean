import bench.tritonbench_g.triton_argmax.Proofs.LaneIR
import bench.tritonbench_g.triton_argmax.Proofs.NativePayload

/-! Local transport rules over the actual scalarized opcode graph. Shuffle
selection executes i32 division/remainder/xor/add and reads a 32-bit member
word. Shared-memory guards use the native lane interpreter. Static byte
addresses retain their decoded unsigned width and cannot wrap within a
shared object. These are local instruction/synchronization certificates;
none contains a value, reduction, launch, frame or output equation. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.Revision004.NativeTransport
open SelectedEmission

def shuffleLane (lane mask : Nat) : Nat :=
  let word := BitVec.ofNat 32 lane
  ((word / (32#32)) * (32#32) + ((word % (32#32)) ^^^ BitVec.ofNat 32 mask)).toNat

def member (members lane : Nat) : Bool := (BitVec.ofNat 32 members).getLsbD lane

theorem member_exact (members lane : Nat) (small : lane<32) : member members lane=Nat.testBit members lane := by
  unfold member
  rw [BitVec.getLsbD_ofNat]
  simp only [small,decide_true,Bool.true_and]

theorem shuffle_exact (lane mask : Nat) (small : lane<2^31) (maskSmall : mask<32) :
    shuffleLane lane mask=lane/32*32+Nat.xor (lane%32) mask := by
  have laneWord : lane<2^32 := by omega
  have remainder : lane%32<2^5 := Nat.mod_lt _ (by decide)
  have xorSmall : Nat.xor (lane%32) mask<32 := Nat.xor_lt_two_pow remainder maskSmall
  have baseSmall : lane/32*32<2^32 := by omega
  have resultSmall : lane/32*32+Nat.xor (lane%32) mask<2^32 := by omega
  simp only [shuffleLane,BitVec.toNat_add,BitVec.toNat_mul,BitVec.toNat_udiv,
    BitVec.toNat_umod,BitVec.toNat_xor,BitVec.toNat_ofNat]
  norm_num only at laneWord baseSmall resultSmall ⊢
  rw [Nat.mod_eq_of_lt laneWord,Nat.mod_eq_of_lt (by omega : mask<4294967296),
    Nat.mod_eq_of_lt baseSmall]
  change (lane/32*32+Nat.xor (lane%32) mask)%4294967296=_
  exact Nat.mod_eq_of_lt resultSmall

structure Shuffle {n : Nat} (graph : Graph n) (destination source : Ref) where
  edge : ShuffleEdge graph destination source
  laneTyped : destination.lane<2^31

theorem Shuffle.selected (edge : Shuffle graph destination source) :
    source=⟨shuffleLane destination.lane edge.edge.mask,edge.edge.sourceRegister⟩ := by
  rw [shuffle_exact _ _ edge.laneTyped edge.edge.maskBound]
  exact edge.edge.selected

theorem Shuffle.members (edge : Shuffle graph destination source) :
    member edge.edge.members (destination.lane%32)=Bool.true ∧
      member edge.edge.members (source.lane%32)=Bool.true := by
  simpa only [member_exact _ _ (Nat.mod_lt _ (by decide))] using
    And.intro edge.edge.destinationMember edge.edge.sourceMember

structure Shared {n : Nat} (graph : Graph n) (destination source : Ref) where
  edge : SharedEdge graph destination source
  loadTyped : LaneIR.GuardValid destination.lane edge.loadGuard
  storesTyped : ∀ site slot width payload guard,
    graph.instruction site=some (.sharedStore slot width payload guard) →
      LaneIR.GuardValid site.lane guard ∧ slot<2^32 ∧ width<2^32 ∧ slot+width≤2^32

theorem Shared.active (edge : Shared graph destination source) :
    LaneIR.guard edge.edge.writer.lane edge.edge.storeGuard=Bool.true ∧
      LaneIR.guard destination.lane edge.edge.loadGuard=Bool.true := by
  rw [LaneIR.guard_exact _ _ (edge.storesTyped _ _ _ _ _ edge.edge.store).1,
    LaneIR.guard_exact _ _ edge.loadTyped]
  exact ⟨edge.edge.storeActive,edge.edge.loadActive⟩

theorem Shared.lastWriter (edge : Shared graph destination source)
    {other payload : Ref} {slot width : Nat} {guard : LaneGuard}
    (instruction : graph.instruction other=some (.sharedStore slot width payload guard))
    (active : LaneIR.guard other.lane guard=Bool.true)
    (overlap : overlaps edge.edge.slot edge.edge.width slot width)
    (earlier : graph.time other<graph.time destination) : graph.time other≤graph.time edge.edge.writer := by
  rw [LaneIR.guard_exact _ _ (edge.storesTyped _ _ _ _ _ instruction).1] at active
  exact edge.edge.lastWriter _ _ _ _ _ instruction active overlap earlier

structure Wide {n : Nat} (graph : Graph n) (destination source : Ref) where
  edge : SelectedEmission.WideShuffle graph destination source
  lowTyped : edge.lowShuffle.lane<2^31
  highTyped : edge.highShuffle.lane<2^31

inductive Checked {n : Nat} (graph : Graph n) (bits : Nat) : Ref → Ref → Type where
  | copy {destination source} (instruction : graph.instruction destination=some (.copy source))
      (earlier : graph.time source<graph.time destination) : Checked graph bits destination source
  | shuffle {destination source} (edge : Shuffle graph destination source)
      (completeWord : bits=32) : Checked graph bits destination source
  | wide {destination source} (edge : Wide graph destination source)
      (completeWord : bits=64) : Checked graph bits destination source
  | shared {destination source} (edge : Shared graph destination source)
      (completeWord : edge.edge.width*8=bits) : Checked graph bits destination source

theorem Checked.source {source : Ref} (edge : Checked graph bits destination source) : Transport graph destination source := by
  cases edge with
  | copy instruction earlier => exact .copy instruction earlier
  | shuffle edge completeWord => exact .shuffle edge.edge
  | wide edge completeWord => exact .wide edge.edge
  | shared edge completeWord => exact .shared edge.edge

/-- A raw shuffle receives one selected b32 operand. The wide macro executes
both selected halves and the exact emitted widening/shift/OR reconstruction.
The shared case preserves the complete checked byte-width word. -/
def Checked.transfer {source : Ref} (edge : Checked graph bits destination source) (word : BitVec bits) : BitVec bits :=
  match edge with
  | .copy _ _ => word
  | .shuffle _ _ => word
  | .shared _ _ => NativePayload.sharedWord word
  | .wide _ width =>
    let word64 := word.cast width
    (NativePayload.join (NativePayload.low word64) (NativePayload.high word64)).cast width.symm

theorem Checked.transfer_exact {source : Ref} (edge : Checked graph bits destination source) (word : BitVec bits) :
    edge.transfer word=word := by
  cases edge with
  | copy => rfl
  | shuffle => rfl
  | shared => exact NativePayload.sharedWord_exact word
  | wide edge width =>
    cases width
    exact NativePayload.reassemble word

#axiomsClean Checked.transfer_exact
#axiomsClean member_exact
#axiomsClean shuffle_exact
#axiomsClean Shuffle.selected
#axiomsClean Shuffle.members
#axiomsClean Shared.active
#axiomsClean Shared.lastWriter
#axiomsClean Checked.source
end VeriTile.Bench.Repairs.TritonArgmax.Revision004.NativeTransport
