import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.CallerBytes

namespace VeriTile.Bench.Repairs.SgmvExpandSlice
namespace Host

inductive Role where
  | input | weight | output | starts | lengths | adapters
  deriving DecidableEq, Repr

def snapshotOrder : List Role := [.input,.weight,.output,.starts,.lengths,.adapters]
def toListOrder : List Role := [.starts,.lengths,.adapters]

inductive Stage where
  | shapeCheck | enter | snapshot (role : Role) | toList (role : Role)
  | validate | launch (batch : Nat) | leave | copy (batch : Nat) | returnNone
  deriving DecidableEq, Repr

/-- All metadata is validated before launching; every private launch and the
context leave precede caller copyback. Active batches appear in source order. -/
def protocol (activeBatches : List Nat) : List Stage :=
  [.shapeCheck,.enter] ++ snapshotOrder.map Stage.snapshot ++ toListOrder.map Stage.toList ++
    [.validate] ++ activeBatches.map Stage.launch ++ [.leave] ++
    activeBatches.map Stage.copy ++ [.returnNone]

inductive Outcome where
  | failed (stage : Stage)
  | returnedNone
  deriving DecidableEq, Repr

/-- A concrete caller-visible result retains the supplied object's identity.
Scratch allocation identities are not substituted for the supplied object. -/
structure Result where
  outcome : Outcome
  (object backing : Nat)
  memory : Bytes

def returned (view : OutputView) (writes : List ByteWrite) (entry : Bytes) : Result :=
  ⟨.returnedNone,view.object,view.backing,writeAll writes entry⟩

def failed (view : OutputView) (stage : Stage) (committed : List ByteWrite) (entry : Bytes) : Result :=
  ⟨.failed stage,view.object,view.backing,writeAll committed entry⟩

theorem returned_identity (view : OutputView) (writes : List ByteWrite) (entry : Bytes) :
    (returned view writes entry).outcome = .returnedNone ∧
    (returned view writes entry).object = view.object ∧
    (returned view writes entry).backing = view.backing := ⟨rfl,rfl,rfl⟩

/-- Before copyback, no caller-byte write has occurred. This covers every
shape/API/clone/tolist/validation/setup/private-launch/context-leave failure. -/
theorem before_copy_failure (view : OutputView) (stage : Stage) (entry : Bytes) :
    (failed view stage [] entry).memory = entry := rfl

theorem successful_frame (p : Shape) (segments : List Segment) (view : OutputView)
    (writes : List ByteWrite) (entry : Bytes)
    (localCopies : ∀ w ∈ writes, AllowedByte p segments view w.backing w.offset)
    (a b : Nat) (outside : ¬ AllowedByte p segments view a b) :
    (returned view writes entry).memory a b = entry a b :=
  authorized_copy_frame p segments view writes entry localCopies a b outside

/-- A failed copy may have committed any subset in its active destination.
The frame is unchanged; no all-or-nothing or write-order premise is used. -/
theorem failed_copy_frame (p : Shape) (segments : List Segment) (view : OutputView)
    (stage : Stage) (committed : List ByteWrite) (entry : Bytes)
    (localCopies : ∀ w ∈ committed, AllowedByte p segments view w.backing w.offset)
    (a b : Nat) (outside : ¬ AllowedByte p segments view a b) :
    (failed view stage committed entry).memory a b = entry a b :=
  authorized_copy_frame p segments view committed entry localCopies a b outside

/-- Nonempty local copy inhabitation, with an actual changed byte. The frame
model is not made consistent merely by allowing every operation to fail. -/
theorem nonempty_copy_witness :
    ∃ (view : OutputView) (entry : Bytes) (write : ByteWrite),
      (returned view [write] entry).memory write.backing write.offset ≠
        entry write.backing write.offset := by
  refine ⟨⟨7,3,0,1,1,2,Bool.false,Or.inl rfl⟩,fun _ _ => 0,⟨3,0,1⟩,?_⟩
  decide

end Host
end VeriTile.Bench.Repairs.SgmvExpandSlice
