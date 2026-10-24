import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.SourceFrames
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.NativeInputs

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Resident
open VeriTile.Triton
open CrossEntropy1.Source (FP32 Cell)
open CrossEntropy1.Host (World Observations TensorArithmetic Preserves)
open CrossEntropy1.Physical (Layout Byte Retains)
noncomputable section

/-- A storage-origin interpretation, independent of numerical expressions.
The sign flag is needed for coordinate-preserving copies of negative views. -/
inductive Marker where
  | oldCell (region : RegionName) (index : Nat) (negative : Bool)
  | computed

instance : Inhabited Marker := ⟨.computed⟩

def Marker.neg : Marker → Marker
  | .oldCell r i sign => .oldCell r i (!sign)
  | .computed => .computed

theorem Marker.neg_neg (x : Marker) : x.neg.neg=x := by cases x <;> simp [Marker.neg]

/-- Loads retain storage identity. Arithmetic and constants cannot forge an
old-cell token; no numerical equality or simplification is performed. -/
def markerFP : FP32 Marker where
  literal := fun _ => .computed
  fromNat := fun _ => .computed
  negInf := .computed
  load := id
  add := fun _ _ => .computed
  sub := fun _ _ => .computed
  mul := fun _ _ => .computed
  div := fun _ _ => .computed
  exp := fun _ => .computed
  log := fun _ => .computed
  sum := fun _ => .computed
  maximum := fun _ => .computed

def markerHost : TensorArithmetic Marker where
  zero := .computed
  add := fun _ _ => .computed
  square := fun _ => .computed
  scale := fun _ _ => .computed
  sum := fun _ => .computed
  logsumexp := fun _ => .computed
  rankSum := fun _ => .computed

/-- Every cell is tagged, not only the coordinates of input tensor views.
The integer/boolean channels retain the source observations. -/
def markWorld {α σ : Type} (s : World α σ) : World Marker σ where
  native := s.native
  live := s.live
  memory r i := ⟨fun dtype => match dtype with
    | .real => .oldCell r i Bool.false
    | .int => (s.memory r i).read .int
    | .bool => (s.memory r i).read .bool
    | .fp32 => .computed
    | .fp16 => .computed
    | .bf16 => .computed
    | .f8e4 => .computed
    | .f8e5 => .computed
    | .nat => (s.memory r i).read .nat
    | .ptr => (s.memory r i).read .ptr
    | .blockPtr => (s.memory r i).read .blockPtr⟩

@[simp] theorem marked_cell {α σ : Type} (s : World α σ) (r : RegionName) (i : Nat) :
    ((markWorld s).memory r i).read .real=Marker.oldCell r i Bool.false := rfl

def rawByte {σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (s : σ) (r : RegionName) (i : Nat) (byte : Fin 4) : Byte :=
  observe s (layout.buffer s r).device ((layout.buffer s r).base+4*i+byte.val)

def byteInside {σ : Type} (layout : Layout σ) (s : σ) (r : RegionName)
    (i : Nat) (byte : Fin 4) : Prop := 4*i+byte.val<(layout.buffer s r).capacity

def signedByte (negative : Bool) (byte : Fin 4) (bits : Byte) : Byte :=
  if negative && decide (byte.val=3) then bits ^^^ 0x80 else bits

/-- Fixed abstract-storage representation. The ordinary simulation contract
must establish it for its actual marker interpreter run. It does not mention
cross entropy, expressions, a task frame, or an expected output. -/
def MarkerRep {σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (origin : σ) (source : World Marker σ) : Prop :=
  ∀ r i oldR oldI sign, source.live r →
    (source.memory r i).read .real=Marker.oldCell oldR oldI sign →
    ∀ byte : Fin 4, byteInside layout source.native r i byte →
      byteInside layout origin oldR oldI byte →
      rawByte layout observe source.native r i byte=
        signedByte sign byte (rawByte layout observe origin oldR oldI byte)

theorem MarkerRep.initial {α σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (s : World α σ) : MarkerRep layout observe s.native (markWorld s) := by
  intro r i oldR oldI sign _ same byte _ _
  simp only [marked_cell,Marker.oldCell.injEq] at same
  rcases same with ⟨rfl,rfl,rfl⟩
  rfl

/-- A source frame retains an opaque storage token, not just an Expr. The
byte index is allowed to lie in a hole or an incomplete final word. -/
theorem MarkerRep.retained_byte {α σ : Type} {layout : Layout σ} {observe}
    {initial : World α σ} {final : World Marker σ}
    (rep : MarkerRep layout observe initial.native final)
    (r : RegionName) (offset : Nat) (live : final.live r)
    (sameBuffer : layout.buffer final.native r=layout.buffer initial.native r)
    (inside : offset<(layout.buffer initial.native r).capacity)
    (same : final.memory r (offset/4)=(markWorld initial).memory r (offset/4)) :
    observe final.native (layout.buffer initial.native r).device
        ((layout.buffer initial.native r).base+offset)=
      observe initial.native (layout.buffer initial.native r).device
        ((layout.buffer initial.native r).base+offset) := by
  let byte : Fin 4 := ⟨offset%4,by omega⟩
  have coordinate : 4*(offset/4)+byte.val=offset := by dsimp [byte]; omega
  have marker : (final.memory r (offset/4)).read .real=Marker.oldCell r (offset/4) Bool.false := by
    rw [same,marked_cell]
  have result := rep r (offset/4) r (offset/4) Bool.false live marker byte
    (by simpa only [byteInside,sameBuffer,coordinate] using inside)
    (by simpa only [byteInside,coordinate] using inside)
  simpa only [rawByte,sameBuffer,signedByte,Bool.false_and,Bool.false_eq_true,if_false,
    Nat.add_assoc,coordinate] using result

#axiomsClean Marker.neg_neg
#axiomsClean MarkerRep.initial
#axiomsClean MarkerRep.retained_byte
#specNonCircular MarkerRep avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Resident
