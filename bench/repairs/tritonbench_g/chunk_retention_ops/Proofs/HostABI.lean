import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.InitialGradient
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace HostABI

structure Grid where
  x : Nat
  y : Nat
  z : Nat
  deriving DecidableEq

def gridH (c : Shape) : Grid := ⟨c.keyBlocks,c.valueBlocks,c.heads⟩
def gridO (c : Shape) : Grid := ⟨c.valueBlocks,c.chunks,c.heads⟩
def gridQKV (c : Shape) : Grid := ⟨c.keyBlocks,c.chunks,c.heads⟩
def Grid.LegalAxes (g : Grid) : Prop := g.x<2^31 ∧ g.y≤65535 ∧ g.z≤65535

/-- Per-axis CUDA launch requirements. Shared driver grid-product overflow is
outside this task's selected contract and is not a numerical assumption. -/
structure LaunchABI (c : Shape) : Prop where
  ordinary : Admitted c
  chunksY : c.chunks≤65535
  valuesY : c.valueBlocks≤65535

theorem key_blocks32 (c : Shape) (ok : Admitted c) : c.keyBlocks<2^31 := by
  have h : c.keyBlocks≤c.paddedKey := Nat.le_mul_of_pos_right _ c.key_tile_pos
  exact lt_of_le_of_lt h ok.paddedKey32

theorem launch_axes (c : Shape) (ok : LaunchABI c) :
    (gridH c).LegalAxes ∧ (gridO c).LegalAxes ∧ (gridQKV c).LegalAxes := by
  exact ⟨⟨key_blocks32 c ok.ordinary,ok.valuesY,ok.ordinary.headGrid⟩,
    ⟨by change c.valueBlocks<2^31; have h:=ok.valuesY; omega,ok.chunksY,ok.ordinary.headGrid⟩,
    ⟨key_blocks32 c ok.ordinary,ok.chunksY,ok.ordinary.headGrid⟩⟩

theorem input_bytes64 (c : Shape) (ok : Admitted c) :
    c.keySize*4<2^63 ∧ c.valueSize*4<2^63 := by
  have hk : c.T*c.K<2^31 := lt_of_le_of_lt (Nat.mul_le_mul_right c.K c.time_cover) ok.paddedKeyTime32
  have hv : c.T*c.V<2^31 := lt_of_le_of_lt (Nat.mul_le_mul_right c.V c.time_cover) ok.paddedValueTime32
  have hh := ok.headGrid
  have hkp := Nat.mul_le_mul hh (Nat.le_of_lt hk)
  have hvp := Nat.mul_le_mul hh (Nat.le_of_lt hv)
  dsimp [Shape.keySize,Shape.keyHead,Shape.valueSize,Shape.valueHead]
  constructor <;> omega

theorem history_address64 (c : Shape) (ok : Admitted c) (h t e j : Nat)
    (hh : h<c.heads) (ht : t<c.chunks) (he : e<c.K) (hj : j<c.V) :
    c.historyOffset h t e j*4<2^63 := by
  have hb := c.history_offset_bound h t e j hh ht he hj
  have hn := ok.historyBytes64
  omega

theorem partial_address64 (c : Shape) (ok : Admitted c) (b h t j : Nat)
    (hb : b<c.keyBlocks) (hh : h<c.heads) (ht : t<c.T) (hj : j<c.V) :
    c.partialOffset b h t j*4<2^63 := by
  have hx := c.partial_offset_bound b h t j hb hh ht hj
  have hn := ok.partialBytes64
  omega

/-- Canonical dense tensor metadata enforced by the four Python helpers. -/
structure TensorMeta where
  shape : List Nat
  strides : List Nat
  fp32 : Bool
  cuda : Bool
  device : Nat
  storageOffset : Nat

/-- Dense views with a nonzero storage offset are allowed: pointers already
include that offset. Fresh output allocations and external d_h0 separation are
ordinary allocation obligations. Input/input aliasing need not be forbidden. -/
def keyMeta (c : Shape) (device offset : Nat) : TensorMeta :=
  ⟨[c.B,c.H,c.T,c.K],[c.H*c.T*c.K,c.T*c.K,c.K,1],Bool.true,Bool.true,device,offset⟩
def valueMeta (c : Shape) (device offset : Nat) : TensorMeta :=
  ⟨[c.B,c.H,c.T,c.V],[c.H*c.T*c.V,c.T*c.V,c.V,1],Bool.true,Bool.true,device,offset⟩
def historyMeta (c : Shape) (device : Nat) : TensorMeta :=
  ⟨[c.B,c.H,c.chunks*c.K,c.V],[c.H*c.chunks*c.K*c.V,c.chunks*c.K*c.V,c.V,1],
    Bool.true,Bool.true,device,0⟩
def stateMeta (c : Shape) (device : Nat) : TensorMeta :=
  ⟨[c.B,c.H,c.K,c.V],[c.H*c.K*c.V,c.K*c.V,c.V,1],Bool.true,Bool.true,device,0⟩
def partialMeta (c : Shape) (device : Nat) : TensorMeta :=
  ⟨[c.keyBlocks,c.B,c.H,c.T,c.V],
    [c.B*c.H*c.T*c.V,c.H*c.T*c.V,c.T*c.V,c.V,1],Bool.true,Bool.true,device,0⟩

theorem allocation_shapes (c : Shape) (device : Nat) :
    (historyMeta c device).shape.prod=c.historySize ∧
    (stateMeta c device).shape.prod=c.stateSize ∧
    (partialMeta c device).shape.prod=c.partialSize := by
  simp only [historyMeta,stateMeta,partialMeta,List.prod_cons,List.prod_nil,Nat.mul_one,
    Shape.historySize,Shape.stateSize,Shape.partialSize,Shape.valueSize,
    Shape.historyHead,Shape.stateHead,Shape.valueHead,Shape.heads]
  constructor
  · ring
  constructor <;> ring

inductive Launch where
  | h : Grid → Bool → Bool → Launch
  | o : Grid → Launch
  | dh : Grid → Bool → Bool → Launch
  | qkv : Grid → Launch
  | sumValues : Nat → Launch

def forwardPlan (c : Shape) (seed final : Bool) : List Launch :=
  [.h (gridH c) seed final,.o (gridO c)]
def backwardPlan (c : Shape) (seed finalGrad needSeedGrad checkpoint : Bool) : List Launch :=
  (if checkpoint then [.h (gridH c) seed Bool.false] else [])++
  [.dh (gridH c) finalGrad (seed && needSeedGrad),.qkv (gridQKV c),.sumValues c.keyBlocks]

/-- Optional cotangents are materialized logically before the kernel launch.
Contiguous copies preserve these logical values, including expanded sum grads. -/
def effectiveInputs (xs : Inputs α) (zero : α) (seed incoming terminal : Bool) : Inputs α :=
  {xs with
    seed := (if seed then xs.seed else fun _ _ _ => zero)
    incoming := (if incoming then xs.incoming else fun _ _ _ => zero)
    finalIncoming := (if terminal then xs.finalIncoming else fun _ _ _ => zero)}

/-- Checkpoint recomputation uses the same immutable input values. -/
theorem checkpoint_same (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (final : Bool) (entry : Heap α) (h t e j : Nat) :
    Reference.history ops c (inputsOfHeap (Source.afterO ops c scale tree final entry)) h t e j=
      Reference.history ops c (inputsOfHeap entry) h t e j := by
  rw [Source.o_inputs]

structure DeviceState where
  current : Nat
  payload : Nat

def withDevice (target : Nat) (body : DeviceState → DeviceState) (entry : DeviceState) : DeviceState :=
  let result := body {entry with current:=target}
  {result with current:=entry.current}

theorem device_restored (target : Nat) (body : DeviceState → DeviceState) (entry : DeviceState) :
    (withDevice target body entry).current=entry.current := rfl

#axiomsClean launch_axes
#axiomsClean input_bytes64
#axiomsClean allocation_shapes
#axiomsClean checkpoint_same
#axiomsClean device_restored
end HostABI
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001

