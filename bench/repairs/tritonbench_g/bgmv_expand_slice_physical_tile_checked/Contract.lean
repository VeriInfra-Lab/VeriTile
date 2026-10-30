import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Kernels
import VeriTile.Torch.Tensor
import Mathlib.Data.Nat.Log

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Contract
open VeriTile.Torch
noncomputable section
open Classical

structure TensorInfo extends TensorMetadata where
  isConj : Bool
  object : Nat
  backing : Nat
  storageOffset : Nat
  strides : List Nat
  storageBase : Nat
  storageBytes : Nat
structure Scalars where
  offset : Int
  size : Int
  offsetInt : Bool
  sizeInt : Bool
  addBool : Bool
  add : Bool
inductive Argument | nonTensor | tensor (metadata : TensorInfo)

def rows (t : TensorInfo) := t.shape[0]?.getD 0
def columns (y : TensorInfo) := y.shape[1]?.getD 0
def rank (w : TensorInfo) := w.shape.getLastD 0
def width (w : TensorInfo) := w.shape[w.shape.length-2]?.getD 0
def block (w : TensorInfo) := 2^(Nat.clog 2 (rank w))
def Dense (t : TensorInfo) : Prop :=
  t.isContiguous=Bool.true ∧ t.isNeg=Bool.false ∧ t.isConj=Bool.false

def Layouts (x w y i : TensorInfo) : Prop :=
  x.layout=.strided ∧ w.layout=.strided ∧ y.layout=.strided ∧ i.layout=.strided
def DTypes (x w y i : TensorInfo) : Prop :=
  x.dtype=.floating .fp32 ∧ w.dtype=.other "float16" ∧ y.dtype=.floating .fp32 ∧
  (i.dtype=.other "int32" ∨ i.dtype=.other "int64")
def Ranks (x w y i : TensorInfo) : Prop :=
  x.shape.length=2 ∧ y.shape.length=2 ∧ i.shape.length=1 ∧ (w.shape.length=3 ∨ w.shape.length=4)
def Singleton (w : TensorInfo) : Prop := w.shape.length=4 → w.shape[1]?.getD 0=1
def Devices (x w y i : TensorInfo) : Prop :=
  x.device.kind="cuda" ∧ w.device=x.device ∧ y.device=x.device ∧ i.device=x.device
def ScalarTypes (s : Scalars) : Prop :=
  s.offsetInt=Bool.true ∧ s.sizeInt=Bool.true ∧ s.addBool=Bool.true
def Shapes (x w y i : TensorInfo) (s : Scalars) : Prop :=
  0<rank w ∧ columns x=rank w ∧ i.numel≤rows x ∧ i.numel≤rows y ∧
  s.size=(width w:Int) ∧ 0≤s.offset ∧ s.offset+(width w:Int)≤columns y
def Dimensions (w y i : TensorInfo) : Prop :=
  i.numel≤65535 ∧ rows w<2^31 ∧ width w<2^31 ∧ rank w<2^31 ∧ columns y<2^31 ∧ width w*rank w<2^31
def Formed (x w y : TensorInfo) : Prop :=
  4*(x.numel+block w)<2^63 ∧ 2*(w.numel+255*rank w+block w)<2^63 ∧ 4*(y.numel+255)<2^63
/-- Actual half-open backing byte ranges; StorageImpl identity is irrelevant. -/
def ByteSeparate (a b : TensorInfo) : Prop :=
  a.storageBytes=0 ∨ b.storageBytes=0 ∨
  a.storageBase+a.storageBytes≤b.storageBase ∨ b.storageBase+b.storageBytes≤a.storageBase
def Separate (x w y i : TensorInfo) : Prop :=
  ByteSeparate y x ∧ ByteSeparate y w ∧ ByteSeparate y i
/-- Empty calls never form the static [256,BK] tile. -/
def StaticTile (w i : TensorInfo) : Prop :=
  i.numel=0 ∨ width w=0 ∨ 256*block w≤2^20
def Aligned (x w y i : TensorInfo) : Prop :=
  i.numel=0 ∨ width w=0 ∨
    ((x.numel=0 ∨ x.dataPtr.toNat%16=0) ∧ (w.numel=0 ∨ w.dataPtr.toNat%16=0) ∧ (y.numel=0 ∨ y.dataPtr.toNat%16=0))
def Metadata (w i : TensorInfo) (indices : Nat → Int) : Prop :=
  i.numel=0 ∨ width w=0 ∨ ∀ b,b<i.numel → indices b=-1 ∨ (0≤indices b ∧ indices b<(rows w:Int))

structure InputContract (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int) : Prop where
  layouts : Layouts x w y i
  dtypes : DTypes x w y i
  ranks : Ranks x w y i
  singleton : Singleton w
  devices : Devices x w y i
  dense : Dense x ∧ Dense w ∧ Dense y ∧ Dense i
  scalarTypes : ScalarTypes s
  shapes : Shapes x w y i s
  dimensions : Dimensions w y i
  formed : Formed x w y
  separate : Separate x w y i
  aligned : Aligned x w y i
  metadata : Metadata w i indices
  tile : StaticTile w i

def pythonChecks (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int) : List Bool :=
  [decide (Layouts x w y i),decide (DTypes x w y i),decide (Ranks x w y i),
   decide (Singleton w),decide (Devices x w y i),decide (Dense x ∧ Dense w ∧ Dense y ∧ Dense i),
   decide (ScalarTypes s),decide (Shapes x w y i s),decide (Dimensions w y i),
   decide (Formed x w y),decide (Separate x w y i),decide (Aligned x w y i),decide (Metadata w i indices),decide (StaticTile w i)]
def Admitted (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int) := (pythonChecks x w y i s indices).all id=Bool.true

theorem python_checks_iff (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int) :
    Admitted x w y i s indices ↔ InputContract x w y i s indices := by
  simp only [Admitted,pythonChecks,List.all_cons,List.all_nil,Bool.and_eq_true,decide_eq_true_eq,id_eq,and_true]
  constructor
  · rintro ⟨a,b,c,d,e,f,g,h,j,k,l,m,n,o⟩;exact ⟨a,b,c,d,e,f,g,h,j,k,l,m,n,o⟩
  · intro h;exact ⟨h.layouts,h.dtypes,h.ranks,h.singleton,h.devices,h.dense,h.scalarTypes,h.shapes,h.dimensions,h.formed,h.separate,h.aligned,h.metadata,h.tile⟩

theorem block_covered (w : TensorInfo) : rank w≤block w :=
  (Nat.clog_le_iff_le_pow (by decide : 1<2)).mp (Nat.le_refl _)

def call (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int) (ok : Admitted x w y i s indices) : Call := by
  have h:=(python_checks_iff x w y i s indices).mp ok
  exact {
    batches := i.numel
    adapters := rows w
    N := width w
    K := rank w
    columns := columns y
    slice := s.offset.toNat
    BK := block w
    blockPositive := by change 0<2^(Nat.clog 2 (rank w));exact Nat.pow_pos (by decide)
    rankCovered := block_covered w
    sliceInside := by have hs:=h.shapes.2.2.2.2.2;omega
    even := decide (rank w%block w=0)
    evenExact := by
      intro he
      have hz : rank w%block w=0 := of_decide_eq_true he
      have covered:=block_covered w
      by_cases eq : rank w=block w
      · exact Or.inl eq
      · have lt : rank w<block w := by omega
        rw [Nat.mod_eq_of_lt lt] at hz
        exact Or.inr hz
    add := s.add
    inputType := .fp32
    weightType := .fp16
    outputType := .fp32 }

theorem primary (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int) (ok : Admitted x w y i s indices) :
    Kernels.Primary (call x w y i s indices ok) := ⟨rfl,rfl,rfl⟩

theorem selected_valid (x w y i : TensorInfo) (s : Scalars) (indices : Nat → Int) (ok : Admitted x w y i s indices)
    (b n : Nat) (hb : b<(call x w y i s indices ok).batches) (hn : n<(call x w y i s indices ok).N) :
    indices b=-1 ∨ (0≤indices b ∧ (indices b).toNat<(call x w y i s indices ok).adapters) := by
  have h:=(python_checks_iff x w y i s indices).mp ok
  change b<i.numel at hb
  change n<width w at hn
  rcases h.metadata with empty|empty|valid
  · omega
  · omega
  · rcases valid b hb with skip|active
    · exact Or.inl skip
    · right;change 0≤indices b ∧ (indices b).toNat<rows w;omega

#axiomsClean python_checks_iff
#axiomsClean call
#axiomsClean primary
#axiomsClean selected_valid
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Contract
