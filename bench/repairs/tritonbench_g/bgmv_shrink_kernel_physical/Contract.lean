import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Target
import bench.repairs.tritonbench_g.bgmv_expand_slice.Contract

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Contract
open VeriTile.Torch
noncomputable section
open Classical
structure TensorInfo extends VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.TensorInfo where
  storageDataPtr : Nat
  storageNbytes : Nat

abbrev rows (t : TensorInfo) := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rows t.toTensorInfo
abbrev columns (t : TensorInfo) := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.columns t.toTensorInfo
abbrev rank (t : TensorInfo) := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rank t.toTensorInfo
abbrev width (t : TensorInfo) := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.width t.toTensorInfo
abbrev Layouts (x w y i : TensorInfo) := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Layouts x.toTensorInfo w.toTensorInfo y.toTensorInfo i.toTensorInfo
abbrev Ranks (x w y i : TensorInfo) := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Ranks x.toTensorInfo w.toTensorInfo y.toTensorInfo i.toTensorInfo
abbrev Singleton (w : TensorInfo) := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Singleton w.toTensorInfo
abbrev Devices (x w y i : TensorInfo) := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Devices x.toTensorInfo w.toTensorInfo y.toTensorInfo i.toTensorInfo
abbrev Dense (t : TensorInfo) := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Dense t.toTensorInfo
abbrev Dimensions (w y i : TensorInfo) := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Dimensions w.toTensorInfo y.toTensorInfo i.toTensorInfo
def ByteSeparate (a b : TensorInfo) : Prop :=
  a.storageNbytes=0 ∨ b.storageNbytes=0 ∨
  a.storageDataPtr+a.storageNbytes ≤ b.storageDataPtr ∨
  b.storageDataPtr+b.storageNbytes ≤ a.storageDataPtr
/-- Empty calls return before both physical and frontend tile checks. -/
def Separate (x w y i : TensorInfo) : Prop :=
  i.numel=0 ∨ width w=0 ∨
  (ByteSeparate y x ∧ ByteSeparate y w ∧ ByteSeparate y i)
abbrev Aligned (x w y i : TensorInfo) := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Aligned x.toTensorInfo w.toTensorInfo y.toTensorInfo i.toTensorInfo
abbrev Metadata (w i : TensorInfo) (indices : Nat → Int) := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Metadata w.toTensorInfo i.toTensorInfo indices

structure Scale where
  value : ℝ
  isFloat : Bool
  finite : Bool
  /-- Actual FP32 launch scalar bits, supplied by the ordinary host ABI. -/
  nativeWord : BitVec 32
def block (w : TensorInfo) := 2^(Nat.clog 2 (width w))
def StaticTile (w i : TensorInfo) : Prop := i.numel=0 ∨ width w=0 ∨ 256*block w ≤ 2^20
def formedK (w : TensorInfo) := max 1 ((rank w+16383)/16384)*16384
def DTypes (x w y i : TensorInfo) : Prop :=
  (x.dtype=.other "float16" ∨ x.dtype=.other "bfloat16") ∧ w.dtype=x.dtype ∧
  y.dtype=.floating .fp32 ∧ (i.dtype=.other "int32" ∨ i.dtype=.other "int64")
def Shapes (x w y i : TensorInfo) : Prop :=
  columns x=rank w ∧ i.numel ≤ rows x ∧ i.numel ≤ rows y ∧ width w ≤ columns y
def Formed (x w y : TensorInfo) : Prop :=
  2*(x.numel+formedK w) < 2^63 ∧ 2*(w.numel+block w*rank w+formedK w) < 2^63 ∧
  4*(y.numel+block w) < 2^63
structure InputContract (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) : Prop where
  layouts : Layouts x w y i
  dtypes : DTypes x w y i
  ranks : Ranks x w y i
  singleton : Singleton w
  devices : Devices x w y i
  dense : Dense x ∧ Dense w ∧ Dense y ∧ Dense i
  scale : s.isFloat=Bool.true ∧ s.finite=Bool.true
  shapes : Shapes x w y i
  dimensions : Dimensions w y i
  formed : Formed x w y
  separate : Separate x w y i
  aligned : Aligned x w y i
  metadata : Metadata w i indices
  staticTile : StaticTile w i

def pythonChecks (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) : List Bool :=
  [decide (Layouts x w y i),decide (DTypes x w y i),decide (Ranks x w y i),decide (Singleton w),
   decide (Devices x w y i),decide (Dense x ∧ Dense w ∧ Dense y ∧ Dense i),
   decide (s.isFloat=Bool.true ∧ s.finite=Bool.true),decide (Shapes x w y i),decide (Dimensions w y i),
   decide (Formed x w y),decide (Separate x w y i),decide (Aligned x w y i),decide (Metadata w i indices),decide (StaticTile w i)]
def Admitted (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) := (pythonChecks x w y i s indices).all id=Bool.true
theorem python_checks_iff (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) :
    Admitted x w y i s indices ↔ InputContract x w y i s indices := by
  simp only [Admitted,pythonChecks,List.all_cons,List.all_nil,Bool.and_eq_true,decide_eq_true_eq,id_eq,and_true]
  constructor
  · rintro ⟨a,b,c,d,e,f,g,h,j,k,l,m,n,o⟩;exact ⟨a,b,c,d,e,f,g,h,j,k,l,m,n,o⟩
  · intro h;exact ⟨h.layouts,h.dtypes,h.ranks,h.singleton,h.devices,h.dense,h.scale,h.shapes,h.dimensions,h.formed,h.separate,h.aligned,h.metadata,h.staticTile⟩

def storage (x : TensorInfo) : Storage := if x.dtype=.other "bfloat16" then .bf16 else .fp16
def call (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) (ok : Admitted x w y i s indices) : Call :=
  let h:=(python_checks_iff x w y i s indices).mp ok
  { batches := i.numel, adapters := rows w, N := width w, K := rank w, BN := block w, columns := columns y
    blockPositive := Nat.pow_pos (by decide)
    rowsCovered := (Nat.clog_le_iff_le_pow (by decide : 1 < 2)).mp (Nat.le_refl _)
    outputCovered := h.shapes.2.2.2
    storage := storage x, outputType := .fp32, scaling := s.value }

theorem selected_valid (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) (ok : Admitted x w y i s indices)
    (b n : Nat) (hb : b < i.numel) (hn : n < width w) :
    indices b=-1 ∨ (0 ≤ indices b ∧ (indices b).toNat < rows w) := by
  have h:=(python_checks_iff x w y i s indices).mp ok
  rcases h.metadata with empty|empty|valid
  · omega
  · change width w=0 at empty;omega
  · rcases valid b hb with skip|active
    · exact Or.inl skip
    · right
      change 0 ≤ indices b ∧ indices b < (rows w:Int) at active
      omega

#axiomsClean python_checks_iff
#axiomsClean selected_valid
#axiomsClean call

/-- The Python overlap predicate is the negation of separation for half-open
backing intervals, including empty storage. No allocation-identity law occurs. -/
theorem byte_checks_iff (a b : TensorInfo) :
    ByteSeparate a b ↔ ¬(a.storageDataPtr < a.storageDataPtr+a.storageNbytes ∧
      b.storageDataPtr < b.storageDataPtr+b.storageNbytes ∧
      a.storageDataPtr < b.storageDataPtr+b.storageNbytes ∧
      b.storageDataPtr < a.storageDataPtr+a.storageNbytes) := by
  unfold ByteSeparate
  omega

theorem launch_tile (ok : Admitted x w y i s indices) (batch : 0 < i.numel) (widthPositive : 0 < width w) :
    256*(call x w y i s indices ok).BN ≤ 2^20 := by
  have h := (python_checks_iff x w y i s indices).mp ok
  rcases h.staticTile with empty|empty|bounded
  · omega
  · omega
  · exact bounded

theorem launch_separate (ok : Admitted x w y i s indices) (batch : 0 < i.numel) (widthPositive : 0 < width w) :
    ByteSeparate y x ∧ ByteSeparate y w ∧ ByteSeparate y i := by
  have h := (python_checks_iff x w y i s indices).mp ok
  rcases h.separate with empty|empty|bounded
  · omega
  · omega
  · exact bounded

#axiomsClean byte_checks_iff
#axiomsClean launch_tile
#axiomsClean launch_separate
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Contract
