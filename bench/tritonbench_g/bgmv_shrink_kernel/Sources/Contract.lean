import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Kernels
import bench.tritonbench_g.bgmv_expand_slice.Sources.Contract

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Contract
open VeriTile.Torch
noncomputable section
open Classical
abbrev TensorInfo := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.TensorInfo
abbrev rows := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rows
abbrev columns := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.columns
abbrev rank := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rank
abbrev width := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.width
abbrev Layouts := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Layouts
abbrev Ranks := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Ranks
abbrev Singleton := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Singleton
abbrev Devices := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Devices
abbrev Dense := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Dense
abbrev Dimensions := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Dimensions
abbrev Separate := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Separate
abbrev Aligned := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Aligned
abbrev Metadata := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Metadata

structure Scale where
  value : ℝ
  isFloat : Bool
  finite : Bool
def block (w : TensorInfo) := 2^(Nat.clog 2 (width w))
def formedK (w : TensorInfo) := max 1 ((rank w+16383)/16384)*16384
def DTypes (x w y i : TensorInfo) : Prop :=
  (x.dtype=.other "float16" ∨ x.dtype=.other "bfloat16") ∧ w.dtype=x.dtype ∧
  y.dtype=.floating .fp32 ∧ (i.dtype=.other "int32" ∨ i.dtype=.other "int64")
def Shapes (x w y i : TensorInfo) : Prop :=
  columns x=rank w ∧ i.numel≤rows x ∧ i.numel≤rows y ∧ width w≤columns y
def Formed (x w y : TensorInfo) : Prop :=
  2*(x.numel+formedK w)<2^63 ∧ 2*(w.numel+block w*rank w+formedK w)<2^63 ∧
  4*(y.numel+block w)<2^63
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

def pythonChecks (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) : List Bool :=
  [decide (Layouts x w y i),decide (DTypes x w y i),decide (Ranks x w y i),decide (Singleton w),
   decide (Devices x w y i),decide (Dense x ∧ Dense w ∧ Dense y ∧ Dense i),
   decide (s.isFloat=Bool.true ∧ s.finite=Bool.true),decide (Shapes x w y i),decide (Dimensions w y i),
   decide (Formed x w y),decide (Separate x w y i),decide (Aligned x w y i),decide (Metadata w i indices)]
def Admitted (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) := (pythonChecks x w y i s indices).all id=Bool.true
theorem python_checks_iff (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) :
    Admitted x w y i s indices ↔ InputContract x w y i s indices := by
  simp only [Admitted,pythonChecks,List.all_cons,List.all_nil,Bool.and_eq_true,decide_eq_true_eq,id_eq,and_true]
  constructor
  · rintro ⟨a,b,c,d,e,f,g,h,j,k,l,m,n⟩;exact ⟨a,b,c,d,e,f,g,h,j,k,l,m,n⟩
  · intro h;exact ⟨h.layouts,h.dtypes,h.ranks,h.singleton,h.devices,h.dense,h.scale,h.shapes,h.dimensions,h.formed,h.separate,h.aligned,h.metadata⟩

def storage (x : TensorInfo) : Storage := if x.dtype=.other "bfloat16" then .bf16 else .fp16
def call (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) (ok : Admitted x w y i s indices) : Direct.Call :=
  let h:=(python_checks_iff x w y i s indices).mp ok
  { batches := i.numel, adapters := rows w, N := width w, K := rank w, BN := block w, columns := columns y
    blockPositive := Nat.pow_pos (by decide)
    rowsCovered := (Nat.clog_le_iff_le_pow (by decide : 1<2)).mp (Nat.le_refl _)
    outputCovered := h.shapes.2.2.2
    storage := storage x, outputType := .fp32, scaling := s.value }

theorem selected_valid (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) (ok : Admitted x w y i s indices)
    (b n : Nat) (hb : b < i.numel) (hn : n<width w) :
    indices b=-1 ∨ (0≤indices b ∧ (indices b).toNat<rows w) := by
  have h:=(python_checks_iff x w y i s indices).mp ok
  rcases h.metadata with empty|empty|valid
  · omega
  · change width w=0 at empty;omega
  · rcases valid b hb with skip|active
    · exact Or.inl skip
    · right
      change 0≤indices b ∧ indices b<(rows w:Int) at active
      omega

#axiomsClean python_checks_iff
#axiomsClean selected_valid
#axiomsClean call
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Contract
