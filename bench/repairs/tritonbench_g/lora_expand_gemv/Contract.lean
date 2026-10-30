import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Geometry
import bench.repairs.tritonbench_g.bgmv_expand_slice.Contract

namespace VeriTile.Bench.Repairs.LoraExpandGemv.Contract
open VeriTile.Torch
noncomputable section
open Classical
set_option maxRecDepth 40000
set_option maxHeartbeats 8000000
abbrev TensorInfo := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.TensorInfo
abbrev rows := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rows
abbrev columns := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.columns
abbrev rank := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.rank
abbrev width := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.width
abbrev Dense := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Dense
abbrev Singleton := VeriTile.Bench.Repairs.BgmvExpandSlice.Contract.Singleton

structure Inputs where
  (x w y indices : TensorInfo)
structure Scalars where
  (addBool add : Bool)

def tensors (t : Inputs) := [t.x,t.w,t.y,t.indices]
def reads (t : Inputs) := [t.x,t.w,t.indices]
def block (t : Inputs) := TritonBenchG.LoraExpandGemv.Host.nextPowerOfTwo (rank t.w)
def geometry (t : Inputs) (s : Scalars) : Geometry where
  B := t.indices.numel
  L := rows t.w
  N := width t.w
  K := rank t.w
  xm := rank t.w
  xk := 1
  l0 := width t.w*rank t.w
  wn := rank t.w
  wk := 1
  ym := columns t.y
  yn := 1
  indexStart := 0
  indexStride := 1
  indexCapacity := t.indices.numel
  xCapacity := t.x.numel
  wCapacity := t.w.numel
  yCapacity := t.y.numel
  add := s.add

def Layouts (t : Inputs) : Prop := ∀ tensor∈tensors t,tensor.layout=.strided
def DTypes (t : Inputs) : Prop :=
  t.x.dtype=.floating .fp32 ∧ t.w.dtype=.other "float16" ∧ t.y.dtype=.floating .fp32 ∧
  (t.indices.dtype=.other "int32" ∨ t.indices.dtype=.other "int64")
def Ranks (t : Inputs) : Prop :=
  t.x.shape.length=2 ∧ t.y.shape.length=2 ∧ t.indices.shape.length=1 ∧ (t.w.shape.length=3 ∨ t.w.shape.length=4)
def Devices (t : Inputs) : Prop := ∀ tensor∈tensors t,tensor.device.kind="cuda" ∧ tensor.device=t.x.device
def Shapes (t : Inputs) : Prop :=
  0< rank t.w ∧ rank t.w≤4096 ∧ columns t.x=rank t.w ∧ t.indices.numel≤ rows t.x ∧
  t.indices.numel≤ rows t.y ∧ width t.w≤ columns t.y
def Limits (t : Inputs) : Prop :=
  t.indices.numel≤65535 ∧ rows t.w<2^31 ∧ width t.w<2^31 ∧ rank t.w<2^31 ∧ columns t.y<2^31 ∧
  width t.w*rank t.w<2^31 ∧ 2*width t.w<2^31 ∧
  t.indices.numel*rank t.w+block t<2^31 ∧
  rows t.w*(width t.w*rank t.w)+width t.w*rank t.w+256*rank t.w+block t<2^31 ∧
  t.indices.numel*columns t.y+width t.w+256<2^31
def Separate (t : Inputs) : Prop := ∀ tensor∈reads t,t.y.backing≠tensor.backing
def Aligned (t : Inputs) : Prop := t.indices.numel=0 ∨ width t.w=0 ∨
  ∀ tensor∈[t.x,t.w,t.y],tensor.numel=0 ∨ tensor.dataPtr.toNat%16=0
def Metadata (t : Inputs) (indices : Nat→Int) : Prop :=
  ∀ b,b< t.indices.numel → indices b=-1 ∨ (0≤ indices b ∧ indices b<(rows t.w:Int))

structure InputContract (t : Inputs) (s : Scalars) (indices : Nat→Int) : Prop where
  layouts : Layouts t
  dtypes : DTypes t
  ranks : Ranks t
  singleton : Singleton t.w
  devices : Devices t
  dense : ∀ tensor∈tensors t,Dense tensor
  scalarType : s.addBool=Bool.true
  shapes : Shapes t
  limits : Limits t
  separate : Separate t
  aligned : Aligned t
  metadata : Metadata t indices

def pythonChecks (t : Inputs) (s : Scalars) (indices : Nat→Int) : List Bool :=
  [decide (Layouts t),decide (DTypes t),decide (Ranks t),decide (Singleton t.w),decide (Devices t),
   decide (∀ tensor∈tensors t,Dense tensor),s.addBool,decide (Shapes t),decide (Limits t),
   decide (Separate t),decide (Aligned t),decide (Metadata t indices)]
def Admitted (t : Inputs) (s : Scalars) (indices : Nat→Int) := (pythonChecks t s indices).all id=Bool.true

theorem python_checks_iff (t : Inputs) (s : Scalars) (indices : Nat→Int) :
    Admitted t s indices ↔ InputContract t s indices := by
  simp only [Admitted,pythonChecks,List.all_cons,List.all_nil,Bool.and_eq_true,decide_eq_true_eq,id_eq,and_true]
  constructor
  · rintro ⟨a,b,c,d,e,f,g,h,i,j,k,l⟩;exact ⟨a,b,c,d,e,f,g,h,i,j,k,l⟩
  · intro h;exact ⟨h.layouts,h.dtypes,h.ranks,h.singleton,h.devices,h.dense,h.scalarType,h.shapes,h.limits,h.separate,h.aligned,h.metadata⟩

/-- Finite verification of the installed six-shift helper on its admitted
native tile range. No idealized unbounded power-of-two helper is substituted. -/
theorem power_profile : ∀ k : Fin 4097,0< k.val →
    k.val≤ TritonBenchG.LoraExpandGemv.Host.nextPowerOfTwo k.val ∧
    TritonBenchG.LoraExpandGemv.Host.nextPowerOfTwo k.val≤4096 ∧
    (k.val%TritonBenchG.LoraExpandGemv.Host.nextPowerOfTwo k.val=0 →
      TritonBenchG.LoraExpandGemv.Host.nextPowerOfTwo k.val=k.val) := by
  decide +kernel

#axiomsClean python_checks_iff
#axiomsClean power_profile
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Contract
