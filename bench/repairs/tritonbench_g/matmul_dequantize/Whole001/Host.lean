import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.FirstGrid
import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.SecondGrid
import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.DequantGrid16
import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.DequantGrid32
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 4000000
namespace First
abbrev Parameters := VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters
end First
namespace D16
abbrev Parameters := VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.Parameters
end D16
inductive Route where
  | gptq | s2 | dequant32 | dequant16 | s1
  deriving DecidableEq
inductive FirstConfig where
  | m64n256 | m128n128
  deriving DecidableEq
def FirstConfig.bm : FirstConfig → Nat | .m64n256 => 64 | .m128n128 => 128
def FirstConfig.bn : FirstConfig → Nat | .m64n256 => 256 | .m128n128 => 128
/-- Exactly the source autotuner candidate family. -/
def firstCandidates : List FirstConfig := [.m64n256,.m128n128]
theorem first_exhaustive (c : FirstConfig) : c∈firstCandidates := by cases c <;> simp [firstCandidates]
inductive StorageDType where
  | fp16 | fp32 | int32
  deriving DecidableEq
structure Tensor where
  region : RegionName
  (rows columns rowStride columnStride storageOffset allocation base device identity : Nat)
  dtype : StorageDType
  resolved : Bool
/-- Dense resolved storage, with a real data-pointer offset into a finite allocation. -/
def Tensor.Valid (t : Tensor) (rows columns : Nat) (dtype : StorageDType) : Prop :=
  t.rows=rows ∧ t.columns=columns ∧ t.dtype=dtype ∧ t.resolved=Bool.true ∧
  t.rowStride=columns ∧ t.columnStride=1 ∧ t.storageOffset+rows*columns≤t.allocation
structure Arguments where
  (M N K group : Nat)
  (a q s z c d : Tensor)
  supplied : Bool

def first (a : Arguments) (c : FirstConfig) : First.Parameters :=
  ⟨a.a.region,a.c.region,a.s.region,a.q.region,a.z.region,a.M,a.N,a.K,
   a.a.rowStride,a.a.columnStride,a.q.rowStride,a.q.columnStride,a.c.rowStride,a.c.columnStride,
   a.s.rowStride,a.s.columnStride,a.z.rowStride,a.z.columnStride,a.group,c.bm,c.bn,32,8,decide (a.group=a.K)⟩
def second (a : Arguments) : FPTypedSecond.Parameters :=
  ⟨a.a.region,a.c.region,a.s.region,a.q.region,a.z.region,a.M,a.N,a.K,
   a.a.rowStride,a.a.columnStride,a.q.rowStride,a.q.columnStride,a.c.rowStride,a.c.columnStride,
   a.s.rowStride,a.s.columnStride,a.z.rowStride,a.z.columnStride,a.group,64,128,32,8,(a.K+31)/32⟩
def dequant16 (a : Arguments) : D16.Parameters :=
  ⟨a.q.region,a.s.region,a.z.region,a.d.region,a.K,a.N,a.group,
   a.q.rowStride,a.q.columnStride,a.s.rowStride,a.s.columnStride,a.z.rowStride,a.z.columnStride,
   a.d.rowStride,a.d.columnStride,64,32⟩
def dequant32 (a : Arguments) : Dequant32.Parameters :=
  ⟨a.q.region,a.s.region,a.z.region,a.d.region,a.K,a.N,a.group,
   a.q.rowStride,a.q.columnStride,a.s.rowStride,a.s.columnStride,a.z.rowStride,a.z.columnStride,
   a.d.rowStride,a.d.columnStride,64,32⟩
def tensors (a : Arguments) : List Tensor := [a.a,a.q,a.s,a.z,a.c,a.d]
def bytes (t : Tensor) : Nat := if t.dtype=.fp16 then 2 else 4
/-- Tensor/API preconditions only. No computed result or successful kernel execution is assumed. -/
structure Admissible (a : Arguments) (route : Route) : Prop where
  positive : 0<a.M ∧ 0<a.N ∧ 0<a.K ∧ 0<a.group
  packed : a.K%8=0 ∧ a.N%8=0 ∧ a.K%a.group=0
  av : a.a.Valid a.M a.K (if route=.s2 then .fp32 else .fp16)
  qv : a.q.Valid (a.K/8) a.N .int32
  sv : a.s.Valid (a.K/a.group) a.N (if route=.s2 ∨ route=.dequant32 then .fp32 else .fp16)
  zv : a.z.Valid (a.K/a.group) (a.N/8) .int32
  cv : a.c.Valid a.M a.N .fp16
  dv : a.d.Valid a.K a.N (if route=.dequant32 then .fp32 else .fp16)
  packedDtype : a.q.dtype=.int32 ∧ a.z.dtype=.int32
  separate : (tensors a).Pairwise (fun x y => x.region≠y.region)
  devices : ∀ t∈tensors a, t.device=a.a.device
  dataPointers : ∀ t∈tensors a, t.base+bytes t*t.allocation<2^64
  gptqK : route=.gptq → a.K%32=0 ∧ a.group%32=0
  gptqN : route=.gptq → a.N%256=0

/-- Packed tensors use signed int32 storage and structural integer cells. -/
def Packed (a : Arguments) {α : Type} (s : State α) : Prop :=
  (∀ o<a.q.allocation-a.q.storageOffset, ∃ w : BitVec 32, s.mem a.q.region o=.mk .int w.toInt) ∧
  (∀ o<a.z.allocation-a.z.storageOffset, ∃ w : BitVec 32, s.mem a.z.region o=.mk .int w.toInt)

def returned (route : Route) (a : Arguments) : Option Nat :=
  if route=.gptq ∧ a.supplied then none
  else if route=.dequant16 ∨ route=.dequant32 then some a.d.identity else some a.c.identity
/-- Allocation/reset behavior is an ordinary local host contract; output is
separate and every logical output is overwritten by SPLIT_K=1. -/
def initialization (route : Route) (a : Arguments) : String :=
  if route=.s2 then "zeros/default; autotuner reset_to_zero c_ptr"
  else if route=.dequant16 ∨ route=.dequant32 then "ones"
  else if a.supplied then "supplied" else "empty"
def launch (a : Arguments) (route : Route) (c : FirstConfig) : List Nat :=
  if route=.gptq then [Geometry.count (FirstGrid.geometry (first a c))]
  else if route=.s2 then [Geometry.count (SecondGrid.geometry (second a)),1]
  else [(a.K+63)/64,(a.N+31)/32]

theorem finite_address (t : Tensor) (rows columns : Nat) (dtype : StorageDType)
    (h : t.Valid rows columns dtype) (m n : Nat) (hm : m<rows) (hn : n<columns) :
    t.storageOffset+m*t.rowStride+n*t.columnStride<t.allocation := by
  rcases h with ⟨_,_,_,_,hr,hc,hroom⟩
  rw [hr,hc,Nat.mul_one]
  have hmn : m*columns+n<rows*columns := by
    have h := Nat.mul_le_mul_right columns (show m+1≤rows by omega)
    rw [Nat.add_mul] at h
    omega
  omega

theorem dense_geometry (g : Geometry.Layout) (hr : g.rowStride=g.columns) (hc : g.columnStride=1)
    (hbm : 0<g.blockRows) (hbn : 0<g.blockColumns) (hgm : 0<g.groupRows) : Geometry.Valid g := by
  refine ⟨hbm,hbn,hgm,?_⟩
  intro x y he
  change x.1.val*g.rowStride+x.2.val*g.columnStride=y.1.val*g.rowStride+y.2.val*g.columnStride at he
  rw [hr,hc,Nat.mul_one,Nat.mul_one] at he
  have hx := x.2.isLt
  have hy := y.2.isLt
  have hp : 0<g.columns := by omega
  have hd := congrArg (fun n => n/g.columns) he
  have hm : x.1.val=y.1.val := by
    dsimp only at hd
    rw [Nat.mul_comm x.1.val g.columns,Nat.mul_comm y.1.val g.columns,
      Nat.mul_add_div hp,Nat.mul_add_div hp,Nat.div_eq_of_lt hx,Nat.div_eq_of_lt hy] at hd
    simpa using hd
  have hn : x.2.val=y.2.val := by rw [hm] at he; omega
  exact Prod.ext (Fin.ext hm) (Fin.ext hn)

theorem first_valid (a : Arguments) (c : FirstConfig) (h : Admissible a route) :
    Geometry.Valid (FirstGrid.geometry (first a c)) := by
  apply dense_geometry _ h.cv.2.2.2.2.1 h.cv.2.2.2.2.2.1
  · cases c <;> norm_num [FirstGrid.geometry,first,FirstConfig.bm,FirstConfig.bn]
  · cases c <;> norm_num [FirstGrid.geometry,first,FirstConfig.bm,FirstConfig.bn]
  · norm_num [FirstGrid.geometry,first]

theorem second_valid (a : Arguments) (h : Admissible a route) :
    Geometry.Valid (SecondGrid.geometry (second a)) :=
  dense_geometry _ h.cv.2.2.2.2.1 h.cv.2.2.2.2.2.1 (by norm_num [SecondGrid.geometry,DequantGrid16.geometry,DequantGrid32.geometry,second,dequant16,dequant32]) (by norm_num [SecondGrid.geometry,DequantGrid16.geometry,DequantGrid32.geometry,second,dequant16,dequant32]) (by norm_num [SecondGrid.geometry,DequantGrid16.geometry,DequantGrid32.geometry,second,dequant16,dequant32])
theorem d16_valid (a : Arguments) (h : Admissible a route) :
    Geometry.Valid (DequantGrid16.geometry (dequant16 a)) :=
  dense_geometry _ h.dv.2.2.2.2.1 h.dv.2.2.2.2.2.1 (by norm_num [SecondGrid.geometry,DequantGrid16.geometry,DequantGrid32.geometry,second,dequant16,dequant32]) (by norm_num [SecondGrid.geometry,DequantGrid16.geometry,DequantGrid32.geometry,second,dequant16,dequant32]) (by norm_num [SecondGrid.geometry,DequantGrid16.geometry,DequantGrid32.geometry,second,dequant16,dequant32])
theorem d32_valid (a : Arguments) (h : Admissible a route) :
    Geometry.Valid (DequantGrid32.geometry (dequant32 a)) :=
  dense_geometry _ h.dv.2.2.2.2.1 h.dv.2.2.2.2.2.1 (by norm_num [SecondGrid.geometry,DequantGrid16.geometry,DequantGrid32.geometry,second,dequant16,dequant32]) (by norm_num [SecondGrid.geometry,DequantGrid16.geometry,DequantGrid32.geometry,second,dequant16,dequant32]) (by norm_num [SecondGrid.geometry,DequantGrid16.geometry,DequantGrid32.geometry,second,dequant16,dequant32])

theorem distinct (a : Arguments) (h : Admissible a route) :
    a.a.region≠a.c.region ∧ a.q.region≠a.c.region ∧ a.s.region≠a.c.region ∧ a.z.region≠a.c.region ∧
    a.q.region≠a.d.region ∧ a.s.region≠a.d.region ∧ a.z.region≠a.d.region := by
  have hh := h.separate
  simp only [tensors,List.pairwise_cons,List.mem_cons,List.mem_singleton,forall_eq_or_imp] at hh
  tauto

#axiomsClean first_exhaustive
#axiomsClean finite_address
#axiomsClean first_valid
#axiomsClean distinct
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Host
