import bench.tritonbench_g.chunk_gla_simple.Sources.DirectReal
import bench.tritonbench_g.chunk_gla_simple.Sources.ScalarGraph
import bench.tritonbench_g.chunk_gla_simple.Sources.Admission
import bench.tritonbench_g.chunk_gla_simple.Sources.NativeABI
import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPStorage

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.PhysicalDirect
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math Real Host
open VeriTile.Bench.Repairs.ChunkGlaFwd
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions (Graph ISA)
noncomputable section
set_option maxHeartbeats 8000000
variable {α : Type}

/-- Physical FP32 word bases. Only the new output needs disjoint storage;
read-only inputs may alias each other, including overlapping offset views. -/
structure Layout (c : Shape) where
  base : RegionName → Nat
  objectId : Nat
  outputDevice : Nat
  outputShape : List Nat
  shape : outputShape=[c.B,c.H,c.T,c.V]
  fresh : ∀ r∈(["q","k","v","h","g"] : List RegionName),
    base "o"+capacity c "o"≤base r ∨ base r+capacity c r≤base "o"

def output (A : Layout c) : FPStorage.Allocation := ⟨A.base "o",capacity c "o"⟩
def commit (A : Layout c) (values : Nat → α) (before : Nat → α) : Nat → α :=
  FPStorage.flush (output A) values before

def realInput (A : Layout c) (before : Nat → MemCell) (scratch : BlockState) : BlockState :=
  {scratch with mem:=fun r i=>before (A.base r+i)}
def fpInput (A : Layout c) (before : Nat → BitVec 32) (scratch : State NativeGraph.Expr) : State NativeGraph.Expr :=
  {scratch with mem:=fun r i=>.mk .real (.leaf (before (A.base r+i)))}

/-- FP32-profile source input pointers read the DSL real view. This equation
binds every load to the caller's physical word, including aliased inputs. -/
theorem fpInput_read (A : Layout c) (before : Nat → BitVec 32)
    (scratch : State NativeGraph.Expr) (r : RegionName) (i : Nat) :
    ((fpInput A before scratch).mem r i).read .real=.leaf (before (A.base r+i)) := by
  exact Cell.read_same _ _

theorem readback (A : Layout c) (values before : Nat → α) (i : Nat) (hi : i<capacity c "o") :
    commit A values before (A.base "o"+i)=values i := by
  letI : Inhabited α := ⟨before 0⟩
  exact FPStorage.flush_readback _ _ _ i hi

theorem frame (A : Layout c) (values before : Nat → α) (p : Nat) (miss : ¬(output A).Contains p) :
    commit A values before p=before p := by
  letI : Inhabited α := ⟨before 0⟩
  exact FPStorage.flush_frame _ _ _ p miss

theorem inputs (A : Layout c) (values before : Nat → α) (r : RegionName)
    (hr : r∈(["q","k","v","h","g"] : List RegionName)) (i : Nat) (hi : i<capacity c r) :
    commit A values before (A.base r+i)=before (A.base r+i) := by
  letI : Inhabited α := ⟨before 0⟩
  exact FPStorage.flush_preserves_input (output A) ⟨A.base r,capacity c r⟩ (A.fresh r hr) values before i hi

theorem tensor_bound (c : Shape) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    h.val*(c.T*c.V)+t.val*c.V+v.val<capacity c "o" := by
  simpa [capacity] using triple_lt h.isLt t.isLt v.isLt

def readReal (cell : MemCell) : ℝ :=
  match cell.readAs .real with | some (some value)=>value | _=>0

#axiomsClean fpInput_read
#axiomsClean inputs
end
end VeriTile.Bench.Repairs.ChunkGlaSimple.PhysicalDirect
