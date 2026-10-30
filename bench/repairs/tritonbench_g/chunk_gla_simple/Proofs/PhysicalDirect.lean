import bench.repairs.tritonbench_g.chunk_gla_simple.Proofs.DirectReal
import bench.repairs.tritonbench_g.chunk_gla_simple.Proofs.ScalarGraph
import bench.repairs.tritonbench_g.chunk_gla_simple.Proofs.Admission
import bench.repairs.tritonbench_g.chunk_gla_simple.Proofs.NativeABI
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.FPStorage

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

def realResult (c : Shape) (A : Layout c) (scale : ℝ) (before : Nat → MemCell) (scratch : BlockState) :=
  commit A ((DirectReal.result c scale (realInput A before scratch)).mem "o") before

/-- Complete Real source, every actual returned tensor cell and caller frame.
The projection reads the original physical input allocations directly. -/
theorem real_complete (c : Shape) (bt : 0<c.BT) (A : Layout c) (scale : ℝ)
    (before : Nat → MemCell) (scratch : BlockState) :
    Nonempty (Kernel.GridLaunchedOrdinary (DirectReal.kernel c scale) (grid c)
      (realInput A before scratch) (DirectReal.result c scale (realInput A before scratch))) ∧
    (∀ h t v, readReal (realResult c A scale before scratch (A.base "o"+(h.val*(c.T*c.V)+t.val*c.V+v.val)))=
      Host.formula c bt (Real.inputs regions (realInput A before scratch)) scale h t v) ∧
    (∀ p,¬(output A).Contains p → realResult c A scale before scratch p=before p) ∧
    (∀ r∈(["q","k","v","h","g"] : List RegionName),∀ i,i<capacity c r →
      realResult c A scale before scratch (A.base r+i)=before (A.base r+i)) := by
  refine ⟨DirectReal.source c scale _,?_,frame A _ before,inputs A _ before⟩
  intro h t v
  unfold realResult
  rw [readback A _ before _ (tensor_bound c h t v)]
  have value:=DirectReal.readback c bt scale (realInput A before scratch) h t v
  exact value

/-- Source execution, native i64 masks/addresses and the scalar native graph
all refer to the same direct input projection and fresh returned allocation. -/
theorem native_complete (I : ISA (BitVec 32)) (c : Shape) (bt : 0<c.BT)
    (setup : NativeControl.Accepted c) (A : Layout c) (scale : BitVec 32)
    (before : Nat → BitVec 32) (scratch : State NativeGraph.Expr) :
    ∃ source, ∃ final : Nat → BitVec 32,
      FPGrid.run ScalarGraph.algebra c ScalarGraph.scaleTag (FPGrid.programs c)
        (fpInput A before scratch)=some source ∧
      source.mem=(NativeABI.grid ScalarGraph.algebra c (ScalarGraph.dots c) setup
        ScalarGraph.scaleTag (FPGrid.programs c) (fpInput A before scratch)).mem ∧
      (∀ h t v,final (A.base "o"+(h.val*(c.T*c.V)+t.val*c.V+v.val))=
        (ScalarGraph.value c (fpInput A before scratch) scale h t v bt).eval I) ∧
      (∀ p,¬(output A).Contains p → final p=before p) ∧
      (∀ r∈(["q","k","v","h","g"] : List RegionName),∀ i,i<capacity c r →
        final (A.base r+i)=before (A.base r+i)) := by
  obtain ⟨source,exec,native,values⟩:=NativeABI.whole_grid ScalarGraph.algebra c (ScalarGraph.dots c)
    setup ScalarGraph.scaleTag (fpInput A before scratch)
  let result:=fun off=>(ScalarGraph.lower scale ((source.mem "o" off).read .fp32)).eval I
  refine ⟨source,commit A result before,exec,native,?_,frame A result before,inputs A result before⟩
  intro h t v
  rw [readback A result before _ (tensor_bound c h t v)]
  dsimp only [result]
  rw [FPGrid.returned ScalarGraph.algebra c bt ScalarGraph.scaleTag (ScalarGraph.dots c)
    (fpInput A before scratch) source values h t v]
  simp only [Cell.read_same,ScalarGraph.value]


/-- Chosen source state is uniquely fixed by execution; it is not supplied by
an allocator, compiler setup result, or numerical interface. -/
def graphSource (c : Shape) (s : State NativeGraph.Expr) : State NativeGraph.Expr :=
  (FPGrid.whole_grid ScalarGraph.algebra c ScalarGraph.scaleTag (ScalarGraph.dots c) s).choose

theorem graphSource_spec (c : Shape) (s : State NativeGraph.Expr) :
    FPGrid.run ScalarGraph.algebra c ScalarGraph.scaleTag (FPGrid.programs c) s=some (graphSource c s) ∧
    FPGrid.Result ScalarGraph.algebra c ScalarGraph.scaleTag (ScalarGraph.dots c) s (graphSource c s) :=
  (FPGrid.whole_grid ScalarGraph.algebra c ScalarGraph.scaleTag (ScalarGraph.dots c) s).choose_spec

def graphResult (c : Shape) (A : Layout c) (scale : BitVec 32)
    (before : Nat → BitVec 32) (scratch : State NativeGraph.Expr) : Nat → Graph :=
  commit A (fun i=>ScalarGraph.lower scale ((graphSource c (fpInput A before scratch)).mem "o" i |>.read .fp32))
    (fun p=>.bits (before p))

theorem graph_readback (c : Shape) (bt : 0<c.BT) (A : Layout c) (scale : BitVec 32)
    (before : Nat → BitVec 32) (scratch : State NativeGraph.Expr)
    (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    graphResult c A scale before scratch (A.base "o"+(h.val*(c.T*c.V)+t.val*c.V+v.val))=
      ScalarGraph.value c (fpInput A before scratch) scale h t v bt := by
  unfold graphResult
  rw [readback A _ _ _ (tensor_bound c h t v)]
  rw [FPGrid.returned ScalarGraph.algebra c bt ScalarGraph.scaleTag (ScalarGraph.dots c)
    (fpInput A before scratch) _ (graphSource_spec c _).2 h t v]
  simp only [Cell.read_same,ScalarGraph.value]

theorem graph_native (c : Shape) (accepted : NativeControl.Accepted c)
    (s : State NativeGraph.Expr) :
    (graphSource c s).mem=(NativeABI.grid ScalarGraph.algebra c (ScalarGraph.dots c) accepted
      ScalarGraph.scaleTag (FPGrid.programs c) s).mem := by
  obtain ⟨t,execution,native,values⟩:=NativeABI.whole_grid ScalarGraph.algebra c (ScalarGraph.dots c)
    accepted ScalarGraph.scaleTag s
  have same : graphSource c s=t := Option.some.inj ((graphSource_spec c s).1.symm.trans execution)
  rw [same]
  exact native

theorem graph_frame (c : Shape) (A : Layout c) (scale : BitVec 32)
    (before : Nat → BitVec 32) (scratch : State NativeGraph.Expr)
    (p : Nat) (outside : ¬(output A).Contains p) :
    graphResult c A scale before scratch p=.bits (before p) := frame A _ _ p outside

theorem graph_inputs (c : Shape) (A : Layout c) (scale : BitVec 32)
    (before : Nat → BitVec 32) (scratch : State NativeGraph.Expr)
    (r : RegionName) (hr : r∈(["q","k","v","h","g"] : List RegionName))
    (i : Nat) (hi : i<capacity c r) :
    graphResult c A scale before scratch (A.base r+i)=.bits (before (A.base r+i)) :=
  inputs A _ _ r hr i hi

#axiomsClean graph_readback
#axiomsClean graph_native

#axiomsClean fpInput_read
#axiomsClean inputs
#axiomsClean real_complete
#axiomsClean native_complete
end
end VeriTile.Bench.Repairs.ChunkGlaSimple.PhysicalDirect
