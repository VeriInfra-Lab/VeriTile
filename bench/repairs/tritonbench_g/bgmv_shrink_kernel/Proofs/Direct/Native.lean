import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Contract
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Direct.NativeGraph
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Direct.FPWorkset
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.Physical

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Native
open VeriTile Triton FP.Structural Contract NativeGraph
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
namespace Shared
abbrev TensorWindow := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.TensorWindow
abbrev indexBytes := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.indexBytes
abbrev half := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.half
abbrev index := VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Native.index
end Shared
namespace Raw
abbrev State := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.Memory
abbrev Word := BitVec 32
abbrev Byte := BitVec 8
abbrev read := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.read
abbrev unpack := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.unpack
abbrev ops := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.ops
abbrev byte_of_read := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.byte_of_read
end Raw

structure Binding (x w y i : TensorInfo) where
  input : Shared.TensorWindow x 2
  weights : Shared.TensorWindow w 2
  output : Shared.TensorWindow y 4
  indices : Shared.TensorWindow i (Shared.indexBytes i)
  squeezed : TensorInfo
  squeezeShape : squeezed.shape=[rows w,width w,rank w]
  squeezePointer : squeezed.dataPtr=w.dataPtr
  squeezeBacking : squeezed.backing=w.backing
  inputSeparate : x.backing≠y.backing → input.backing.separate output.backing
  weightSeparate : w.backing≠y.backing → weights.backing.separate output.backing
  indexSeparate : i.backing≠y.backing → indices.backing.separate output.backing

def data (a : Binding x w y i) (memory : Raw.State) : FPValues.Data NG.Term where
  input := fun off => .leaf "X" off (Shared.half memory a.input.window.owner (a.input.window.baseByte+2*off))
  weights := fun off => .leaf "W" off (Shared.half memory a.weights.window.owner (a.weights.window.baseByte+2*off))
  output := fun off => .leaf "Y" off (Raw.read memory (a.output.window.key off))
  indices := Shared.index memory i a.indices.window
  other := fun _ _ => .mk .real (.bits 0)
  padding := fun dtype _ _ => defaultValue dtype

abbrev Cell (h : Call) := Fin h.batches × Fin h.N
def address (h : Call) (c : Cell h) := Geometry.outputOffset h c.1.val c.2.val
def cells (h : Call) := (List.finRange h.batches).product (List.finRange h.N)
def jobs (h : Call) (indices : Nat → Int) := (cells h).filter (fun c => decide (indices c.1.val≠-1))
theorem cells_complete (h : Call) (c : Cell h) : c∈cells h := by rcases c with ⟨b,n⟩;simp [cells]
theorem address_injective (h : Call) : Function.Injective (address h) := by
  intro p q same
  have eq:=Geometry.output_injective h p.1.val q.1.val p.2 q.2 same
  exact Prod.ext (Fin.ext eq.1) eq.2

abbrev Run (plans : ScalarTree.Plans) (h : Call) (orders : Workset.Orders h)
    (a : Binding x w y i) (before : Raw.State) :=
  FPWorkset.Run algebra (primitives plans) h orders (data a before)
def reference (plans : ScalarTree.Plans) (S : Selection) (I : Instructions)
    (h : Call) (orders : Workset.Orders h) (a : Binding x w y i) (before : Raw.State) (c : Cell h) :=
  eval I S (FPValues.target algebra (primitives plans) h orders (data a before) c.1 c.2)
def outputWord (binding : CompilerBinding plans S) (I : Instructions)
    (h : Call) (orders : Workset.Orders h) (a : Binding x w y i) (before : Raw.State)
    (run : Run plans h orders a before) (c : Cell h) : Raw.Word :=
  (binding.program (run.output c.1 c.2)).eval I

/-- Physical interpretation of the completed local atomic traces at the actual
supplied output keys. This is direct storage, not a tensor copyback stage. -/
def sourceBytes (binding : CompilerBinding plans S) (I : Instructions)
    (h : Call) (orders : Workset.Orders h) (a : Binding x w y i) (before : Raw.State)
    (run : Run plans h orders a before) : Raw.State :=
  stores Raw.ops (fun c => a.output.window.key (address h c))
    (outputWord binding I h orders a before run) (jobs h (data a before).indices) before

theorem source_read (binding : CompilerBinding plans S) (I : Instructions)
    (h : Call) (orders : Workset.Orders h) (a : Binding x w y i) (before : Raw.State)
    (run : Run plans h orders a before) (c : Cell h) (active : (data a before).indices c.1.val≠-1) :
    Raw.read (sourceBytes binding I h orders a before run) (a.output.window.key (address h c))=
      outputWord binding I h orders a before run c := by
  apply readback Raw.ops _ _ _ before c (List.mem_filter.mpr ⟨cells_complete h c,by simpa using active⟩)
  · intro q _ same
    have offsets:=congrArg Key.address same
    simp only [Allocation.key] at offsets
    have equal : address h q=address h c := by omega
    rw [address_injective h equal]
  · intro q _ different
    apply Allocation.indices_disjoint
    intro equal
    exact different (congrArg a.output.window.key equal.symm)

theorem source_frame (binding : CompilerBinding plans S) (I : Instructions)
    (h : Call) (orders : Workset.Orders h) (a : Binding x w y i) (before : Raw.State)
    (run : Run plans h orders a before) (owner ptr : Nat)
    (outside : ∀ c : Cell h,(data a before).indices c.1.val≠-1 → ¬(a.output.window.key (address h c)).covers owner ptr) :
    sourceBytes binding I h orders a before run owner ptr=before owner ptr := by
  apply bytes_frame Raw.ops
  intro c member
  exact outside c (by simpa using (List.mem_filter.mp member).2)

structure Result (y : TensorInfo) where
  tensor : TensorInfo
  same : tensor=y
  returned : Unit
  bytes : Raw.State
def invoke (binding : CompilerBinding plans S) (I : Instructions)
    (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) (ok : Admitted x w y i s indices)
    (orders : Workset.Orders (call x w y i s indices ok)) (a : Binding x w y i) (before : Raw.State)
    (run : Run plans (call x w y i s indices ok) orders a before) : Result y :=
  ⟨y,rfl,(),sourceBytes binding I (call x w y i s indices ok) orders a before run⟩

theorem checked_source (plans : ScalarTree.Plans) (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (a : Binding x w y i) (before : Raw.State)
    (metadata : ∀ b,b < i.numel → (data a before).indices b=indices b) :
    FPWorkset.SourceWorkset algebra (primitives plans) (call x w y i s indices ok) (data a before) := by
  intro b n split active current
  have active' : indices b.val≠-1 := by rw [metadata b.val b.isLt] at active;exact active
  have valid:=(selected_valid x w y i s indices ok b.val n.val b.isLt n.isLt).resolve_left active'
  exact FPWorkset.source_event algebra (primitives plans) _ (data a before) b n split
    (by rw [metadata b.val b.isLt];exact valid.1) active current

theorem whole (plans : ScalarTree.Plans) (S : Selection) (I : Instructions)
    (binding : CompilerBinding plans S) (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int)
    (ok : Admitted x w y i s indices) (orders : Workset.Orders (call x w y i s indices ok))
    (a : Binding x w y i) (before : Raw.State) (run : Run plans (call x w y i s indices ok) orders a before)
    (metadata : ∀ b,b < i.numel → (data a before).indices b=indices b) :
    let h:=call x w y i s indices ok
    let result:=invoke binding I x w y i s indices ok orders a before run
    result.tensor=y ∧ result.returned=() ∧ FPWorkset.SourceWorkset algebra (primitives plans) h (data a before) ∧
    (∀ c : Cell h,indices c.1.val≠-1 → ∀ byte : Fin 4,
      result.bytes a.output.window.owner ((a.output.window.key (address h c)).address+byte.val)=
        Raw.unpack (reference plans S I h orders a before c) byte) ∧
    (∀ owner ptr,(∀ c : Cell h,indices c.1.val≠-1 → ¬(a.output.window.key (address h c)).covers owner ptr) →
      result.bytes owner ptr=before owner ptr) := by
  dsimp only
  let h:=call x w y i s indices ok
  refine ⟨rfl,rfl,checked_source plans x w y i s indices ok a before metadata,?_,?_⟩
  · intro c active byte
    have hp : (data a before).indices c.1.val≠-1 := by rw [metadata c.1.val c.1.isLt];exact active
    change sourceBytes binding I h orders a before run a.output.window.owner
      ((a.output.window.key (address h c)).address+byte.val)=_
    calc
      _=Raw.unpack (Raw.read (sourceBytes binding I h orders a before run) (a.output.window.key (address h c))) byte :=
        Raw.byte_of_read _ _ byte
      _=_ := by
        rw [source_read binding I h orders a before run c hp]
        unfold outputWord
        rw [compiled_expression,FPWorkset.readback]
        rfl
  · intro owner ptr outside
    apply source_frame
    intro c active
    apply outside c
    rw [metadata c.1.val c.1.isLt] at active
    exact active

#axiomsClean address_injective
#axiomsClean source_read
#axiomsClean source_frame
#axiomsClean checked_source
#axiomsClean whole
#specNonCircular reference avoiding [FPWorkset.SourceWorkset,Kernels.source,FP.Structural.exec,sourceBytes]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Native
