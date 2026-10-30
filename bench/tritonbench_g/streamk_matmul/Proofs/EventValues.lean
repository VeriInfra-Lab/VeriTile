import bench.tritonbench_g.streamk_matmul.IO
import bench.tritonbench_g.streamk_matmul.Proofs.SourceReadsSafety
import bench.tritonbench_g.streamk_matmul.Proofs.KernelResults

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.EventIO
open Whole001 Source VeriTile.Triton Concurrent
noncomputable section
set_option maxHeartbeats 5000000

theorem index_cell (p : Parameters) (i : C p) : Returned.index p (cell p i)=i := by
  apply Fin.ext
  simpa only [Returned.index,cell,Nat.mul_comm] using Nat.div_add_mod i.val p.h.n

theorem bound_a (p : Parameters) (observe : V → O) (values : (io p observe).Inputs)
    (s : Concurrent.State V) (h : (io p observe).InputBinding values s) :
    ClosedSource.viewA p s=(logicalInputs p values).a := by
  funext i
  simpa [io,scalarInputs,aWindow,arrayWindow,Window.address,ClosedSource.viewA,logicalInputs]
    using h.1 ⟨0,by change 0<4; decide⟩ i rfl

theorem bound_b (p : Parameters) (observe : V → O) (values : (io p observe).Inputs)
    (s : Concurrent.State V) (h : (io p observe).InputBinding values s) :
    ClosedSource.viewB p s=(logicalInputs p values).b := by
  funext i
  simpa [io,scalarInputs,bWindow,arrayWindow,Window.address,ClosedSource.viewB,logicalInputs]
    using h.1 ⟨1,by change 1<4; decide⟩ i rfl

theorem real_values (p : Parameters) (values : (realIO p).Inputs) (s : Concurrent.State ℝ)
    (h : (realIO p).InputBinding values s) (i : C p) :
    (Concurrent.run (ClosedSource.backend Arithmetic.real) (ClosedSource.completeTape p) s).mem 0 i.val=
      LogicalIO.realFormula p (logicalInputs p values) (cell p i) := by
  have result := congrArg (fun t : Source.State ℝ p 0 => t.output i)
    (ClosedSource.complete_projection p Arithmetic.real 0 s)
  change _ = (Source.call p Arithmetic.real (ClosedSource.extract p 0 s)).output i at result
  have value := Returned.real_contract p (ClosedSource.extract p 0 s) (cell p i)
  change (Source.call p Arithmetic.real (ClosedSource.extract p 0 s)).output (Returned.index p (cell p i)) = _ at value
  rw [index_cell] at value
  exact result.trans (value.trans (by
    change Reference.matrixProduct p (ClosedSource.viewA p s) (ClosedSource.viewB p s) _ _ = _
    rw [bound_a p id values s h,bound_b p id values s h]
    rfl))

theorem native_values (reference : Native.Reference) (p : Parameters) (values : (nativeIO p).Inputs)
    (s : Concurrent.State Native.Word) (h : (nativeIO p).InputBinding values s) (i : C p) :
    (Concurrent.run (ClosedSource.backend (Native.operations (Native.referenceInstructions reference)))
      (ClosedSource.completeTape p) s).mem 0 i.val=
      LogicalIO.nativeFormula reference p (logicalInputs p values) (cell p i) := by
  let ops := Native.operations (Native.referenceInstructions reference)
  have result := congrArg (fun t : Source.State Native.Word p 0 => t.output i)
    (ClosedSource.complete_projection p ops 0 s)
  change _ = (Source.call p ops (ClosedSource.extract p 0 s)).output i at result
  have value := Returned.value_contract p ops (ClosedSource.extract p 0 s) (cell p i)
  change (Source.call p ops (ClosedSource.extract p 0 s)).output (Returned.index p (cell p i)) = _ at value
  rw [index_cell] at value
  exact result.trans (value.trans (by
    change Returned.value p ops (ClosedSource.viewA p s) (ClosedSource.viewB p s) _ = _
    rw [bound_a p observeNative values s h,bound_b p observeNative values s h]
    rfl))

#axiomsClean index_cell
#axiomsClean real_values
#axiomsClean native_values
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.EventIO
