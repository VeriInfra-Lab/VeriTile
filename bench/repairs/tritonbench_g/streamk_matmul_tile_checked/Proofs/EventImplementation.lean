import VeriTile.Triton.Concurrent.EventSafety
import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.EventValues
import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.EventFrame
import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.PhysicalRefinement

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.EventIO
open Whole001 Source VeriTile.Triton Concurrent
noncomputable section
set_option maxHeartbeats 5000000

theorem safe_in_bounds (p : Parameters) (observe : V → O) (bs : Concurrent.Bounds)
    (h : (io p observe).InBounds bs) : TapeSafe bs (ClosedSource.completeTape p) := by
  obtain ⟨inputs,memory,words⟩ := input_bounds p observe bs h
  exact TapeSafe.mono _ _ ⟨inputs,memory,words⟩ _ (ClosedSource.complete_safe p)

theorem real_outputs (p : Parameters) (values : (realIO p).Inputs) (s : Concurrent.State ℝ)
    (h : (realIO p).InputBinding values s) :
    (realIO p).OutputBinding (realFormula p values)
      (Concurrent.run realArithmetic (ClosedSource.completeTape p) s) := by
  refine ⟨?_,?_⟩
  · intro port lane _
    have zero : port=(0 : Fin 1) := Fin.eq_zero port
    subst port
    simpa [realIO,io,cWindow,arrayWindow,Window.address,realFormula,outputIndex,realArithmetic]
      using real_values p values s h lane
  · intro port
    exact Fin.elim0 port

theorem native_outputs (reference : Native.Reference) (p : Parameters)
    (values : (nativeIO p).Inputs) (s : Concurrent.State Native.Word)
    (h : (nativeIO p).InputBinding values s) :
    (nativeIO p).OutputBinding (nativeFormula reference p values)
      (Concurrent.run (nativeArithmetic reference) (ClosedSource.completeTape p) s) := by
  refine ⟨?_,?_⟩
  · intro port lane _
    have zero : port=(0 : Fin 1) := Fin.eq_zero port
    subst port
    simpa [nativeIO,io,cWindow,arrayWindow,Window.address,nativeFormula,outputIndex,nativeArithmetic]
      using congrArg observeNative (native_values reference p values s h lane)
  · intro port
    exact Fin.elim0 port

theorem real_implements (p : Parameters) :
    (realIO p).Implements realArithmetic (realFormula p) := by
  intro values _ initial binding
  refine ⟨.tape (ClosedSource.completeTape p),?_,ClosedSource.completeTape p,
    Concurrent.run realArithmetic (ClosedSource.completeTape p) initial,
    tape_runs _ _ _,safe_in_bounds p id,real_outputs p values initial binding,
    complete_frame p id realArithmetic initial⟩
  change (ClosedSource.sourceTape p).map Program.tape=some _
  rw [ClosedSource.sourceTape_eq]
  rfl

theorem native_implements (reference : Native.Reference) (p : Parameters) :
    (nativeIO p).Implements (nativeArithmetic reference) (nativeFormula reference p) := by
  intro values _ initial binding
  refine ⟨.tape (ClosedSource.completeTape p),?_,ClosedSource.completeTape p,
    Concurrent.run (nativeArithmetic reference) (ClosedSource.completeTape p) initial,
    tape_runs _ _ _,safe_in_bounds p observeNative,native_outputs reference p values initial binding,
    complete_frame p observeNative (nativeArithmetic reference) initial⟩
  change (ClosedSource.sourceTape p).map Program.tape=some _
  rw [ClosedSource.sourceTape_eq]
  rfl

#axiomsClean safe_in_bounds
#axiomsClean real_implements
#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.EventIO
