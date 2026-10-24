import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.Real
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.Bounds
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.RealTensor
open VeriTile.Triton FP.Structural Contract
noncomputable section
open Classical
abbrev Cell (h : Call) := Fin h.batches × Fin h.N
def address (h : Call) (c : Cell h) := c.1.val*h.columns+c.2.val

theorem unique (h : Call) : Function.Injective (address h) := by
  intro p q same
  have eq := VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.output_injective
    h p.1.val q.1.val p.2 q.2 same
  exact Prod.ext (Fin.ext eq.1) eq.2

/-- Interpret completed per-address atomic traces in the supplied tensor.
The coordinate choice depends solely on geometry and the sentinel, never on
an expected value or a reference implementation. -/
def value (h : Call) (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders h)
    (u : BlockState) (indices : Nat → Int) (run : Real.Run h orders u indices) (off : Nat) : ℝ :=
  if hit : ∃ c : Cell h,indices c.1.val ≠ -1 ∧ off=address h c then
    run.output (Classical.choose hit).1 (Classical.choose hit).2
  else u.readMem "Y" off

theorem active_value (h : Call) (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders h)
    (u : BlockState) (indices : Nat → Int) (run : Real.Run h orders u indices)
    (c : Cell h) (active : indices c.1.val ≠ -1) :
    value h orders u indices run (address h c)=run.output c.1 c.2 := by
  have hit : ∃ d : Cell h,indices d.1.val ≠ -1 ∧ address h c=address h d := ⟨c,active,rfl⟩
  rw [value,dif_pos hit]
  have same : Classical.choose hit=c := unique h (Classical.choose_spec hit).2.symm
  rw [same]

def memory (h : Call) (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders h)
    (u : BlockState) (indices : Nat → Int) (run : Real.Run h orders u indices) (r : RegionName) (off : Nat) :
    FP.Structural.Cell ℝ :=
  if r="Y" then .mk h.outputType.toTileDType (ofFloat h.outputType (value h orders u indices run off))
  else (Real.data h u indices).cells h r off

structure Result (y : TensorInfo) where
  tensor : TensorInfo
  same : tensor=y
  returned : Unit
  cells : RegionName → Nat → FP.Structural.Cell ℝ

def invoke (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) (ok : Admitted x w y i s indices)
    (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders (call x w y i s indices ok))
    (u : BlockState) (run : Real.Run (call x w y i s indices ok) orders u indices) : Result y :=
  ⟨y,rfl,(),memory (call x w y i s indices ok) orders u indices run⟩

theorem returned_value (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) (ok : Admitted x w y i s indices)
    (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders (call x w y i s indices ok))
    (u : BlockState) (run : Real.Run (call x w y i s indices ok) orders u indices)
    (c : Cell (call x w y i s indices ok)) (active : indices c.1.val ≠ -1) :
    (invoke x w y i s indices ok orders u run).cells "Y" (address (call x w y i s indices ok) c)=
      .mk .fp32 (Target.real (call x w y i s indices ok) u indices c.1 c.2) := by
  simp only [invoke,memory,if_true,active_value _ _ _ _ _ c active,Real.readback]
  rfl

theorem frame (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) (ok : Admitted x w y i s indices)
    (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders (call x w y i s indices ok))
    (u : BlockState) (run : Real.Run (call x w y i s indices ok) orders u indices)
    (r : RegionName) (off : Nat)
    (outside : ¬(r="Y" ∧ ∃ c : Cell (call x w y i s indices ok),indices c.1.val ≠ -1 ∧ off=address _ c)) :
    (invoke x w y i s indices ok orders u run).cells r off=
      (Real.data (call x w y i s indices ok) u indices).cells (call x w y i s indices ok) r off := by
  by_cases output : r="Y"
  · subst r
    have noHit : ¬∃ c : Cell (call x w y i s indices ok),indices c.1.val ≠ -1 ∧ off=address _ c :=
      fun hit => outside ⟨rfl,hit⟩
    simp only [invoke,memory,if_true]
    rw [value,dif_neg noHit]
    simp [VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.Data.cells,call,Real.data,ofFloat]
  · simp [invoke,memory,output]

theorem all_values (x w y i : TensorInfo) (s : Scale) (indices : Nat → Int) (ok : Admitted x w y i s indices)
    (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders (call x w y i s indices ok))
    (u : BlockState) (run : Real.Run (call x w y i s indices ok) orders u indices)
    (c : Cell (call x w y i s indices ok)) :
    (invoke x w y i s indices ok orders u run).cells "Y" (address (call x w y i s indices ok) c)=
      .mk .fp32 (Target.real (call x w y i s indices ok) u indices c.1 c.2) := by
  by_cases active : indices c.1.val ≠ -1
  · exact returned_value x w y i s indices ok orders u run c active
  · have skipped : indices c.1.val=-1 := not_ne_iff.mp active
    rw [frame x w y i s indices ok orders u run]
    · simp [VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues.Data.cells,call,
        Real.data,Target.real,skipped,address,ofFloat]
    · rintro ⟨_,d,hd,equal⟩
      have same := unique (call x w y i s indices ok) equal
      subst d
      exact active hd

#axiomsClean returned_value
#axiomsClean all_values
#axiomsClean frame
#specNonCircular value avoiding [Target.real,Real.readback,Surface.source,Kernels.source,Kernels.originalFP16,Kernels.originalBF16,FP.Structural.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.RealTensor
