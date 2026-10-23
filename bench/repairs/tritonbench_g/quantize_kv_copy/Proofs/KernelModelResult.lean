import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelModels
import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelResult

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.KernelModelResult
open VeriTile.Triton FP.Structural Kernel KernelModels
noncomputable section
local instance : Inhabited Word := ⟨KernelModels.zero⟩

 theorem native (a : NativeInstructions) (tree : ReductionTree) (p : Parameters)
    (s : State Word) (g : Fin p.block) (hg : g.val<p.groups) (x : Nat → Word)
    (reads : ∀ c,c<8 → (s.mem "K" (inputOffset p (s.pids 0*p.heads+s.pids 1) g.val c)).read .real=x c) :
    Kernel.scale (nativeOps a tree) p s g=nativeScale a tree x ∧
    ∀ c : Fin 8, Kernel.code (nativeOps a tree) p s (g,c,PUnit.unit)=(nativeCode a tree x c.val).toInt := by
  have hl (c : Fin 8) : loaded (nativeOps a tree) p s (g,c,PUnit.unit)=x c.val := by
    simp [loaded,hg,reads c.val c.isLt]
  have hm : Kernel.maximum (nativeOps a tree) p s g=nativeMax a tree x := by
    simp only [Kernel.maximum,hl]
    exact native_maximum a tree x
  constructor
  · simp only [Kernel.scale,hm]
    rfl
  · intro c
    simp only [Kernel.code,hl,Kernel.divisor,Kernel.scale,hm]
    rfl

 theorem mathematical (p : Parameters) (s : State ℝ) (g : Fin p.block)
    (hg : g.val<p.groups) (x : Nat → ℝ)
    (reads : ∀ c,c<8 → (s.mem "K" (inputOffset p (s.pids 0*p.heads+s.pids 1) g.val c)).read .real=x c) :
    Kernel.scale KernelModels.realOps p s g=realScale x ∧
    ∀ c : Fin 8, Kernel.code KernelModels.realOps p s (g,c,PUnit.unit)=(realCode x c.val).toInt := by
  have hl (c : Fin 8) : loaded KernelModels.realOps p s (g,c,PUnit.unit)=x c.val := by
    simp [loaded,hg,reads c.val c.isLt]
  have hm : Kernel.maximum KernelModels.realOps p s g=mathematicalMaximum x 8 := by
    simp only [Kernel.maximum,hl]
    exact realMaximum_abs x
  constructor
  · simp only [Kernel.scale,hm]
    rfl
  · intro c
    simp only [Kernel.code,hl,Kernel.divisor,Kernel.scale,hm]
    by_cases hz : mathematicalMaximum x 8=0
    · have hx := (RowwiseQuantizationTriton.logicalMaximum_zero x 8).mp hz c.val c.isLt
      simp [KernelModels.realOps,realCode,hz,hx]
    · simp [KernelModels.realOps,realCode,realScale,hz]

#axiomsClean native
#axiomsClean mathematical
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.KernelModelResult
