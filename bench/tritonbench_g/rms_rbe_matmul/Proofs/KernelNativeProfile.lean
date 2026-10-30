import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelNative
open VeriTile.Triton FP.Structural
noncomputable section

/-- Selected default SM80 family. Offline TTIR shows the separate source
dot/add absorbed into a dot seed; the source graph still retains both nodes.
Rotary add/sub pairs emit a ternary FMA with explicit operand/sign order. -/
structure Profile where
  schedule : Schedule
  dotCarry : ∀ carry x y, schedule.dotCarry
    (.binary (some .fp32) .real .add carry (.dot x y))=Bool.true
  rotaryAdd : ∀ a b c d, schedule.fusion (.binary (some .fp32) .real .add
      (.binary (some .fp32) .real .mul a b) (.binary (some .fp32) .real .mul c d))∈
        [SoftmaxFlaggems.Native.Fusion.right,SoftmaxFlaggems.Native.Fusion.rightSwap]
  rotarySub : ∀ a b c d, schedule.fusion (.binary (some .fp32) .real .sub
      (.binary (some .fp32) .real .mul a b) (.binary (some .fp32) .real .mul c d))∈
        [SoftmaxFlaggems.Native.Fusion.left,SoftmaxFlaggems.Native.Fusion.leftSwap]

/-- Inhabitant of the selected structural family. Numerical primitive results
remain explicit; this is not a native-conformance or split/FMA equality law. -/
def sourceDefault : Profile where
  schedule := {
    fusion := fun node => match node with
      | .binary _ _ .sub _ _ => .left
      | _ => .right
    swapMultiply := fun _ => Bool.false
    swapAdd := fun _ => Bool.false
    eraseFP32Cast := fun _ => Bool.true
    dotCarry := fun _ => Bool.true }
  dotCarry := by intros; rfl
  rotaryAdd := by intros; simp
  rotarySub := by intros; simp

#axiomsClean sourceDefault
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelNative
