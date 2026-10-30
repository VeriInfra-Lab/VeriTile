import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.Host
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Triton FP.Structural
open Host
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def OutputPost (x : Arguments) (initial final : State Word) (value : Index x → Int) : Prop :=
  (∀ i, final.mem x.c.region (Host.address x i)=Cell.mk .int (value i)) ∧
  (∀ r o, (r≠x.c.region ∨ ∀ i, o≠Host.address x i) → final.mem r o=initial.mem r o)

theorem empty_index (x : Arguments) (he : emptyOutput x) (i : Index x) : False := by
  rcases he with h|h
  · have hi := i.1.isLt; omega
  · have hi := i.2.isLt; omega

/-- Whole unscaled public call. The result is the supplied C tensor with
all logical cells equal to the independent signed8/modular32 matrix product. -/
specification int_matmul_correct (rest : Algebra Word) (x : Arguments) (initial : State Word)
    (h : Valid x) (_ha : Int8Storage x.a initial) (_hb : Int8Storage x.b initial) :
    ∃ final, Host.int_matmul_kernel rest x initial=some (x.c,final) ∧
      OutputPost x initial final (fun i => BlockWord.logicalDot (blockParameters x) initial i.1.val i.2.val) := by
  by_cases he : emptyOutput x
  · refine ⟨initial,by simp [Host.int_matmul_kernel,h.shape,he],?_,?_⟩
    · intro i; exact False.elim (empty_index x he i)
    · intro r o _; rfl
  · obtain ⟨final,hf,hread,hframe⟩ := BlockWord.grid_correct rest (blockParameters x)
      (geometry_valid x h) h.blockInner h.aSeparate h.bSeparate initial
    refine ⟨final,?_,?_,?_⟩
    · simp [Host.int_matmul_kernel,h.shape,he,hf]
    · intro i; exact hread i trivial
    · intro r o hh
      exact hframe r o (hh.imp id (fun h i _ => h i))

/-- Whole scaled public call with the source FP32 operations and final
defined signed32 conversion; no unbounded-integer or real-valued substitute. -/
specification int_scaled_matmul_correct (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (x : Arguments) (scale : Tensor) (initial : State Word)
    (h : Valid x) (hs : ScaleValid x scale)
    (_ha : Int8Storage x.a initial) (_hb : Int8Storage x.b initial) (_hs : FP32Storage scale initial)
    (hd : ScaledWord.LogicalDefined cfg (scaledParameters x scale) initial) :
    ∃ final, Host.int_scaled_matmul_kernel rest cfg u x scale initial=some (x.c,final) ∧
      OutputPost x initial final (fun i => ScaledWord.logicalResult cfg (scaledParameters x scale) initial i.1.val i.2.val) := by
  by_cases he : emptyOutput x
  · refine ⟨initial,by simp [Host.int_scaled_matmul_kernel,h.shape,he],?_,?_⟩
    · intro i; exact False.elim (empty_index x he i)
    · intro r o _; rfl
  · obtain ⟨final,hf,hread,hframe⟩ := ScaledWord.grid_correct rest cfg u (scaledParameters x scale)
      (scaled_geometry_valid x scale h) (launch_even x scale h) (launch_cover x scale h)
      h.aSeparate h.bSeparate hs.separate initial hd
    refine ⟨final,?_,?_,?_⟩
    · simp [Host.int_scaled_matmul_kernel,h.shape,he,hf]
    · intro i; exact hread i trivial
    · intro r o hh
      exact hframe r o (hh.imp id (fun h i _ => h i))

theorem scaled_result_conversion (cfg : FP.Config) (x : Arguments) (scale : Tensor) (s : State Word)
    (hd : ScaledWord.LogicalDefined cfg (scaledParameters x scale) s) (i : Index x) :
    ∃ q, (ScaledWord.logicalScaled cfg (scaledParameters x scale) s i.1.val i.2.val).toRat?=some q ∧
      FP.IntegerConversion.TowardZero q (ScaledWord.logicalResult cfg (scaledParameters x scale) s i.1.val i.2.val) ∧
      FP.IntegerConversion.SignedRange 32 (ScaledWord.logicalResult cfg (scaledParameters x scale) s i.1.val i.2.val) := by
  have hi := hd i.1.val i.2.val i.1.isLt i.2.isLt
  cases hz : FP.IntegerConversion.toSigned 32 (ScaledWord.logicalScaled cfg (scaledParameters x scale) s i.1.val i.2.val) with
  | none => simp [hz] at hi
  | some z =>
    simpa [ScaledWord.logicalResult,hz] using
      (FP.IntegerConversion.toSigned_spec 32 _ z).mp hz

theorem zero_inner_unscaled (x : Arguments) (s : State Word) (hK : x.a.columns=0) (r c : Nat) :
    BlockWord.logicalDot (blockParameters x) s r c=0 := by
  simp [BlockWord.logicalDot,blockParameters,hK,LogicalSum.sum,BlockWord.wrap]

#axiomsClean empty_index
#axiomsClean int_matmul_correct
#axiomsClean int_scaled_matmul_correct
#axiomsClean scaled_result_conversion
#axiomsClean zero_inner_unscaled
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
