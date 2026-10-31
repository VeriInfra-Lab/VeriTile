import bench.tritonbench_g.chunk_gated_attention.FPEquiv
import bench.tritonbench_g.chunk_gated_attention.Proofs.HSourceFP
import bench.tritonbench_g.cache_transform_triton.Sources.Mathematical_IndexRewrite

/-! Check the typed source before erasure: actual FP32 storage, optional seed
addition, history/final store conversions, exact endpoint addressing and
intrinsic exp. The FP tactic must retain the actual reached domain. -/
namespace FPChunkGatedSourceTests
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open VeriTile.Bench.Repairs.ChunkGatedAttention
open scoped VeriTile.Spec
set_option linter.unusedVariables false
set_option maxRecDepth 18000
set_option maxHeartbeats 3000000

private def source (BT BK BV : Nat) (gateK initial final : Bool) :=
  NativeSource.hOriginal .fp32 .fp32 .fp32 .fp32 .fp32 "k" "v" "g" "h" "h0" "ht"
    15 3 1 15 3 1 18 3 1 5 3 3 BT BK BV 2 7 gateK initial final
private def loop (BT BK BV : Nat) (gateK initial final : Bool) :=
  (Source.intRange? (source BT BK BV gateK initial final).surfaceBody "i_t").getD
    ⟨[],.constInt 0,.constInt 0,.constNat 0,[],[]⟩
private def gate (BT BK BV : Nat) (gateK initial final : Bool) :=
  (Source.conditional? (loop BT BK BV gateK initial final).body (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
private def seed (BT BK BV : Nat) (gateK initial final : Bool) :=
  (Source.conditional? (loop BT BK BV gateK initial final).before (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
private def finalBranch (BT BK BV : Nat) (gateK initial final : Bool) :=
  (Source.conditional? (loop BT BK BV gateK initial final).after (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
private def assignment (body : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) :=
  (body.drop (Source.prefixBefore body name occurrence).length).head?

/-- Loading the FP32 seed is followed by an addition, not an assignment. -/
theorem seed_addition (BT BK BV : Nat) (gateK initial final : Bool) :
    assignment (seed BT BK BV gateK initial final).yes "b_h" = some
      (.assign .real [BK,BV] "b_h" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [BK,BV] "b_h")
          (.castFloat .fp32 .real (.load .fp32
            (.blockPtr (.ref .blockPtr [BK,BV] "p_h") [0,1]) .none .unspecified)))))) := rfl

theorem history_storage (BT BK BV : Nat) (gateK initial final : Bool) :
    ((loop BT BK BV gateK initial final).body.drop
      (Source.prefixBeforeStore (loop BT BK BV gateK initial final).body).length).head? = some
      (.store .fp32 [BK,BV] (.blockPtr (.ref .blockPtr [BK,BV] "p_h") [0,1])
        (.alg (.castFloat .fp32 .fp32 (.withPrecision .fp32 .fp32
          (.castFloat .real .fp32 (.ref .real [BK,BV] "b_h"))))) .none) := rfl

theorem final_storage (BT BK BV : Nat) (gateK initial final : Bool) :
    ((finalBranch BT BK BV gateK initial final).yes.drop
      (Source.prefixBeforeStore (finalBranch BT BK BV gateK initial final).yes).length).head? = some
      (.store .fp32 [BK,BV] (.blockPtr (.ref .blockPtr [BK,BV] "p_h") [0,1])
        (.alg (.castFloat .fp32 .fp32 (.withPrecision .fp32 .fp32
          (.castFloat .real .fp32 (.ref .real [BK,BV] "b_h"))))) .none) := rfl

/-- Repaired endpoint: rebased pointer, native offsets exactly zero. -/
theorem key_endpoint (BT BK BV : Nat) (gateK initial final : Bool) :
    assignment (gate BT BK BV gateK initial final).yes "p_gn" = some
      (.assign .blockPtr [BK] "p_gn" (.alg (.makeBlockPtrDynMetadata "g"
        (.castIntToNat (.add .int .nil
          (.add .int .nil (.mul .int .nil (.ref .int [] "i_bh") (.ref .int [] "s_k_h"))
            (.mul .int .nil (.sub .int .nil
              (.add .int .nil (.mul .int .nil (.ref .int [] "i_t") (.castNatToInt (.constNat BT)))
                (.castNatToInt (.constNat BT))) (.constInt 1)) (.ref .int [] "s_k_t")))
          (.mul .int .nil (.mul .int .nil (.ref .int [] "i_k") (.castNatToInt (.constNat BK)))
            (.ref .int [] "s_k_d"))))
        [BK] [BK] [.castIntToNat (.ref .int [] "s_k_d")] [.constNat 0]))) := rfl

theorem intrinsic_key_gate (BT BK BV : Nat) (gateK initial final : Bool) :
    assignment (gate BT BK BV gateK initial final).yes "b_h" = some
      (.assign .real [BK,BV] "b_h" (.compute (.alg .fp32
        (.mul .real (.consSame (.consR .nil)) (.ref .real [BK,BV] "b_h")
          (.expandDim ⟨1,by simp⟩ (.exp (.castFloat .fp32 .real (.ref .fp32 [BK] "b_gn")))))))) := rfl

theorem intrinsic_value_gate (BT BK BV : Nat) (gateK initial final : Bool) :
    assignment (gate BT BK BV gateK initial final).no "b_h" = some
      (.assign .real [BK,BV] "b_h" (.compute (.alg .fp32
        (.mul .real (.consR (.consSame .nil)) (.ref .real [BK,BV] "b_h")
          (.expandDim ⟨0,by simp⟩ (.exp (.castFloat .fp32 .real (.ref .fp32 [BV] "b_gn")))))))) := rfl

/-- Both dot operands remain FP32 after the gated operand's storage cast. -/
theorem final_addition_precision (BT BK BV : Nat) (gateK initial final : Bool) :
    assignment (gate BT BK BV gateK initial final).yes "b_h" 1 = some
      (.assign .real [BK,BV] "b_h" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [BK,BV] "b_h")
          (HSourceFP.blockDot .fp32 BK BT BV))))) := rfl

theorem final_value_addition_precision (BT BK BV : Nat) (gateK initial final : Bool) :
    assignment (gate BT BK BV gateK initial final).no "b_h" 1 = some
      (.assign .real [BK,BV] "b_h" (.compute (.alg .fp32
        (.add .real (.consSame (.consSame .nil)) (.ref .real [BK,BV] "b_h")
          (HSourceFP.blockDot .fp32 BK BT BV))))) := rfl

example (R : FP.ScalarArithmetic.Rules) : True := by
  fail_if_success
    have :
        ({ NativeFP.recurrent .fp32 .fp32 .fp32 .fp32 .fp32 "k" "v" "g" "h" "h0" "ht"
          15 3 1 15 3 1 18 3 1 5 3 3 4 2 2 2 7 Bool.true Bool.true Bool.true with domain := [] } : Program) ≡[R]
        ({ NativeFP.recurrentSumLeft .fp32 .fp32 .fp32 .fp32 .fp32 "k" "v" "g" "h" "h0" "ht"
          15 3 1 15 3 1 18 3 1 5 3 3 4 2 2 2 7 Bool.true Bool.true Bool.true with domain := [] } : Program) := by
      equiv_decompose; fp_prove
  trivial

private def ptrProgram (swapped : Bool) (rowOffset : Nat) : Program where
  kernel := .mk [] [] [.assign .blockPtr [2,2] "p" (.alg
    (.makeBlockPtrDynOffsets "x"
      (if swapped then .mul .nat .nil (.constNat 8) (.programId 0)
       else .mul .nat .nil (.programId 0) (.constNat 8)) [4,4] [2,2] [4,1]
      [.constNat rowOffset,.constNat 0]))]
  domain := []

theorem integer_block_pointer :
    ptrProgram Bool.false 0 ≡[([] : Spec.Assumptions FP.GuardedFragment)] ptrProgram Bool.true 0 := by
  equiv_decompose; fp_prove

-- The new base congruence must not silently change the pointer's offsets.
example : True := by
  fail_if_success
    have : ptrProgram Bool.false 0 ≡[([] : Spec.Assumptions FP.GuardedFragment)] ptrProgram Bool.true 1 := by
      equiv_decompose; fp_prove
  trivial

/-- info: FP assumptions used by prefix_source_equiv:
---
info:   none -/
#guard_msgs in
#print_fp_assumptions VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.prefix_source_equiv
/-- info: FP assumptions used by inner_source_equiv:
---
info:   none
---
info: Strict model certificates used by inner_source_equiv:
---
info:   model: VeriTile.Software.strictFP32/ieeeValue/maxnum/documented-exp-special-values
    certificate: List.Forall₂.cons (FP.ScalarArithmetic.strict_certificate FP.ScalarArithmetic.Atom.addCommute rfl) List.Forall₂.nil -/
#guard_msgs in
#print_fp_assumptions VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.inner_source_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end FPChunkGatedSourceTests
