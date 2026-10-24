import bench.tritonbench_g.fused_recurrent_retention.Proofs.Source
import bench.tritonbench_g.fused_recurrent_retention.Proofs.Model
import VeriTile.Meta.StatementAudit

/-! Exact decompositions of the full source trees. The pieces are obtained
from the actual source, and the equalities retain all statements, including
the optional final-state store and the backward barrier and rebasing. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.Program
open VeriTile.Triton
set_option maxHeartbeats 1500000
set_option maxRecDepth 16000

def forward (swapped : Bool) (dtype out_dtype : FloatDType) (p : Layout)
    (r : ForwardRegions) (hasSeed storeFinal : Bool) : ComputeKernel :=
  TypedSource.forward swapped dtype out_dtype r.q r.k r.v r.o r.initial r.final
    p.qHead p.qTime p.qChannel p.vHead p.vTime p.vChannel p.B p.H p.T p.scale
    p.BK p.BV p.DK p.DV hasSeed storeFinal

def backward (swapped : Bool) (dtype out_dtype : FloatDType) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) : ComputeKernel :=
  TypedSource.backward swapped dtype out_dtype r.q r.k r.v r.do_ r.dq r.dk r.dv r.initial
    p.qHead p.qTime p.qChannel p.vHead p.vTime p.vChannel p.B p.H p.T p.scale
    p.BK p.BV p.DK p.DV hasSeed

def forwardParts (swapped : Bool) (dtype out_dtype : FloatDType) (p : Layout)
    (r : ForwardRegions) (hasSeed storeFinal : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (forward swapped dtype out_dtype p r hasSeed storeFinal).surfaceBody
    "_t").getD ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

def backwardFirst (swapped : Bool) (dtype out_dtype : FloatDType) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (backward swapped dtype out_dtype p r hasSeed).surfaceBody
    "i").getD ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

def backwardSecond (swapped : Bool) (dtype out_dtype : FloatDType) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (backwardFirst swapped dtype out_dtype p r hasSeed).after
    "_t").getD ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

theorem forward_split (swapped : Bool) (dtype out_dtype : FloatDType) (p : Layout)
    (r : ForwardRegions) (hasSeed storeFinal : Bool) :
    (forward swapped dtype out_dtype p r hasSeed storeFinal).surfaceBody =
      (forwardParts swapped dtype out_dtype p r hasSeed storeFinal).before ++
        [.forRange "_t" 0 p.T 1
          (forwardParts swapped dtype out_dtype p r hasSeed storeFinal).body] ++
        (forwardParts swapped dtype out_dtype p r hasSeed storeFinal).after := by
  cases swapped <;> rfl

theorem backward_split (swapped : Bool) (dtype out_dtype : FloatDType) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) :
    (backward swapped dtype out_dtype p r hasSeed).surfaceBody =
      (backwardFirst swapped dtype out_dtype p r hasSeed).before ++
        [.forRange "i" 0 p.T 1
          (backwardFirst swapped dtype out_dtype p r hasSeed).body] ++
        (backwardSecond swapped dtype out_dtype p r hasSeed).before ++
        [.forRange "_t" 0 p.T 1
          (backwardSecond swapped dtype out_dtype p r hasSeed).body] := by
  cases swapped <;> rfl

theorem forward_body_same (dtype out_dtype : FloatDType) (p : Layout)
    (r : ForwardRegions) (hasSeed storeFinal : Bool) :
    (forwardParts Bool.false dtype out_dtype p r hasSeed storeFinal).body =
      (forwardParts Bool.true dtype out_dtype p r hasSeed storeFinal).body := rfl

theorem forward_suffix_same (dtype out_dtype : FloatDType) (p : Layout)
    (r : ForwardRegions) (hasSeed storeFinal : Bool) :
    (forwardParts Bool.false dtype out_dtype p r hasSeed storeFinal).after =
      (forwardParts Bool.true dtype out_dtype p r hasSeed storeFinal).after := rfl

theorem backward_first_body_same (dtype out_dtype : FloatDType) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) :
    (backwardFirst Bool.false dtype out_dtype p r hasSeed).body =
      (backwardFirst Bool.true dtype out_dtype p r hasSeed).body := rfl

theorem backward_suffix_same (dtype out_dtype : FloatDType) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) :
    (backwardFirst Bool.false dtype out_dtype p r hasSeed).after =
      (backwardFirst Bool.true dtype out_dtype p r hasSeed).after := rfl

theorem forward_body_length (swapped : Bool) (dtype out_dtype : FloatDType) (p : Layout)
    (r : ForwardRegions) (hasSeed storeFinal : Bool) :
    (forwardParts swapped dtype out_dtype p r hasSeed storeFinal).body.length = 11 := by
  cases swapped <;> rfl

theorem backward_first_length (swapped : Bool) (dtype out_dtype : FloatDType) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) :
    (backwardFirst swapped dtype out_dtype p r hasSeed).body.length = 11 := by
  cases swapped <;> rfl

theorem backward_second_length (swapped : Bool) (dtype out_dtype : FloatDType) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) :
    (backwardSecond swapped dtype out_dtype p r hasSeed).body.length = 16 := by
  cases swapped <;> rfl

#axiomsClean forward_split
#axiomsClean backward_split
#axiomsClean forward_body_same
#axiomsClean forward_suffix_same
#axiomsClean backward_first_body_same
#axiomsClean backward_suffix_same
#axiomsClean forward_body_length
#axiomsClean backward_first_length
#axiomsClean backward_second_length
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.Program
