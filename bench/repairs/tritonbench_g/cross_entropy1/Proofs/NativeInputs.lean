import bench.repairs.tritonbench_g.cross_entropy1.Proofs.LinkedReturn

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host Floating
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Exact little-endian assembly from four actual native bytes. -/
def logicalWord {σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (state : σ) (tensor : Tensor) (index : Nat) : Floating.Word :=
  let byte := tensorByte layout observe state tensor index
  let raw := ((byte ⟨3,by decide⟩).append (byte ⟨2,by decide⟩)).append
    ((byte ⟨1,by decide⟩).append (byte ⟨0,by decide⟩))
  if tensor.isNeg then flipSign raw else raw

/-- The label view uses its actual eight-byte element offset. Lazy integer
negation is performed in int64 before signed interpretation, including MIN. -/
def logicalInt64 {σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (state : σ) (tensor : Tensor) (index : Nat) : Int :=
  let byte := fun j => observe state (layout.buffer state tensor.region).device
    ((layout.buffer state tensor.region).base+8*tensor.index index+j)
  let low := ((byte 3).append (byte 2)).append ((byte 1).append (byte 0))
  let high := ((byte 7).append (byte 6)).append ((byte 5).append (byte 4))
  let raw : BitVec 64 := high.append low
  (if tensor.isNeg then -raw else raw).toInt

/-- Ordinary typed observations at individual caller coordinates. These bind
source input atoms to actual input backing bytes and complete tensor views;
none constrains a kernel, saved result, reduction, or returned output. -/
structure InitialInputViews {σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (obs : Observations Expr σ) (call : Entry.Invocation obs algebra) : Prop where
  logits : ∀ row, row<call.data.rows → ∀ col, col<call.data.cols →
    value obs call.initial call.logits (row*call.data.cols+col)=
      .raw (logicalWord layout observe call.initial.native call.logits (row*call.data.cols+col))
  labels : ∀ row, row<call.data.rows → obs.integer call.initial call.labels row=
    logicalInt64 layout observe call.initial.native call.labels row

structure UpstreamInputViews {σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (obs : Observations Expr σ) (before : World Expr σ) (upstream : Tensor) (rows : Nat) : Prop where
  values : ∀ row, row<rows → value obs before upstream row=
    .raw (logicalWord layout observe before.native upstream row)

#specNonCircular logicalWord avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#specNonCircular logicalInt64 avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
