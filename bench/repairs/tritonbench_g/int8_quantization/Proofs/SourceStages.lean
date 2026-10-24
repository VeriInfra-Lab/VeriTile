import bench.repairs.tritonbench_g.int8_quantization.Proofs.Model

namespace VeriTile.Bench.Repairs.Int8Quantization.Revision002
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

variable {α : Type} [Inhabited α]

abbrev Lane (p : Parameters) := TileIndex [p.width]
abbrev scalar32 (M : Algebra α) (v : ℝ) : α := M.literal (some .fp32) .real v

def loaded (M : Algebra α) (p : Parameters) (entry : State α) : Lane p → α :=
  fun i=> M.fp32Load (if Model.active p entry i then
    (entry.mem xRegion (Model.offset p entry i)).read .real else scalar32 M 0.0)

def transformed (M : Algebra α) (p : Parameters) (entry : State α) : Lane p → α :=
  fun i=> if p.query then M.binary (some .fp32) .real .mul (loaded M p entry i) (scalar32 M p.factor)
    else loaded M p entry i

def flags (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (entry : State α) : Lane p → Bool :=
  fun i=> ops.le (M.unary none (.abs .real) (transformed M p entry i))
    (M.literal none .real 340282346638528859811704183484516925440.0)

def badValue (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) : Int :=
  Finset.univ.sup' ⟨⟨0,positive⟩,Finset.mem_univ _⟩
    (fun j : Fin p.width=> if flags M ops p entry (j,PUnit.unit) then 0 else 1)

def sanitized (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (entry : State α) : Lane p → α :=
  fun i=> if flags M ops p entry i then transformed M p entry i else scalar32 M 0.0

def maximum (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (entry : State α) : α :=
  M.reduceMax (some .fp32) (shape:=[p.width]) ⟨0,by simp⟩ Bool.false
    (fun i=> M.unary (some .fp32) (.abs .real) (sanitized M ops p entry i)) PUnit.unit

def scale (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (entry : State α) : α :=
  M.binary (some .fp32) .real .div (maximum M ops p entry) (scalar32 M 127.0)

def usable (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) : Bool :=
  decide (badValue M ops p positive entry=0) && ops.less (M.literal none .real 0.0) (scale M ops p entry)

def denominator (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) : α :=
  if usable M ops p positive entry then scale M ops p entry else scalar32 M 1.0

def quotient (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) : Lane p → α :=
  fun i=> M.binary (some .fp32) .real .div (sanitized M ops p entry i) (denominator M ops p positive entry)

def normalized (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) : Lane p → α :=
  fun i=> if usable M ops p positive entry then quotient M ops p positive entry i else scalar32 M 0.0

def clipped (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) : Lane p → α :=
  fun i=> let lower:=M.binary (some .fp32) .real .max (normalized M ops p positive entry i) (scalar32 M (-127.0))
    ops.minimum lower (scalar32 M 127.0)

def biased (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) : Lane p → α :=
  fun i=> let z:=clipped M ops p positive entry i
    M.binary (some .fp32) .real .add z (M.binary (some .fp32) .real .mul (scalar32 M 0.5)
      (if ops.le (scalar32 M 0.0) z then scalar32 M 1.0 else scalar32 M (-1.0)))

def quantized (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) : Lane p → Int :=
  fun i=> ops.int8 (biased M ops p positive entry i)

def savedScale (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) : α :=
  if badValue M ops p positive entry=0 then scale M ops p entry else scalar32 M 0.0

def addressed (M : Algebra α) (p : Parameters) (entry : State α) : State α :=
  ((((((((entry.setReg "block" .nat [] (fun _=> Model.group p entry)).setReg "batch" .nat [] (fun _=> Model.group p entry/p.blocks)).setReg "local_block" .nat [] (fun _=> Model.group p entry%p.blocks)).setReg "lane" .nat [p.width] (fun i=> i.1.val)).setReg "row" .nat [p.width] (fun i=> Model.group p entry%p.blocks*p.blockRows+i.1.val/p.channels)).setReg "offset" .nat [p.width] (Model.offset p entry)).setReg "active" .bool [p.width] (fun i=> decide (Model.active p entry i))).setReg "x" .real [p.width] (loaded M p entry))

def prescaled (M : Algebra α) (p : Parameters) (entry : State α) : State α :=
  if p.query then (addressed M p entry).setReg "x" .real [p.width] (transformed M p entry)
  else addressed M p entry

def classified (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) : State α :=
  (((prescaled M p entry).setReg "finite" .bool [p.width] (flags M ops p entry)).setReg "bad" .int [] (fun _=> badValue M ops p positive entry)).setReg "x" .real [p.width] (sanitized M ops p entry)

def reduced (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) : State α :=
  ((((classified M ops p positive entry).setReg "maximum" .real [] (fun _=> maximum M ops p entry)).setReg "scale" .real [] (fun _=> scale M ops p entry)).setReg "usable" .bool [] (fun _=> usable M ops p positive entry)).setReg "denominator" .real [] (fun _=> denominator M ops p positive entry)

def registers (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) : State α :=
  (((((reduced M ops p positive entry).setReg "normalized" .real [p.width] (quotient M ops p positive entry)).setReg "normalized" .real [p.width] (normalized M ops p positive entry)).setReg "normalized" .real [p.width] (clipped M ops p positive entry)).setReg "biased" .real [p.width] (biased M ops p positive entry)).setReg "quantized" .int [p.width] (quantized M ops p positive entry)

def segment (p : Parameters) (start count : Nat) : List ComputeStmt :=
  ((source p).surfaceBody.drop start).take count

theorem addressed_run (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (entry : State α) :
    run (Model.backend M ops) (segment p 0 8) entry=some (addressed M p entry) := by
  simp [segment,source,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Model.backend,Option.bind,numericLt,numericLe,numericEq,natLt,natLe,
    bop,numeric,ofFloat,toFloat,Region.cast,ComputeDType.eraseDType,
    integralMod,integralFloorDiv,addressed,loaded,Model.group,Model.offset,Model.active]
  all_goals
    delta bop Model.offset Model.group loaded transformed flags sanitized scalar32 maximum scale usable denominator quotient normalized clipped biased quantized
    first | rfl | simp_all (config := {maxSteps:=1000000}) only [Bool.false_eq_true, ↓reduceIte]
    all_goals try rfl

theorem prescaled_run (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (entry : State α) :
    run (Model.backend M ops) (segment p 8 1) (addressed M p entry)=some (prescaled M p entry) := by
  cases query : p.query <;>
    simp [segment,source,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Model.backend,Option.bind,bop,numeric,addressed,prescaled,transformed,query,ComputeDType.eraseDType]
  all_goals
    delta bop Model.offset Model.group loaded transformed flags sanitized scalar32 maximum scale usable denominator quotient normalized clipped biased quantized
    first | rfl | simp_all (config := {maxSteps:=1000000}) only [Bool.false_eq_true, ↓reduceIte]
    all_goals try rfl

theorem classified_run (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    run (Model.backend M ops) (segment p 9 3) (prescaled M p entry)=some (classified M ops p positive entry) := by
  cases query : p.query <;>
    simp [segment,source,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Model.backend,Option.bind,numericLe,numericEq,bop,numeric,ofFloat,toFloat,
      classified,prescaled,addressed,flags,badValue,sanitized,transformed,query,
      TileShape.discreteMaximum,TileShape.axisDim,positive,TileShape.insertAxisIndex,ComputeDType.eraseDType]
  all_goals
    delta bop Model.offset Model.group loaded transformed flags sanitized scalar32 maximum scale usable denominator quotient normalized clipped biased quantized
    first | rfl | simp_all (config := {maxSteps:=1000000}) only [Bool.false_eq_true, ↓reduceIte]
    all_goals try rfl

theorem reduced_run (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    run (Model.backend M ops) (segment p 12 4) (classified M ops p positive entry)=some (reduced M ops p positive entry) := by
  simp [segment,source,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Model.backend,Option.bind,numericLt,numericEq,bop,numeric,ofFloat,toFloat,
    reduced,classified,maximum,scale,usable,denominator,TileShape.axisDim,positive,ComputeDType.eraseDType]
  all_goals
    delta bop Model.offset Model.group loaded transformed flags sanitized scalar32 maximum scale usable denominator quotient normalized clipped biased quantized
    first | rfl | simp_all (config := {maxSteps:=1000000}) only [Bool.false_eq_true, ↓reduceIte]
    all_goals try rfl

theorem registers_run (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    run (Model.backend M ops) (segment p 16 5) (reduced M ops p positive entry)=some (registers M ops p positive entry) := by
  simp [segment,source,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Model.backend,Option.bind,numericLt,numericLe,bop,numeric,ofFloat,toFloat,
    registers,reduced,classified,quotient,normalized,clipped,biased,quantized,ComputeDType.eraseDType]
  all_goals
    delta bop Model.offset Model.group loaded transformed flags sanitized scalar32 maximum scale usable denominator quotient normalized clipped biased quantized
    first | rfl | simp_all (config := {maxSteps:=1000000}) only [Bool.false_eq_true, ↓reduceIte]
    all_goals try rfl

#axiomsClean addressed_run
#axiomsClean prescaled_run
#axiomsClean classified_run
#axiomsClean reduced_run
#axiomsClean registers_run
end
end VeriTile.Bench.Repairs.Int8Quantization.Revision002
