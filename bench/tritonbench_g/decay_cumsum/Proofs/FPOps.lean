import bench.tritonbench_g.decay_cumsum.Proofs.FPReference
import VeriTile.Triton.Float.Control
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.FPStep

namespace VeriTile.Bench.Repairs.DecayCumsum.FPOps
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def ptr (BK : Nat) (r : RegionName) (off : Fin BK → Nat) : Values α .ptr [BK] :=
  fun i => (r,off i.1)
def raw (BK : Nat) (r : RegionName) (off : Fin BK → Nat) (u : State α) : Values α .real [BK] :=
  fun i => (u.mem r (off i.1)).read .real
def maskOp (BK : Nat) : Op .bool [BK] := .ref .bool [BK] "mask"
def pointerOp (BK : Nat) (name : RegName) : Op .ptr [BK] := .ref .ptr [BK] name
def loadOp (BK : Nat) (name : RegName) : Op .real [BK] :=
  .load .real (.ptr (pointerOp BK name)) (.maskOther (maskOp BK) ((Op.const (0:ℝ)).broadcast [BK]))
def gateOp (BK : Nat) (name : RegName) : Op .real [BK] := .castFloat .real .real (loadOp BK name)
def refOp (BK : Nat) (name : RegName) : Op .real [BK] := .ref .real [BK] name
def fwdOp (BK : Nat) : Op .real [BK] :=
  .add .real (.consSame .nil) (refOp BK "cum_decay")
    (.mul .real .scalarR (refOp BK "_g") (.const 1.44269504))
def prepareQOp (BK : Nat) (scale : ℝ) : Op .real [BK] :=
  .mul .real (.consSame .nil) (refOp BK "_q")
    (.mul .real .scalarR (.exp2 (refOp BK "_g")) (.const scale))
def gatedOp (BK : Nat) (value gateName : RegName) : Op .real [BK] :=
  .mul .real (.consSame .nil) (refOp BK value) (.exp2 (refOp BK gateName))
def differenceGatedOp (BK : Nat) (value last gateName : RegName) : Op .real [BK] :=
  .mul .real (.consSame .nil) (refOp BK value)
    (.exp2 (.sub .real (.consSame .nil) (refOp BK last) (refOp BK gateName)))
def addOp (BK : Nat) (x y : RegName) : Op .real [BK] :=
  .add .real (.consSame .nil) (refOp BK x) (refOp BK y)
def gradientOp (BK : Nat) : Op .real [BK] :=
  .sub .real (.consSame .nil)
    (.mul .real (.consSame .nil) (refOp BK "_dq") (refOp BK "_q"))
    (.mul .real (.consSame .nil) (refOp BK "_dk") (refOp BK "_k"))

def storeCode (BK : Nat) (pointer value : RegName) : ComputeStmt :=
  .store .real [BK] (.ptr (pointerOp BK pointer))
    (.compute (.alg .fp32 (.castFloat .real .real (refOp BK value)))) (.mask (maskOp BK))
def advanceCode (BK width : Nat) (name : RegName) : ComputeStmt :=
  .assign .ptr [BK] name (.alg (.ptrAdd .scalarR (pointerOp BK name) (.constNat width)))
def retreatCode (BK width : Nat) (name : RegName) : ComputeStmt :=
  .assign .ptr [BK] name (.alg (.ptrSub .scalarR (pointerOp BK name) (.constNat width)))
def assign32 (BK : Nat) (name : RegName) (op : Op .real [BK]) : ComputeStmt :=
  .assign .real [BK] name (.compute (.alg .fp32 op))
def dataLoadCode (BK : Nat) (name pointer : RegName) : ComputeStmt :=
  .assign .real [BK] name (.compute (.load .fp32 (.ptr (pointerOp BK pointer))
    (.maskOther (maskOp BK) ((Op.const (0:ℝ)).broadcast [BK]))))

def forwardBody (h : Call) : List ComputeStmt :=
  [assign32 h.BK "_g" (gateOp h.BK "p_g"),
   assign32 h.BK "cum_decay" (fwdOp h.BK),storeCode h.BK "p_go" "cum_decay",
   advanceCode h.BK h.width "p_g",advanceCode h.BK h.width "p_go"]

def prepareBody (h : Call) (scale : ℝ) : List ComputeStmt :=
  [dataLoadCode h.BK "_q" "p_q",dataLoadCode h.BK "_k" "p_k",
   assign32 h.BK "_g" (gateOp h.BK "p_g"),
   assign32 h.BK "_q" (prepareQOp h.BK scale),
   assign32 h.BK "_k" (differenceGatedOp h.BK "_k" "last_decay" "_g"),
   storeCode h.BK "p_kg" "_k",storeCode h.BK "p_qg" "_q",
   advanceCode h.BK h.width "p_q",advanceCode h.BK h.width "p_g",advanceCode h.BK h.width "p_k",
   advanceCode h.BK h.width "p_kg",advanceCode h.BK h.width "p_qg"]

def backwardArithmetic (h : Call) : List ComputeStmt :=
  [dataLoadCode h.BK "_dq1" "p_dq_inner",dataLoadCode h.BK "_dq2" "p_dq_inter",
   assign32 h.BK "_dq2" (gatedOp h.BK "_dq2" "_g"),
   assign32 h.BK "_dq" (addOp h.BK "_dq1" "_dq2"),storeCode h.BK "p_dq_inter" "_dq",
   dataLoadCode h.BK "_dk1" "p_dk_inner",dataLoadCode h.BK "_dk2" "p_dk_inter",
   assign32 h.BK "_dk2" (differenceGatedOp h.BK "_dk2" "last_g" "_g"),
   assign32 h.BK "_dk" (addOp h.BK "_dk1" "_dk2"),storeCode h.BK "p_dk_inter" "_dk",
   dataLoadCode h.BK "_q" "p_q",dataLoadCode h.BK "_k" "p_k",
   assign32 h.BK "_dg" (gradientOp h.BK),
   assign32 h.BK "cum_grad_dg" (addOp h.BK "cum_grad_dg" "_dg"),
   storeCode h.BK "p_dg" "cum_grad_dg",
   retreatCode h.BK h.width "p_g",retreatCode h.BK h.width "p_k",retreatCode h.BK h.width "p_q",
   retreatCode h.BK h.width "p_dq_inner",retreatCode h.BK h.width "p_dk_inner",
   retreatCode h.BK h.width "p_dq_inter",retreatCode h.BK h.width "p_dk_inter",
   retreatCode h.BK h.width "p_dg"]

def reverseIndex : ComputeStmt := .assign .int [] "j"
  (.alg (.sub .int .nil (.constInt 0) (.ref .int [] "__rev_j")))
def lastGate (BK span : Nat) : ComputeStmt :=
  .ifThen (.eq .int .nil (.ref .int [] "j")
    ((Op.sub .nat .nil (.constNat span) (.constNat 1)).castNatToInt))
    [assign32 BK "last_g" (refOp BK "_g")]

def backwardBody (h : Call) (p : Phase) : List ComputeStmt :=
  [reverseIndex,assign32 h.BK "_g" (gateOp h.BK "p_g"),lastGate h.BK p.span]++backwardArithmetic h

theorem forward_body (h : Call) (p : Phase) (scale : ℝ) :
    FPProgram.loopBody .forward h p scale=forwardBody h := by rfl
theorem prepare_body (h : Call) (p : Phase) (scale : ℝ) :
    FPProgram.loopBody .prepare h p scale=prepareBody h scale := by rfl
theorem backward_body (h : Call) (p : Phase) (scale : ℝ) :
    FPProgram.loopBody .backward h p scale=backwardBody h p := by rfl

theorem raw_load (M : Algebra α) (BK : Nat) (r : RegionName) (off : Fin BK → Nat)
    (name : RegName) (u : State α)
    (hp : u.regs .ptr [BK] name=some (ptr BK r off))
    (hm : u.regs .bool [BK] "mask"=some (fun _ => Bool.true)) :
    FP.Structural.evalOp (engine M) (some .fp32) (loadOp BK name) u=some (raw BK r off u) := by
  simp only [loadOp,pointerOp,maskOp,FP.Structural.evalOp,hp,hm]
  rfl

theorem gate_load (M : Algebra α) (BK : Nat) (r : RegionName) (off : Fin BK → Nat)
    (name : RegName) (u : State α)
    (hp : u.regs .ptr [BK] name=some (ptr BK r off))
    (hm : u.regs .bool [BK] "mask"=some (fun _ => Bool.true)) :
    FP.Structural.evalOp (engine M) (some .fp32) (gateOp BK name) u=
      some (fun i => FPReference.gate M (raw BK r off u i)) := by
  change (do
    let v ← FP.Structural.evalOp (engine M) (some .fp32) (loadOp BK name) u
    pure (fun i => FPReference.gate M (v i))) = _
  rw [raw_load M BK r off name u hp hm]
  rfl

theorem data_load (M : Algebra α) (BK : Nat) (r : RegionName) (off : Fin BK → Nat)
    (name pointer : RegName) (u : State α)
    (hp : u.regs .ptr [BK] pointer=some (ptr BK r off))
    (hm : u.regs .bool [BK] "mask"=some (fun _ => Bool.true)) :
    step (engine M) (dataLoadCode BK name pointer) u=
      some (u.setReg name .real [BK] (fun i => M.fp32Load (raw BK r off u i))) := by
  simp only [dataLoadCode,step,evalExpr,evalComputeOp]
  change (do
    let v ← (do
      let v ← FP.Structural.evalOp (engine M) (some .fp32) (loadOp BK pointer) u
      pure (fun i => M.fp32Load (v i)))
    pure (u.setReg name .real [BK] v))=_
  rw [raw_load M BK r off pointer u hp hm]
  rfl

theorem forward_update (M : Algebra α) (BK : Nat) (acc g : Values α .real [BK]) (u : State α)
    (ha : u.regs .real [BK] "cum_decay"=some acc) (hg : u.regs .real [BK] "_g"=some g) :
    FP.Structural.evalOp (engine M) (some .fp32) (fwdOp BK) u=
      some (fun i => FPReference.add M (acc i)
        (FPReference.mul M (g i) (M.literal (some .fp32) .real 1.44269504))) := by
  simp only [fwdOp,refOp,FP.Structural.evalOp,ha,hg]
  rfl

theorem prepare_q (M : Algebra α) (BK : Nat) (scale : ℝ) (q g : Values α .real [BK]) (u : State α)
    (hq : u.regs .real [BK] "_q"=some q) (hg : u.regs .real [BK] "_g"=some g) :
    FP.Structural.evalOp (engine M) (some .fp32) (prepareQOp BK scale) u=
      some (fun i => FPReference.mul M (q i)
        (FPReference.mul M (FPReference.exp2 M (g i)) (M.literal (some .fp32) .real scale))) := by
  simp only [prepareQOp,refOp,FP.Structural.evalOp,hq,hg]
  rfl

theorem difference_gated (M : Algebra α) (BK : Nat) (value last gateName : RegName)
    (v l g : Values α .real [BK]) (u : State α)
    (hv : u.regs .real [BK] value=some v) (hl : u.regs .real [BK] last=some l)
    (hg : u.regs .real [BK] gateName=some g) :
    FP.Structural.evalOp (engine M) (some .fp32) (differenceGatedOp BK value last gateName) u=
      some (fun i => FPReference.mul M (v i) (FPReference.exp2 M (FPReference.sub M (l i) (g i)))) := by
  simp only [differenceGatedOp,refOp,FP.Structural.evalOp,hv,hl,hg]
  rfl

theorem gated (M : Algebra α) (BK : Nat) (value gateName : RegName)
    (v g : Values α .real [BK]) (u : State α)
    (hv : u.regs .real [BK] value=some v) (hg : u.regs .real [BK] gateName=some g) :
    FP.Structural.evalOp (engine M) (some .fp32) (gatedOp BK value gateName) u=
      some (fun i => FPReference.mul M (v i) (FPReference.exp2 M (g i))) := by
  simp only [gatedOp,refOp,FP.Structural.evalOp,hv,hg]
  rfl

theorem add_eval (M : Algebra α) (BK : Nat) (x y : RegName)
    (v w : Values α .real [BK]) (u : State α)
    (hv : u.regs .real [BK] x=some v) (hw : u.regs .real [BK] y=some w) :
    FP.Structural.evalOp (engine M) (some .fp32) (addOp BK x y) u=
      some (fun i => FPReference.add M (v i) (w i)) := by
  simp only [addOp,refOp,FP.Structural.evalOp,hv,hw]
  rfl

theorem gradient_eval (M : Algebra α) (BK : Nat)
    (dq dk q k : Values α .real [BK]) (u : State α)
    (hdq : u.regs .real [BK] "_dq"=some dq) (hdk : u.regs .real [BK] "_dk"=some dk)
    (hq : u.regs .real [BK] "_q"=some q) (hk : u.regs .real [BK] "_k"=some k) :
    FP.Structural.evalOp (engine M) (some .fp32) (gradientOp BK) u=
      some (fun i => FPReference.sub M (FPReference.mul M (dq i) (q i)) (FPReference.mul M (dk i) (k i))) := by
  simp only [gradientOp,refOp,FP.Structural.evalOp,hdq,hdk,hq,hk]
  rfl

theorem advance_eval (M : Algebra α) (BK width : Nat) (r : RegionName) (off : Fin BK → Nat)
    (name : RegName) (u : State α) (hp : u.regs .ptr [BK] name=some (ptr BK r off)) :
    step (engine M) (advanceCode BK width name) u=
      some (u.setReg name .ptr [BK] (ptr BK r (fun i => off i+width))) := by
  simp only [advanceCode,pointerOp,step,evalExpr,FP.Structural.evalOp,hp]
  rfl

theorem retreat_eval (M : Algebra α) (BK width : Nat) (r : RegionName) (off : Fin BK → Nat)
    (name : RegName) (u : State α) (hp : u.regs .ptr [BK] name=some (ptr BK r off)) :
    step (engine M) (retreatCode BK width name) u=
      some (u.setReg name .ptr [BK] (ptr BK r (fun i => off i-width))) := by
  simp only [retreatCode,pointerOp,step,evalExpr,FP.Structural.evalOp,hp]
  rfl

def scatter (M : Algebra α) (BK : Nat) (r : RegionName) (off : Fin BK → Nat)
    (value : Values α .real [BK]) (u : State α) : State α :=
  (TileShape.allIndices [BK]).foldl (fun v i => v.write r (off i.1) (FPReference.stored M (value i))) u

@[simp] theorem scatter_regs (M : Algebra α) (BK : Nat) (r : RegionName) (off : Fin BK → Nat)
    (value : Values α .real [BK]) (u : State α) : (scatter M BK r off value u).regs=u.regs := by
  apply State.scatter_regs

@[simp] theorem scatter_other (M : Algebra α) (BK : Nat) (r : RegionName) (off : Fin BK → Nat)
    (value : Values α .real [BK]) (u : State α) (r' : RegionName) (o : Nat) (different : r'≠r) :
    (scatter M BK r off value u).mem r' o=u.mem r' o :=
  State.scatter_frame _ _ _ _ _ _ (Or.inl different) _

theorem store_eval (M : Algebra α) (BK : Nat) (r : RegionName) (off : Fin BK → Nat)
    (pointer value : RegName) (v : Values α .real [BK]) (u : State α)
    (hp : u.regs .ptr [BK] pointer=some (ptr BK r off))
    (hm : u.regs .bool [BK] "mask"=some (fun _ => Bool.true))
    (hv : u.regs .real [BK] value=some v) :
    step (engine M) (storeCode BK pointer value) u=some (scatter M BK r off v u) := by
  simp only [storeCode,step,evalExpr,evalComputeOp,refOp,pointerOp,maskOp,FP.Structural.evalOp,
    FloatDType.toTileDType,hv,store,hp,hm]
  rfl

#axiomsClean forward_body
#axiomsClean prepare_body
#axiomsClean backward_body
#axiomsClean gate_load
#axiomsClean store_eval
end
end VeriTile.Bench.Repairs.DecayCumsum.FPOps
