import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPStep
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.Geometry

namespace VeriTile.Bench.Repairs.DecayCumsum.FPPrefix
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPOps Geometry
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def ids : List ComputeStmt :=
  [.assign .int [] "i_k" (.alg (Op.programId 0).castNatToInt),
   .assign .int [] "i_c" (.alg (Op.programId 1).castNatToInt),
   .assign .int [] "i_bh" (.alg (Op.programId 2).castNatToInt)]
def headOp (h : Call) : Op .int [] :=
  .mul .int .nil (.ref .int [] "i_bh") (Op.constNat h.headStride).castNatToInt
def featureOp (h : Call) : Op .int [] :=
  .mul .int .nil (.ref .int [] "i_k") (Op.constNat h.BK).castNatToInt
def chunkOp (p : Phase) : Op .int [] :=
  .mul .int .nil (.ref .int [] "i_c") (Op.constNat p.span).castNatToInt
def lastOp (p : Phase) : Op .int [] :=
  .sub .int .nil (.add .int .nil (chunkOp p) (Op.constNat p.span).castNatToInt) (.constInt 1)
def forwardPointer (h : Call) (p : Phase) (r : RegionName) : Op .ptr [h.BK] :=
  .ptrAdd .scalarL (.ptrBase r)
    (Op.add .int .scalarL
      (.add .int .nil (.add .int .nil (headOp h) (.mul .int .nil (chunkOp p) (Op.constNat h.width).castNatToInt))
        (featureOp h)) (Op.arange h.BK).castNatToInt).castIntToNat
def lastOffset (h : Call) (p : Phase) : Op .nat [h.BK] :=
  (Op.add .int .scalarL
    (.add .int .nil (.add .int .nil (headOp h) (.mul .int .nil (lastOp p) (Op.constNat h.width).castNatToInt))
      (featureOp h)) (Op.arange h.BK).castNatToInt).castIntToNat
def backwardPointer (h : Call) (p : Phase) (r : RegionName) : Op .ptr [h.BK] :=
  .ptrAdd .scalarL (.ptrBase r)
    (Op.add .int .scalarR
      (.add .int .scalarL (.add .int .nil (headOp h) (featureOp h)) (Op.arange h.BK).castNatToInt)
      (.mul .int .nil (lastOp p) (Op.constNat h.width).castNatToInt)).castIntToNat
def pcode (h : Call) (name : RegName) (value : Op .ptr [h.BK]) : ComputeStmt :=
  .assign .ptr [h.BK] name (.alg value)
def zeroCode (BK : Nat) (name : RegName) : ComputeStmt :=
  .assign .real [BK] name (.compute (.full [BK] (.alg .fp32 (.const 0))))
def maskCode (h : Call) : ComputeStmt :=
  .assign .bool [h.BK] "mask" (.alg (.lt .int .scalarR
    (.add .int .scalarL (featureOp h) (Op.arange h.BK).castNatToInt) (Op.constNat h.width).castNatToInt))
def prefixSource (kind : Kind) (h : Call) (p : Phase) : List ComputeStmt :=
  match kind with
  | .forward => ids ++ [pcode h "p_g" (forwardPointer h p "g"),pcode h "p_go" (forwardPointer h p "go"),
      zeroCode h.BK "cum_decay",maskCode h]
  | .prepare => ids ++ [pcode h "p_q" (forwardPointer h p "q"),pcode h "p_g" (forwardPointer h p "g"),
      pcode h "p_k" (forwardPointer h p "k"),pcode h "p_qg" (forwardPointer h p "qg"),
      pcode h "p_kg" (forwardPointer h p "kg"),maskCode h,
      .assign .real [h.BK] "last_decay" (.compute (.load .fp32 (.region "g" (lastOffset h p)) .none))]
  | .backward => ids ++ [pcode h "p_q" (backwardPointer h p "q"),pcode h "p_k" (backwardPointer h p "k"),
      pcode h "p_g" (backwardPointer h p "g"),pcode h "p_dg" (backwardPointer h p "dg"),
      pcode h "p_dq_inner" (backwardPointer h p "dqi"),pcode h "p_dk_inner" (backwardPointer h p "dki"),
      pcode h "p_dq_inter" (backwardPointer h p "dqo"),pcode h "p_dk_inter" (backwardPointer h p "dko"),
      zeroCode h.BK "cum_grad_dg",maskCode h,zeroCode h.BK "last_g"]

theorem source_init (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) :
    FPProgram.initCode kind h p scale=prefixSource kind h p := by cases kind <;> rfl

def off (h : Call) (p : Phase) (prog : Program h p) (t : Nat) (j : Fin h.BK) : Nat :=
  prog.row*h.headStride+prog.chunk*p.span*h.width+prog.feature*h.BK+j.val+t*h.width

def initialIds (h : Call) (p : Phase) (prog : Program h p) (u : State α) : State α :=
  ((u.setReg "i_k" .int [] (fun _ => (prog.feature : Int))).setReg "i_c" .int [] (fun _ => (prog.chunk : Int))).setReg
    "i_bh" .int [] (fun _ => (prog.row : Int))

def initial (M : Algebra α) (kind : Kind) (h : Call) (p : Phase)
    (prog : Program h p) (u : State α) : State α :=
  let v := initialIds h p prog u
  let addr := off h p prog 0
  let last := fun j : Fin h.BK => ((prog.row : Int)*h.headStride+((prog.chunk : Int)*p.span+p.span-1)*h.width+(prog.feature : Int)*h.BK+j.val).toNat
  let back := fun j : Fin h.BK => ((prog.row : Int)*h.headStride+(prog.feature : Int)*h.BK+j.val+((prog.chunk : Int)*p.span+p.span-1)*h.width).toNat
  let mask : Values α .bool [h.BK] := fun i => decide (prog.feature*h.BK+i.1.val<h.width)
  match kind with
  | .forward =>
      let v := v.setReg "p_g" .ptr [h.BK] (ptr h.BK "g" addr)
      let v := v.setReg "p_go" .ptr [h.BK] (ptr h.BK "go" addr)
      let v := v.setReg "cum_decay" .real [h.BK] (fun _ => FPReference.zero M)
      v.setReg "mask" .bool [h.BK] mask
  | .prepare =>
      let v := v.setReg "p_q" .ptr [h.BK] (ptr h.BK "q" addr)
      let v := v.setReg "p_g" .ptr [h.BK] (ptr h.BK "g" addr)
      let v := v.setReg "p_k" .ptr [h.BK] (ptr h.BK "k" addr)
      let v := v.setReg "p_qg" .ptr [h.BK] (ptr h.BK "qg" addr)
      let v := v.setReg "p_kg" .ptr [h.BK] (ptr h.BK "kg" addr)
      let v := v.setReg "mask" .bool [h.BK] mask
      v.setReg "last_decay" .real [h.BK] (fun i => M.fp32Load (raw h.BK "g" last u i))
  | .backward =>
      let v := v.setReg "p_q" .ptr [h.BK] (ptr h.BK "q" back)
      let v := v.setReg "p_k" .ptr [h.BK] (ptr h.BK "k" back)
      let v := v.setReg "p_g" .ptr [h.BK] (ptr h.BK "g" back)
      let v := v.setReg "p_dg" .ptr [h.BK] (ptr h.BK "dg" back)
      let v := v.setReg "p_dq_inner" .ptr [h.BK] (ptr h.BK "dqi" back)
      let v := v.setReg "p_dk_inner" .ptr [h.BK] (ptr h.BK "dki" back)
      let v := v.setReg "p_dq_inter" .ptr [h.BK] (ptr h.BK "dqo" back)
      let v := v.setReg "p_dk_inter" .ptr [h.BK] (ptr h.BK "dko" back)
      let v := v.setReg "cum_grad_dg" .real [h.BK] (fun _ => FPReference.zero M)
      let v := v.setReg "mask" .bool [h.BK] mask
      v.setReg "last_g" .real [h.BK] (fun _ => FPReference.zero M)

theorem pointer_eval (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (r : RegionName) (u : State α)
    (hf : u.regs .int [] "i_k"=some (fun _ => (prog.feature : Int)))
    (hc : u.regs .int [] "i_c"=some (fun _ => (prog.chunk : Int)))
    (hr : u.regs .int [] "i_bh"=some (fun _ => (prog.row : Int))) :
    FP.Structural.evalOp (engine M) none (forwardPointer h p r) u=some (ptr h.BK r (off h p prog 0)) := by
  simp only [forwardPointer,headOp,featureOp,chunkOp,FP.Structural.evalOp,hf,hc,hr]
  apply congrArg some
  funext i
  simp [ptr,off,numeric,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]

theorem backward_pointer_eval (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (r : RegionName) (u : State α)
    (hf : u.regs .int [] "i_k"=some (fun _ => (prog.feature : Int)))
    (hc : u.regs .int [] "i_c"=some (fun _ => (prog.chunk : Int)))
    (hr : u.regs .int [] "i_bh"=some (fun _ => (prog.row : Int))) :
    FP.Structural.evalOp (engine M) none (backwardPointer h p r) u=
      some (ptr h.BK r (fun j => ((prog.row : Int)*h.headStride+(prog.feature : Int)*h.BK+j.val+((prog.chunk : Int)*p.span+p.span-1)*h.width).toNat)) := by
  simp only [backwardPointer,headOp,featureOp,chunkOp,lastOp,FP.Structural.evalOp,hf,hc,hr]
  apply congrArg some
  funext i
  simp [ptr,numeric,bop,Broadcast.leftIndex,Broadcast.rightIndex]

theorem mask_eval (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (u : State α)
    (hf : u.regs .int [] "i_k"=some (fun _ => (prog.feature : Int))) :
    step (engine M) (maskCode h) u=some (u.setReg "mask" .bool [h.BK]
      (fun i => decide (prog.feature*h.BK+i.1.val<h.width))) := by
  simp only [maskCode,featureOp,step,evalExpr,FP.Structural.evalOp,hf]
  simp [numericLt,numeric,bop,Broadcast.leftIndex,Broadcast.rightIndex,
    ← Int.natCast_mul, ← Int.natCast_add]
  apply congrArg (u.setReg "mask" .bool [h.BK])
  funext i
  change decide ((prog.feature : Int)*h.BK+i.1.val<(h.width : Int)) = decide (prog.feature*h.BK+i.1.val<h.width)
  simp only [← Int.natCast_mul, ← Int.natCast_add, Int.ofNat_lt]

theorem zero_eval (M : Algebra α) (BK : Nat) (name : RegName) (u : State α) :
    step (engine M) (zeroCode BK name) u=some (u.setReg name .real [BK] (fun _ => FPReference.zero M)) := by
  simp only [zeroCode,step,evalExpr,evalComputeOp,FP.Structural.evalOp]
  rfl

theorem ids_eval (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (u : State α)
    (hf : u.pids 0=prog.feature) (hc : u.pids 1=prog.chunk) (hr : u.pids 2=prog.row) :
    run (engine M) ids u=some (initialIds h p prog u) := by
  simp [ids,step,run,evalExpr,FP.Structural.evalOp,hf,hc,hr,initialIds]

theorem last_eval (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (u : State α)
    (hf : u.regs .int [] "i_k"=some (fun _ => (prog.feature : Int)))
    (hc : u.regs .int [] "i_c"=some (fun _ => (prog.chunk : Int)))
    (hr : u.regs .int [] "i_bh"=some (fun _ => (prog.row : Int))) :
    step (engine M) (.assign .real [h.BK] "last_decay"
      (.compute (.load .fp32 (.region "g" (lastOffset h p)) .none))) u=
      some (u.setReg "last_decay" .real [h.BK] (fun i => M.fp32Load (raw h.BK "g"
        (fun j => ((prog.row : Int)*h.headStride+((prog.chunk : Int)*p.span+p.span-1)*h.width+(prog.feature : Int)*h.BK+j.val).toNat) u i))) := by
  simp only [lastOffset,headOp,featureOp,chunkOp,lastOp,step,evalExpr,evalComputeOp,FP.Structural.evalOp,hf,hc,hr]
  congr 2

open VeriTile.Bench.Repairs.ChunkCumsumVector.FPStep (assign_alg)
theorem executes (M : Algebra α) (kind : Kind) (h : Call) (p : Phase)
    (prog : Program h p) (u : State α)
    (hf : u.pids 0=prog.feature) (hc : u.pids 1=prog.chunk) (hr : u.pids 2=prog.row) :
    run (engine M) (prefixSource kind h p) u=some (initial M kind h p prog u) := by
  cases kind
  case forward =>
    unfold prefixSource pcode
    rw [run_append,ids_eval M h p prog u hf hc hr]
    simp only [bind,Option.bind_some]
    rw [assign_alg _ _ _ _ _ _ _ _ (pointer_eval M h p prog "g" _ (by simp [initialIds]) (by simp [initialIds]) (by simp [initialIds]))]
    rw [assign_alg _ _ _ _ _ _ _ _ (pointer_eval M h p prog "go" _ (by simp [initialIds]) (by simp [initialIds]) (by simp [initialIds]))]
    rw [run,zero_eval M h.BK "cum_decay"]
    simp only [bind,Option.bind_some]
    rw [run,mask_eval M h p prog _ (by simp [initialIds])]
    simp only [bind,Option.bind_some]
    simp only [bind,Option.bind_some,run]
    rfl

  case prepare =>
    unfold prefixSource pcode
    rw [run_append,ids_eval M h p prog u hf hc hr]
    simp only [bind,Option.bind_some]
    rw [assign_alg _ _ _ _ _ _ _ _ (pointer_eval M h p prog "q" _ (by simp [initialIds]) (by simp [initialIds]) (by simp [initialIds]))]
    rw [assign_alg _ _ _ _ _ _ _ _ (pointer_eval M h p prog "g" _ (by simp [initialIds]) (by simp [initialIds]) (by simp [initialIds]))]
    rw [assign_alg _ _ _ _ _ _ _ _ (pointer_eval M h p prog "k" _ (by simp [initialIds]) (by simp [initialIds]) (by simp [initialIds]))]
    rw [assign_alg _ _ _ _ _ _ _ _ (pointer_eval M h p prog "qg" _ (by simp [initialIds]) (by simp [initialIds]) (by simp [initialIds]))]
    rw [assign_alg _ _ _ _ _ _ _ _ (pointer_eval M h p prog "kg" _ (by simp [initialIds]) (by simp [initialIds]) (by simp [initialIds]))]
    rw [run,mask_eval M h p prog _ (by simp [initialIds])]
    simp only [bind,Option.bind_some]
    rw [run,last_eval M h p prog _ (by simp [initialIds]) (by simp [initialIds]) (by simp [initialIds])]
    simp only [bind,Option.bind_some,run]
    rfl

  case backward =>
    unfold prefixSource pcode
    rw [run_append,ids_eval M h p prog u hf hc hr]
    simp only [bind,Option.bind_some]
    rw [assign_alg _ _ _ _ _ _ _ _ (backward_pointer_eval M h p prog "q" _ (by simp [initialIds]) (by simp [initialIds]) (by simp [initialIds]))]
    rw [assign_alg _ _ _ _ _ _ _ _ (backward_pointer_eval M h p prog "k" _ (by simp [initialIds]) (by simp [initialIds]) (by simp [initialIds]))]
    rw [assign_alg _ _ _ _ _ _ _ _ (backward_pointer_eval M h p prog "g" _ (by simp [initialIds]) (by simp [initialIds]) (by simp [initialIds]))]
    rw [assign_alg _ _ _ _ _ _ _ _ (backward_pointer_eval M h p prog "dg" _ (by simp [initialIds]) (by simp [initialIds]) (by simp [initialIds]))]
    rw [assign_alg _ _ _ _ _ _ _ _ (backward_pointer_eval M h p prog "dqi" _ (by simp [initialIds]) (by simp [initialIds]) (by simp [initialIds]))]
    rw [assign_alg _ _ _ _ _ _ _ _ (backward_pointer_eval M h p prog "dki" _ (by simp [initialIds]) (by simp [initialIds]) (by simp [initialIds]))]
    rw [assign_alg _ _ _ _ _ _ _ _ (backward_pointer_eval M h p prog "dqo" _ (by simp [initialIds]) (by simp [initialIds]) (by simp [initialIds]))]
    rw [assign_alg _ _ _ _ _ _ _ _ (backward_pointer_eval M h p prog "dko" _ (by simp [initialIds]) (by simp [initialIds]) (by simp [initialIds]))]
    rw [run,zero_eval M h.BK "cum_grad_dg"]
    simp only [bind,Option.bind_some]
    rw [run,mask_eval M h p prog _ (by simp [initialIds])]
    simp only [bind,Option.bind_some]
    rw [run,zero_eval M h.BK "last_g"]
    simp only [bind,Option.bind_some,run]
    rfl

#axiomsClean source_init
#axiomsClean executes
end
end VeriTile.Bench.Repairs.DecayCumsum.FPPrefix
