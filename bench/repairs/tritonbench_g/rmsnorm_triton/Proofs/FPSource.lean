import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.FPGraph
import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.TensorSource

namespace VeriTile.Bench.Repairs.RmsnormTriton.Symbolic
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxFlaggems.FPLoop
open VeriTile.Bench.Repairs.SoftmaxFlaggems.FPConstants
open VeriTile.Bench.Repairs.SoftmaxFlaggems.FPReference (scatter writeFold_mem_congr)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def code (p : Parameters) (B : Nat) :=
  (kernel "x" "weights" "out" (p.input.stride 0) (p.input.stride 1) (p.input.stride 2)
    (p.weights.stride 0) (p.output.stride 0) (p.output.stride 1) (p.output.stride 2)
    p.width B p.eps).surfaceBody

def read (s : State Term) (r : RegionName) (offset : Nat) : Term := (s.mem r offset).read .real

def inputAddress (p : Parameters) (b m k : Nat) := b*p.input.stride 0+m*p.input.stride 1+k*p.input.stride 2

def inputValue (s : State Term) (p : Parameters) (b m k : Nat) : Term :=
  if k<p.width then read s "x" (inputAddress p b m k) else lit 0

def accumulate (base : State Term) (p : Parameters) (b m B cur : Nat)
    (value : Values Term .real [B]) : Values Term .real [B] := fun i =>
  add (value i) (square (cast32 (inputValue base p b m (cur+i.1.val))))

def Ready (base : State Term) (p : Parameters) (b m B : Nat)
    (value : Values Term .real [B]) (s : State Term) : Prop :=
  s.regs .nat [] "pid_batch"=some (fun _=>b) ∧
  s.regs .nat [] "pid_m"=some (fun _=>m) ∧
  s.regs .nat [] "offs_m"=some (fun _=>b*p.input.stride 0+m*p.input.stride 1) ∧
  s.regs .nat [B] "block_N"=some (fun i=>i.1.val) ∧
  s.regs .real [B] "var"=some value ∧ s.mem=base.mem ∧ s.undef=base.undef

theorem accumulate_step (base s : State Term) (p : Parameters) (b m B cur : Nat)
    (value : Values Term .real [B]) (hs : Ready base p b m B value s) :
    ∃ t,run algebra (rangeBody (code p B) 5) (s.setReg "block_n_start_idx" .nat [] (fun _=>cur))=some t ∧
      Ready base p b m B (accumulate base p b m B cur value) t := by
  rcases hs with ⟨hb,hm,ho,hN,hv,hmem,hu⟩
  simp [rangeBody,code,kernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,natLt,bop,ofFloat,toFloat,Region.cast,ComputeDType.eraseDType,State.setReg,Option.bind,
    hb,hm,ho,hN,hv,hmem,hu,algebra,Ready,accumulate,inputValue,inputAddress,read,add,square,cast32,lit]
  funext i
  rcases i with ⟨i,u⟩
  cases u
  simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,decimal_zero,accumulate,inputValue,inputAddress,read,add,square,cast32,lit]

def initial (s : State Term) (p : Parameters) (B : Nat) : State Term :=
  let s₁ := s.setReg "pid_batch" .nat [] (fun _=>s.pids 0)
  let s₂ := s₁.setReg "pid_m" .nat [] (fun _=>s.pids 1)
  let s₃ := s₂.setReg "offs_m" .nat [] (fun _=>s.pids 0*p.input.stride 0+s.pids 1*p.input.stride 1)
  let s₄ := s₃.setReg "block_N" .nat [B] (fun i=>i.1.val)
  s₄.setReg "var" .real [B] (fun _=>lit 0)

theorem initial_ready (s : State Term) (p : Parameters) (B : Nat) :
    Ready s p (s.pids 0) (s.pids 1) B (fun _=>lit 0) (initial s p B) := by
  simp [Ready,initial,State.setReg]

theorem initial_exec (s : State Term) (p : Parameters) (B : Nat) :
    run algebra (code p B) s=
      (range algebra "block_n_start_idx" 0 p.width B (rangeBody (code p B) 5) (initial s p B)).bind
        (run algebra ((code p B).drop 6)) := by
  simp [rangeBody,code,kernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,natLt,bop,ofFloat,toFloat,Region.cast,ComputeDType.eraseDType,State.setReg,initial,algebra,lit,
    List.drop_succ_cons,List.drop_zero,decimal_zero]
  delta bop
  simp [Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_assoc,Option.bind_some]

def reduced (B : Nat) (value : Values Term .real [B]) : Term :=
  .sum (some .fp32) B (fun j=>value (j,PUnit.unit))

def standard (p : Parameters) (B : Nat) (value : Values Term .real [B]) : Term :=
  rsqrt (add (divide (reduced B value) (lit p.width)) (lit p.eps))

def writeInitial (s : State Term) (p : Parameters) (B : Nat) (value : Values Term .real [B]) : State Term :=
  let s₁ := s.setReg "var" .real [] (fun _=>divide (reduced B value) (lit p.width))
  s₁.setReg "rstd" .real [] (fun _=>standard p B value)

def WriteReady (base : State Term) (p : Parameters) (b m B : Nat) (std : Term) (s : State Term) : Prop :=
  s.regs .nat [] "pid_batch"=some (fun _=>b) ∧
  s.regs .nat [] "pid_m"=some (fun _=>m) ∧
  s.regs .nat [] "offs_m"=some (fun _=>b*p.input.stride 0+m*p.input.stride 1) ∧
  s.regs .nat [B] "block_N"=some (fun i=>i.1.val) ∧
  s.regs .real [] "rstd"=some (fun _=>std) ∧ s.mem=base.mem ∧ s.undef=base.undef

theorem after_accumulate (base s : State Term) (p : Parameters) (b m B : Nat)
    (value : Values Term .real [B]) (hs : Ready base p b m B value s) :
    run algebra ((code p B).drop 6) s=
      range algebra "block_n_start_idx" 0 p.width B (rangeBody (code p B) 8) (writeInitial s p B value) ∧
    WriteReady base p b m B (standard p B value) (writeInitial s p B value) := by
  rcases hs with ⟨hb,hm,ho,hN,hv,hmem,hu⟩
  simp [code,kernel,rangeBody,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,natLt,bop,ofFloat,toFloat,Region.cast,ComputeDType.eraseDType,State.setReg,
    hb,hm,ho,hN,hv,hmem,hu,algebra,graphSum,writeInitial,WriteReady,standard,reduced,add,divide,rsqrt,lit,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex]
  rfl

def weightValue (s : State Term) (p : Parameters) (k : Nat) : Term :=
  if k<p.width then read s "weights" (k*p.weights.stride 0)
    else s.undef .real "weights" (k*p.weights.stride 0)

def storedValue (s : State Term) (p : Parameters) (b m k : Nat) (std : Term) : Term :=
  mul (mul (cast32 (inputValue s p b m k)) std) (weightValue s p k)

def emit (p : Parameters) (b m B : Nat) (std : Term) (cur : Nat) (base : State Term) : State Term :=
  scatter base "out" (fun i : TileIndex [B]=>TensorSource.rowAddress p b m (cur+i.1.val))
    (fun i : TileIndex [B]=>decide (cur+i.1.val<p.width))
    (fun i=>storedValue base p b m (cur+i.1.val) std)

@[simp] theorem scatter_undef {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (l : List ι) (s : State α) :
    (l.foldl (fun t i => t.write r (off i) (val i)) s).undef=s.undef := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih => exact ih _

@[simp] theorem masked_undef {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active] (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) s).undef=s.undef := by
  rw [State.masked_scatter_eq_filter,scatter_undef]

theorem emit_step (base s : State Term) (p : Parameters) (b m B cur : Nat) (std : Term)
    (hs : WriteReady base p b m B std s) :
    ∃ t,run algebra (rangeBody (code p B) 8) (s.setReg "block_n_start_idx" .nat [] (fun _=>cur))=some t ∧
      WriteReady (emit p b m B std cur base) p b m B std t := by
  rcases hs with ⟨hb,hm,ho,hN,hstd,hmem,hu⟩
  simp [rangeBody,code,kernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,store,
    numeric,natLt,bop,ofFloat,toFloat,Region.cast,ComputeDType.eraseDType,State.setReg,Option.bind,
    hb,hm,ho,hN,hstd,hmem,hu,algebra,WriteReady,emit,storedValue,weightValue,inputValue,inputAddress,
    TensorSource.rowAddress,read,add,mul,cast32,lit,scatter,decimal_zero]
  apply writeFold_mem_congr
  rfl

def graph (s : State Term) (p : Parameters) (B : Nat) : State Term :=
  let v := (walk (accumulate s p (s.pids 0) (s.pids 1) B) p.width B 0 (fun _=>lit 0)).2
  (walk (emit p (s.pids 0) (s.pids 1) B (standard p B v)) p.width B 0 s).2

theorem complete (s : State Term) (p : Parameters) (B : Nat) :
    ∃ final,run algebra (code p B) s=some final ∧ final.mem=(graph s p B).mem := by
  let f := accumulate s p (s.pids 0) (s.pids 1) B
  let seed : Values Term .real [B] := fun _=>lit 0
  obtain ⟨mid,he,hi⟩ := range_model algebra "block_n_start_idx" (rangeBody (code p B) 5)
    f (fun _ v st=>Ready s p (s.pids 0) (s.pids 1) B v st) p.width B
    (fun cur v st _ hs=>accumulate_step s st p (s.pids 0) (s.pids 1) B cur v hs)
    0 seed (initial s p B) (initial_ready s p B)
  obtain ⟨tail,ready⟩ := after_accumulate s mid p (s.pids 0) (s.pids 1) B (walk f p.width B 0 seed).2 hi
  let std := standard p B (walk f p.width B 0 seed).2
  obtain ⟨final,hfinal,hI⟩ := range_model algebra "block_n_start_idx" (rangeBody (code p B) 8)
    (emit p (s.pids 0) (s.pids 1) B std)
    (fun _ base st=>WriteReady base p (s.pids 0) (s.pids 1) B std st) p.width B
    (fun cur base st _ hs=>emit_step base st p (s.pids 0) (s.pids 1) B cur std hs)
    0 s (writeInitial mid p B (walk f p.width B 0 seed).2) ready
  refine ⟨final,?_,hI.2.2.2.2.2.1⟩
  rw [initial_exec,he]
  simp only [Option.bind_some]
  exact tail.trans hfinal

#axiomsClean complete
end
end VeriTile.Bench.Repairs.RmsnormTriton.Symbolic
