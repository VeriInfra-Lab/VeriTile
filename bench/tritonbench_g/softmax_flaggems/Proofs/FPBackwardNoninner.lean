import bench.tritonbench_g.softmax_flaggems.Proofs.FPLoop
import Mathlib.Tactic.Ring

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPBackwardNoninner
open VeriTile Triton FP.Structural FPReference FPLoop
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def code (P G Y : RegionName) (rows N K B C : Nat) : List ComputeStmt :=
  branch (Source.softmax_backward_kernel_non_inner_fp32 P G Y rows N K B C Bool.false) 3

def offset {B C : Nat} (row col N K cur : Nat) (i : TileIndex [B,C]) :=
  row*N*K+(cur+i.1.val)*K+(col*C+i.2.1.val)

def active {B C : Nat} (row col rows N K cur : Nat) (i : TileIndex [B,C]) :=
  decide (cur+i.1.val<N) && decide (col*C+i.2.1.val<K)

def load (M : Algebra α) (base : State α) (P : RegionName)
    (row col rows N K B C cur : Nat) (i : TileIndex [B,C]) : α :=
  M.fp32Load (if active row col rows N K cur i then FPReference.read base P (offset row col N K cur i)
    else literal M 0)

def accumulate (M : Algebra α) (base : State α) (P G : RegionName)
    (row col rows N K B C cur : Nat) (scale : Values α .real [B,C]) : Values α .real [B,C] :=
  fun i => add M (scale i)
    (mul M (load M base P row col rows N K B C cur i) (load M base G row col rows N K B C cur i))

def Ready (base : State α) (row col N K B C cur : Nat)
    (scale : Values α .real [B,C]) (s : State α) : Prop :=
  s.regs .int [] "pid_m" = some (fun _ => ((row : Nat) : Int)) ∧
  s.regs .int [C] "offsets_k" = some (fun i => ((col*C+i.1.val : Nat) : Int)) ∧
  s.regs .int [B] "offsets_n" = some (fun i => ((cur+i.1.val : Nat) : Int)) ∧
  s.regs .int [B,C] "offsets" = some (fun i=>((offset row col N K cur i : Nat) : Int)) ∧
  s.regs .real [B,C] "scale" = some scale ∧ s.mem=base.mem

theorem accumulate_step (M : Algebra α) (base s : State α) (P G Y : RegionName)
    (row col rows N K B C cur : Nat) (scale : Values α .real [B,C])
    (hs : Ready base row col N K B C cur scale s) :
    ∃ t, run M (rangeBody (code P G Y rows N K B C) 3)
      (s.setReg "_" .int [] (fun _ => ((cur : Nat) : Int))) = some t ∧
      Ready base row col N K B C (cur+B) (accumulate M base P G row col rows N K B C cur scale) t := by
  rcases hs with ⟨hpid,hm,hn,ho,hscale,hmem⟩
  simp [rangeBody, code, branch, Source.softmax_backward_kernel_non_inner_fp32,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner_kernel,
    ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    numeric, numericLt, numericLe, natLt, natLe, bop, ofFloat, toFloat,
    Region.cast, ComputeDType.eraseDType, State.setReg, Option.bind,
    hpid,hm,hn,ho,hscale,hmem,Ready,accumulate,load,active,offset,
    literal,add,mul,FPReference.read,Int.add_assoc,Int.add_left_comm,Int.add_comm,Int.add_mul,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm,Nat.add_mul]

  try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
  refine ⟨?_,?_,?_⟩ <;> funext i
  all_goals simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,offset,accumulate,load,active,
    add,mul,literal,FPReference.read,Int.add_assoc,Int.add_left_comm,Int.add_comm,Int.add_mul,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm,Nat.add_mul] <;> first | rfl | (rcases i with ⟨a,b,u⟩; cases u; rfl) | ring

def storedValue (M : Algebra α) (base : State α) (P G : RegionName)
    (row col rows N K B C cur : Nat) (scale : Values α .real [C]) (i : TileIndex [B,C]) : α :=
  (mul M (load M base P row col rows N K B C cur i)
    (sub M (load M base G row col rows N K B C cur i) (scale ⟨i.2.1,PUnit.unit⟩)))

def emit (M : Algebra α) (P G Y : RegionName) (row col rows N K B C : Nat)
    (scale : Values α .real [C]) (cur : Nat) (base : State α) : State α :=
  scatter base Y (offset row col N K cur) (active row col rows N K cur)
    (storedValue M base P G row col rows N K B C cur scale)

def WriteReady (base : State α) (row col N K B C cur : Nat)
    (scale : Values α .real [C]) (s : State α) : Prop :=
  s.regs .int [] "pid_m" = some (fun _ => ((row : Nat) : Int)) ∧
  s.regs .int [C] "offsets_k" = some (fun i => ((col*C+i.1.val : Nat) : Int)) ∧
  s.regs .int [B] "offsets_n" = some (fun i => ((cur+i.1.val : Nat) : Int)) ∧
  s.regs .int [B,C] "offsets" = some (fun i=>((offset row col N K cur i : Nat) : Int)) ∧
  s.regs .real [C] "scale" = some scale ∧ s.mem=base.mem

theorem emit_step (M : Algebra α) (base s : State α) (P G Y : RegionName)
    (row col rows N K B C cur : Nat) (scale : Values α .real [C])
    (hs : WriteReady base row col N K B C cur scale s) :
    ∃ t, run M (rangeBody (code P G Y rows N K B C) 7)
      (s.setReg "_" .int [] (fun _ => ((cur : Nat) : Int))) = some t ∧
      WriteReady (emit M P G Y row col rows N K B C scale cur base) row col N K B C (cur+B) scale t := by
  rcases hs with ⟨hpid,hm,hn,ho,hscale,hmem⟩
  simp [rangeBody, code, branch, Source.softmax_backward_kernel_non_inner_fp32,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner_kernel,
    ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold, store,
    numeric, numericLt, numericLe, natLt, natLe, bop, ofFloat, toFloat,
    Region.cast, ComputeDType.eraseDType, State.setReg, Option.bind,
    hpid,hm,hn,ho,hscale,hmem,WriteReady,emit,storedValue,load,active,offset,
    literal,add,mul,sub,FPReference.cast,FPReference.read,scatter]
  try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
  refine ⟨?_,?_,?_⟩
  · funext i; simp [bop,Int.add_assoc,Int.add_left_comm,Int.add_comm,Int.add_mul,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm,Nat.add_mul] <;> first | rfl | (rcases i with ⟨a,b,u⟩; cases u; rfl) | ring
  · funext i; simp [bop,offset,Int.add_assoc,Int.add_left_comm,Int.add_comm,Int.add_mul,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm,Nat.add_mul] <;> first | rfl | (rcases i with ⟨a,b,u⟩; cases u; rfl) | ring
  · apply writeFold_mem_congr
    rfl

def initial (M : Algebra α) (s : State α) (N K B C : Nat) : State α :=
  let s₁ := s.setReg "pid_m" .int [] (fun _ => ((s.pids 0 : Nat) : Int))
  let s₂ := s₁.setReg "pid_k" .int [] (fun _ => ((s.pids 1 : Nat) : Int))
  let s₃ := s₂.setReg "offsets_k" .int [C] (fun i => ((s.pids 1*C+i.1.val : Nat) : Int))
  let s₄ := s₃.setReg "offsets_n" .int [B] (fun i => ((i.1.val : Nat) : Int))
  let s₅ := s₄.setReg "offsets" .int [B,C] (fun i=>((offset (s.pids 0) (s.pids 1) N K 0 i : Nat) : Int))
  s₅.setReg "scale" .real [B,C] (fun _ => literal M 0)

theorem initial_ready (M : Algebra α) (s : State α) (N K B C : Nat) :
    Ready s (s.pids 0) (s.pids 1) N K B C 0 (fun _ => literal M 0) (initial M s N K B C) := by
  simp [Ready,initial,State.setReg]

theorem initial_exec (M : Algebra α) (s : State α) (P G Y : RegionName)
    (rows N K B C : Nat) :
    exec M (Source.softmax_backward_kernel_non_inner_fp32 P G Y rows N K B C Bool.false) s =
    (rangeInt M "_" (0 : Int) (N : Int) B (rangeBody (code P G Y rows N K B C) 3)
      (initial M s N K B C)).bind (run M ((code P G Y rows N K B C).drop 4)) := by
  simp [exec,run,step,Source.softmax_backward_kernel_non_inner_fp32,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner_kernel,code,branch,rangeBody,
    ComputeKernel.surfaceBody,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,numericLt,natLt,bop,ofFloat,toFloat,Region.cast,ComputeDType.eraseDType,
    State.setReg,initial,offset,literal,List.drop_succ_cons,List.drop_zero]
  try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
  delta offset bop
  simp [Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_assoc,Option.bind_some]
  try simp only [Option.bind_some,←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
  rfl

def reduced (M : Algebra α) (B C : Nat) (v : Values α .real [B,C]) : Values α .real [C] :=
  M.reduceSum (some .fp32) (shape := [B,C]) ⟨0,by simp⟩ Bool.false v

def writeInitial (M : Algebra α) (s : State α) (row col N K B C : Nat)
    (v : Values α .real [B,C]) : State α :=
  let s₁ := s.setReg "scale" .real [C] (reduced M B C v)
  let s₂ := s₁.setReg "offsets_n" .int [B] (fun i => ((i.1.val : Nat) : Int))
  s₂.setReg "offsets" .int [B,C] (fun i=>((offset row col N K 0 i : Nat) : Int))

theorem after_accumulate (M : Algebra α) (base mid : State α) (P G Y : RegionName)
    (row col rows N K B C cur : Nat) (v : Values α .real [B,C])
    (hs : Ready base row col N K B C cur v mid) :
    run M ((code P G Y rows N K B C).drop 4) mid =
      rangeInt M "_" (0 : Int) (N : Int) B (rangeBody (code P G Y rows N K B C) 7)
        (writeInitial M mid row col N K B C v) ∧
    WriteReady base row col N K B C 0 (reduced M B C v) (writeInitial M mid row col N K B C v) := by
  rcases hs with ⟨hpid,hm,hn,ho,hv,hmem⟩
  simp [code,branch,rangeBody,Source.softmax_backward_kernel_non_inner_fp32,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.backward_noninner_kernel,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,bop,ofFloat,toFloat,
    Region.cast,ComputeDType.eraseDType,State.setReg,hpid,hm,hv,hmem,
    writeInitial,reduced,offset,WriteReady]
  try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
  delta offset bop
  simp [Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_some]
  simp only [Option.bind, hpid, hm, String.reduceEq, ↓reduceIte, ↓reduceDIte]
  simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]

def graph (M : Algebra α) (base : State α) (P G Y : RegionName)
    (row col rows N K B C : Nat) : State α :=
  let v := (walk (accumulate M base P G row col rows N K B C) N B 0 (fun _ => literal M 0)).2
  (walk (emit M P G Y row col rows N K B C (reduced M B C v)) N B 0 base).2

theorem complete (M : Algebra α) (s : State α) (P G Y : RegionName)
    (rows N K B C : Nat) :
    ∃ final, exec M (Source.softmax_backward_kernel_non_inner_fp32 P G Y rows N K B C Bool.false) s = some final ∧
      final.mem = (graph M s P G Y (s.pids 0) (s.pids 1) rows N K B C).mem := by
  let f := accumulate M s P G (s.pids 0) (s.pids 1) rows N K B C
  obtain ⟨mid,he,hi⟩ := range_model_int M "_" (rangeBody (code P G Y rows N K B C) 3)
    f (fun cur v st => Ready s (s.pids 0) (s.pids 1) N K B C cur v st) N B
    (fun cur v st _ hs => accumulate_step M s st P G Y (s.pids 0) (s.pids 1) rows N K B C cur v hs)
    0 (fun _ => literal M 0) (initial M s N K B C) (initial_ready M s N K B C)
  obtain ⟨htail,hready⟩ := after_accumulate M s mid P G Y (s.pids 0) (s.pids 1) rows N K B C
    (walk f N B 0 (fun _ => literal M 0)).1 (walk f N B 0 (fun _ => literal M 0)).2 hi
  let scale := reduced M B C (walk f N B 0 (fun _ => literal M 0)).2
  obtain ⟨final,hfinal,hmem⟩ := range_model_int M "_" (rangeBody (code P G Y rows N K B C) 7)
    (emit M P G Y (s.pids 0) (s.pids 1) rows N K B C scale)
    (fun cur base st => WriteReady base (s.pids 0) (s.pids 1) N K B C cur scale st) N B
    (fun cur base st _ hs => emit_step M base st P G Y (s.pids 0) (s.pids 1) rows N K B C cur scale hs)
    0 s (writeInitial M mid (s.pids 0) (s.pids 1) N K B C (walk f N B 0 (fun _ => literal M 0)).2) hready
  refine ⟨final,?_,hmem.2.2.2.2.2⟩
  rw [initial_exec]
  exact (congrArg (fun result : Option (State α)=>result.bind
    (run M ((code P G Y rows N K B C).drop 4))) he).trans
      (by simpa only [Option.bind_some] using htail.trans hfinal)

#axiomsClean accumulate_step
#axiomsClean emit_step
#axiomsClean complete
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPBackwardNoninner
