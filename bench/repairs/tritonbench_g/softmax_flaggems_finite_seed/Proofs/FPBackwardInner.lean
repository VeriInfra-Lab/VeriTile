import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPLoop

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPBackwardInner
open VeriTile Triton FP.Structural FPReference FPLoop
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def code (P G Y : RegionName) (rows N B C : Nat) : List ComputeStmt :=
  branch (Source.softmax_backward_kernel_inner_fp32 P G Y rows N C B Bool.false) 2

def offset {B C : Nat} (row N cur : Nat) (i : TileIndex [C,B]) :=
  (row*C+i.1.val)*N+(cur+i.2.1.val)

def active {B C : Nat} (row rows N cur : Nat) (i : TileIndex [C,B]) :=
  decide (row*C+i.1.val<rows) && decide (cur+i.2.1.val<N)

def load (M : Algebra α) (base : State α) (P : RegionName)
    (row rows N B C cur : Nat) (i : TileIndex [C,B]) : α :=
  M.fp32Load (if active row rows N cur i then FPReference.read base P (offset row N cur i)
    else literal M 0)

def accumulate (M : Algebra α) (base : State α) (P G : RegionName)
    (row rows N B C cur : Nat) (scale : Values α .real [C,B]) : Values α .real [C,B] :=
  fun i => add M (scale i)
    (mul M (load M base P row rows N B C cur i) (load M base G row rows N B C cur i))

def Ready (base : State α) (row N B C cur : Nat)
    (scale : Values α .real [C,B]) (s : State α) : Prop :=
  s.regs .nat [C] "m_offsets" = some (fun i => row*C+i.1.val) ∧
  s.regs .nat [B] "n_offsets" = some (fun i => cur+i.1.val) ∧
  s.regs .nat [C,B] "offsets" = some (offset row N cur) ∧
  s.regs .real [C,B] "scale" = some scale ∧ s.mem=base.mem

theorem accumulate_step (M : Algebra α) (base s : State α) (P G Y : RegionName)
    (row rows N B C cur : Nat) (scale : Values α .real [C,B])
    (hs : Ready base row N B C cur scale s) :
    ∃ t, run M (rangeBody (code P G Y rows N B C) 3)
      (s.setReg "unused_start_n" .nat [] (fun _ => cur)) = some t ∧
      Ready base row N B C (cur+B) (accumulate M base P G row rows N B C cur scale) t := by
  rcases hs with ⟨hm,hn,ho,hscale,hmem⟩
  simp [rangeBody, code, branch, Source.softmax_backward_kernel_inner_fp32,
    ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    numeric, numericLt, numericLe, natLt, natLe, bop, ofFloat, toFloat,
    Region.cast, ComputeDType.eraseDType, State.setReg, Option.bind,
    hm,hn,ho,hscale,hmem,Ready,accumulate,load,active,offset,
    literal,add,mul,FPReference.read,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]

  refine ⟨?_,?_,?_⟩ <;> funext i
  all_goals simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,offset,accumulate,load,active,
    add,mul,literal,FPReference.read,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]

def storedValue (M : Algebra α) (base : State α) (P G : RegionName)
    (row rows N B C cur : Nat) (scale : Values α .real [C]) (i : TileIndex [C,B]) : α :=
  FPReference.cast M (mul M (load M base P row rows N B C cur i)
    (sub M (load M base G row rows N B C cur i) (scale ⟨i.1,PUnit.unit⟩)))

def emit (M : Algebra α) (P G Y : RegionName) (row rows N B C : Nat)
    (scale : Values α .real [C]) (cur : Nat) (base : State α) : State α :=
  scatter base Y (offset row N cur) (active row rows N cur)
    (storedValue M base P G row rows N B C cur scale)

def WriteReady (base : State α) (row N B C cur : Nat)
    (scale : Values α .real [C]) (s : State α) : Prop :=
  s.regs .nat [C] "m_offsets" = some (fun i => row*C+i.1.val) ∧
  s.regs .nat [B] "n_offsets" = some (fun i => cur+i.1.val) ∧
  s.regs .nat [C,B] "offsets" = some (offset row N cur) ∧
  s.regs .real [C] "scale" = some scale ∧ s.mem=base.mem

theorem emit_step (M : Algebra α) (base s : State α) (P G Y : RegionName)
    (row rows N B C cur : Nat) (scale : Values α .real [C])
    (hs : WriteReady base row N B C cur scale s) :
    ∃ t, run M (rangeBody (code P G Y rows N B C) 7)
      (s.setReg "unused_start_n" .nat [] (fun _ => cur)) = some t ∧
      WriteReady (emit M P G Y row rows N B C scale cur base) row N B C (cur+B) scale t := by
  rcases hs with ⟨hm,hn,ho,hscale,hmem⟩
  simp [rangeBody, code, branch, Source.softmax_backward_kernel_inner_fp32,
    ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold, store,
    numeric, numericLt, numericLe, natLt, natLe, bop, ofFloat, toFloat,
    Region.cast, ComputeDType.eraseDType, State.setReg, Option.bind,
    hm,hn,ho,hscale,hmem,WriteReady,emit,storedValue,load,active,offset,
    literal,add,mul,sub,FPReference.cast,FPReference.read,scatter]
  refine ⟨?_,?_,?_⟩
  · funext i; simp [bop,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]
  · funext i; simp [bop,offset,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]
  · apply writeFold_mem_congr
    rfl

def initial (M : Algebra α) (s : State α) (N B C : Nat) : State α :=
  let s₁ := s.setReg "pid_m" .nat [] (fun _ => s.pids 0)
  let s₂ := s₁.setReg "m_offsets" .nat [C] (fun i => s.pids 0*C+i.1.val)
  let s₃ := s₂.setReg "scale" .real [C,B] (fun _ => literal M 0)
  let s₄ := s₃.setReg "n_offsets" .nat [B] (fun i => i.1.val)
  s₄.setReg "offsets" .nat [C,B] (offset (s.pids 0) N 0)

theorem initial_ready (M : Algebra α) (s : State α) (N B C : Nat) :
    Ready s (s.pids 0) N B C 0 (fun _ => literal M 0) (initial M s N B C) := by
  simp [Ready,initial,State.setReg]

theorem initial_exec (M : Algebra α) (s : State α) (P G Y : RegionName)
    (rows N B C : Nat) :
    exec M (Source.softmax_backward_kernel_inner_fp32 P G Y rows N C B Bool.false) s =
    (range M "unused_start_n" 0 N B (rangeBody (code P G Y rows N B C) 3)
      (initial M s N B C)).bind (run M ((code P G Y rows N B C).drop 4)) := by
  simp [exec,run,step,Source.softmax_backward_kernel_inner_fp32,code,branch,rangeBody,
    ComputeKernel.surfaceBody,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,numericLt,natLt,bop,ofFloat,toFloat,Region.cast,ComputeDType.eraseDType,
    State.setReg,initial,offset,literal,List.drop_succ_cons,List.drop_zero]
  delta offset bop
  simp [Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_assoc,Option.bind_some]
  rfl

def reduced (M : Algebra α) (B C : Nat) (v : Values α .real [C,B]) : Values α .real [C] :=
  M.reduceSum (some .fp32) (shape := [C,B]) ⟨1,by simp⟩ Bool.false v

def writeInitial (M : Algebra α) (s : State α) (row N B C : Nat)
    (v : Values α .real [C,B]) : State α :=
  let s₁ := s.setReg "scale" .real [C] (reduced M B C v)
  let s₂ := s₁.setReg "n_offsets" .nat [B] (fun i => i.1.val)
  s₂.setReg "offsets" .nat [C,B] (offset row N 0)

theorem after_accumulate (M : Algebra α) (base mid : State α) (P G Y : RegionName)
    (row rows N B C cur : Nat) (v : Values α .real [C,B])
    (hs : Ready base row N B C cur v mid) :
    run M ((code P G Y rows N B C).drop 4) mid =
      range M "unused_start_n" 0 N B (rangeBody (code P G Y rows N B C) 7)
        (writeInitial M mid row N B C v) ∧
    WriteReady base row N B C 0 (reduced M B C v) (writeInitial M mid row N B C v) := by
  rcases hs with ⟨hm,hn,ho,hv,hmem⟩
  simp [code,branch,rangeBody,Source.softmax_backward_kernel_inner_fp32,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,bop,ofFloat,toFloat,
    Region.cast,ComputeDType.eraseDType,State.setReg,hm,hv,hmem,
    writeInitial,reduced,offset,WriteReady]
  delta offset bop
  simp [Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_some]
  simp only [Option.bind, hm, String.reduceEq, ↓reduceIte, ↓reduceDIte]

/-- Pure typed graph: each source accumulation precedes the axis reduction;
writeback rereads the saved tensors in source range order. -/
def graph (M : Algebra α) (base : State α) (P G Y : RegionName)
    (row rows N B C : Nat) : State α :=
  let v := (walk (accumulate M base P G row rows N B C) N B 0 (fun _ => literal M 0)).2
  (walk (emit M P G Y row rows N B C (reduced M B C v)) N B 0 base).2

theorem complete (M : Algebra α) (s : State α) (P G Y : RegionName)
    (rows N B C : Nat) :
    ∃ final, exec M (Source.softmax_backward_kernel_inner_fp32 P G Y rows N C B Bool.false) s = some final ∧
      final.mem = (graph M s P G Y (s.pids 0) rows N B C).mem := by
  let f := accumulate M s P G (s.pids 0) rows N B C
  obtain ⟨mid,he,hi⟩ := range_model M "unused_start_n" (rangeBody (code P G Y rows N B C) 3)
    f (fun cur v st => Ready s (s.pids 0) N B C cur v st) N B
    (fun cur v st _ hs => accumulate_step M s st P G Y (s.pids 0) rows N B C cur v hs)
    0 (fun _ => literal M 0) (initial M s N B C) (initial_ready M s N B C)
  obtain ⟨htail,hready⟩ := after_accumulate M s mid P G Y (s.pids 0) rows N B C
    (walk f N B 0 (fun _ => literal M 0)).1 (walk f N B 0 (fun _ => literal M 0)).2 hi
  let scale := reduced M B C (walk f N B 0 (fun _ => literal M 0)).2
  obtain ⟨final,hfinal,hmem⟩ := range_model M "unused_start_n" (rangeBody (code P G Y rows N B C) 7)
    (emit M P G Y (s.pids 0) rows N B C scale)
    (fun cur base st => WriteReady base (s.pids 0) N B C cur scale st) N B
    (fun cur base st _ hs => emit_step M base st P G Y (s.pids 0) rows N B C cur scale hs)
    0 s (writeInitial M mid (s.pids 0) N B C (walk f N B 0 (fun _ => literal M 0)).2) hready
  refine ⟨final,?_,hmem.2.2.2.2⟩
  rw [initial_exec,he]
  simp only [Option.bind_some]
  exact htail.trans hfinal

#axiomsClean accumulate_step
#axiomsClean emit_step
#axiomsClean complete

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPBackwardInner
