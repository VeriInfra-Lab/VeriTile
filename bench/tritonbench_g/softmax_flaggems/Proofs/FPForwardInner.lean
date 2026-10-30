import bench.tritonbench_g.softmax_flaggems.Proofs.FPConstants

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPForwardInner
open VeriTile Triton FP.Structural FPReference FPLoop
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.FullSource
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def code (Y X : RegionName) (rows N B : Nat) :=
  branch (Source.forward_inner Y X rows N B Bool.false) 1

def previous (N B : Nat) := (N+B-1)/B*B-B

def offset {B : Nat} (row N cur : Nat) (i : TileIndex [B]) := row*N+(cur+i.1.val)
def active {B : Nat} (masked : Bool) (N cur : Nat) (i : TileIndex [B]) :=
  !masked || decide (cur+i.1.val<N)
def load (M : Algebra α) (base : State α) (X : RegionName)
    (masked : Bool) (row N B cur : Nat) (i : TileIndex [B]) : α :=
  M.fp32Load (if active masked N cur i then FPReference.read base X (offset row N cur i) else M.negInf)

abbrev Acc (α : Type) (B : Nat) := Values α .real [B] × Values α .real [B]
def accumulate (M : Algebra α) (base : State α) (X : RegionName) (masked : Bool)
    (row N B cur : Nat) (v : Acc α B) : Acc α B :=
  let inp := load M base X masked row N B cur
  let mx := fun i => FPReference.max M (v.1 i) (inp i)
  (mx,fun i => add M (mul M (v.2 i) (exp M (sub M (v.1 i) (mx i)))) (exp M (sub M (inp i) (mx i))))

def Ready (base : State α) (X Y : RegionName) (row N B : Nat) (v : Acc α B) (s : State α) : Prop :=
  s.regs .ptr [] "input_ptr"=some (fun _ => (X,row*N)) ∧
  s.regs .ptr [] "output_ptr"=some (fun _ => (Y,row*N)) ∧
  s.regs .nat [] "previous_multiple"=some (fun _ => previous N B) ∧
  s.regs .real [B] "m"=some v.1 ∧ s.regs .real [B] "z"=some v.2 ∧ s.mem=base.mem

theorem accumulate_step (M : Algebra α) (base s : State α) (X Y : RegionName)
    (masked : Bool) (row rows N B cur : Nat) (v : Acc α B)
    (hs : Ready base X Y row N B v s) :
    ∃ t, run M (rangeBody (code Y X rows N B) (if masked then 6 else 5))
      (s.setReg "start_n" .nat [] (fun _ => cur))=some t ∧
      Ready base X Y row N B (accumulate M base X masked row N B cur v) t := by
  rcases hs with ⟨hx,hy,hprev,hm,hz,hmem⟩
  cases masked <;>
    simp [rangeBody,code,branch,Source.forward_inner,softmax_kernel_inner,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      numeric,numericLt,numericLe,natLt,natLe,bop,ofFloat,toFloat,Region.cast,
      ComputeDType.eraseDType,State.setReg,Option.bind,hx,hy,hprev,hm,hz,hmem,
      Ready,accumulate,load,active,offset,literal,add,mul,sub,exp,FPReference.max,FPReference.read]
  all_goals constructor <;> funext i
  all_goals simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,load,active,offset,
    FPReference.read,add,mul,sub,exp,FPReference.max]

def maxReduced (M : Algebra α) (B : Nat) (v : Acc α B) : α :=
  M.reduceMax (some .fp32) (shape:=[B]) ⟨0,by simp⟩ Bool.false v.1 PUnit.unit

def sumReduced (M : Algebra α) (B : Nat) (v : Acc α B) : α :=
  M.reduceSum (some .fp32) (shape:=[B]) ⟨0,by simp⟩ Bool.false
    (fun i => mul M (v.2 i) (exp M (sub M (v.1 i) (maxReduced M B v)))) PUnit.unit

def storedValue (M : Algebra α) (base : State α) (X : RegionName) (masked : Bool)
    (row N B cur : Nat) (mx z : α) (i : TileIndex [B]) : α :=
  FPReference.cast M (FPReference.div M
    (exp M (sub M (load M base X masked row N B (previous N B-cur) i) mx)) z)

def emit (M : Algebra α) (X Y : RegionName) (masked : Bool) (row N B : Nat)
    (mx z : α) (cur : Nat) (base : State α) : State α :=
  scatter base Y (offset row N (previous N B-cur)) (active masked N (previous N B-cur))
    (storedValue M base X masked row N B cur mx z)

def WriteReady (base : State α) (X Y : RegionName) (row N B : Nat) (mx z : α) (s : State α) : Prop :=
  s.regs .ptr [] "input_ptr"=some (fun _ => (X,row*N)) ∧
  s.regs .ptr [] "output_ptr"=some (fun _ => (Y,row*N)) ∧
  s.regs .nat [] "previous_multiple"=some (fun _ => previous N B) ∧
  s.regs .real [] "m"=some (fun _ => mx) ∧ s.regs .real [] "z"=some (fun _ => z) ∧ s.mem=base.mem

theorem emit_step (M : Algebra α) (base s : State α) (X Y : RegionName)
    (masked : Bool) (row rows N B cur : Nat) (mx z : α)
    (hs : WriteReady base X Y row N B mx z s) :
    ∃ t, run M (rangeBody (code Y X rows N B) (if masked then 11 else 12))
      (s.setReg "start_n" .nat [] (fun _ => cur))=some t ∧
      WriteReady (emit M X Y masked row N B mx z cur base) X Y row N B mx z t := by
  rcases hs with ⟨hx,hy,hprev,hm,hz,hmem⟩
  cases masked <;>
    simp [rangeBody,code,branch,Source.forward_inner,softmax_kernel_inner,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,store,
      numeric,numericLt,numericLe,natLt,natLe,bop,ofFloat,toFloat,Region.cast,
      ComputeDType.eraseDType,State.setReg,Option.bind,hx,hy,hprev,hm,hz,hmem,
      WriteReady,emit,storedValue,load,active,offset,literal,add,mul,sub,exp,
      FPReference.cast,FPReference.div,FPReference.read,scatter]
  · apply unmasked_mem
    rfl
  · apply writeFold_mem_congr
    rfl

def initial (M : Algebra α) (s : State α) (X Y : RegionName) (N B : Nat) : State α :=
  let s₁ := s.setReg "pid_m" .nat [] (fun _ => s.pids 0)
  let s₂ := s₁.setReg "m" .real [B] (fun _ => literal M (-340282346638528859811704183484516925440.0))
  let s₃ := s₂.setReg "z" .real [B] (fun _ => literal M 0)
  let s₄ := s₃.setReg "input_ptr" .ptr [] (fun _ => (X,s.pids 0*N))
  let s₅ := s₄.setReg "output_ptr" .ptr [] (fun _ => (Y,s.pids 0*N))
  s₅.setReg "previous_multiple" .nat [] (fun _ => previous N B)

theorem initial_ready (M : Algebra α) (s : State α) (X Y : RegionName) (N B : Nat) :
    Ready s X Y (s.pids 0) N B ((fun _ => literal M (-340282346638528859811704183484516925440.0)),fun _ => literal M 0)
      (initial M s X Y N B) := by simp [Ready,initial,State.setReg]

theorem initial_exec (M : Algebra α) (s : State α) (X Y : RegionName) (rows N B : Nat) :
    exec M (Source.forward_inner Y X rows N B Bool.false) s =
    (range M "start_n" 0 (previous N B) B (rangeBody (code Y X rows N B) 5) (initial M s X Y N B)).bind
      (run M ((code Y X rows N B).drop 6)) := by
  simp [exec,run,step,Source.forward_inner,softmax_kernel_inner,code,branch,rangeBody,
    ComputeKernel.surfaceBody,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,
    integralFloorDiv,bop,ofFloat,toFloat,Region.cast,ComputeDType.eraseDType,State.setReg,
    initial,literal,previous,List.drop_succ_cons,List.drop_zero]
  delta bop
  simp [Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_assoc,Option.bind_some]
  simp only [FPConstants.decimal_zero]
  all_goals rfl

theorem between_accumulate (M : Algebra α) (base mid : State α) (X Y : RegionName)
    (row rows N B : Nat) (v : Acc α B) (hs : Ready base X Y row N B v mid) :
    run M ((code Y X rows N B).drop 6) mid =
    (range M "start_n" (previous N B) N B (rangeBody (code Y X rows N B) 6) mid).bind
      (run M ((code Y X rows N B).drop 7)) := by
  rcases hs with ⟨hx,hy,hprev,hm,hz,hmem⟩
  simp [code,branch,rangeBody,Source.forward_inner,softmax_kernel_inner,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,hprev,List.drop_succ_cons,List.drop_zero]

def writeInitial (M : Algebra α) (s : State α) (N B : Nat) (v : Acc α B) : State α :=
  let s₁ := s.setReg "m_reduced" .real [] (fun _ => maxReduced M B v)
  let s₂ := s₁.setReg "z" .real [] (fun _ => sumReduced M B v)
  let s₃ := s₂.setReg "m" .real [] (fun _ => maxReduced M B v)
  s₃.setReg "previous_multiple" .nat [] (fun _ => previous N B)

theorem after_accumulate (M : Algebra α) (base mid : State α) (X Y : RegionName)
    (row rows N B : Nat) (hB : 0<B) (v : Acc α B) (hs : Ready base X Y row N B v mid) :
    run M ((code Y X rows N B).drop 7) mid =
    (range M "start_n" 0 B B (rangeBody (code Y X rows N B) 11) (writeInitial M mid N B v)).bind
      (run M ((code Y X rows N B).drop 12)) ∧
    WriteReady base X Y row N B (maxReduced M B v) (sumReduced M B v) (writeInitial M mid N B v) := by
  rcases hs with ⟨hx,hy,hprev,hm,hz,hmem⟩
  simp [code,branch,rangeBody,Source.forward_inner,softmax_kernel_inner,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,bop,ofFloat,toFloat,
    integralFloorDiv,Region.cast,ComputeDType.eraseDType,State.setReg,hx,hy,hprev,hm,hz,hmem,
    writeInitial,maxReduced,sumReduced,previous,WriteReady,TileShape.axisDim,hB,mul,sub,exp,
    List.drop_succ_cons,List.drop_zero]
  delta bop maxReduced
  simp [Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_some]
  simp only [Option.bind,hx,hy,hm,hz,String.reduceEq,↓reduceIte,↓reduceDIte]
  have tile_eta (i : TileIndex [B]) : (i.1,PUnit.unit)=i := by rcases i with ⟨a,u⟩; cases u; rfl
  have scalar_eta (f : Values α .real []) : (fun _ : TileIndex [] => f PUnit.unit)=f := by
    funext i; cases i; rfl
  simp only [Function.comp_apply,tile_eta,scalar_eta]

theorem between_emit (M : Algebra α) (s : State α) (X Y : RegionName) (rows N B : Nat) :
    run M ((code Y X rows N B).drop 12) s =
      range M "start_n" B N B (rangeBody (code Y X rows N B) 12) s := by
  simp [code,branch,rangeBody,Source.forward_inner,softmax_kernel_inner,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,List.drop_succ_cons,List.drop_zero]

def graph (M : Algebra α) (base : State α) (X Y : RegionName) (row N B : Nat) : State α :=
  let seed : Acc α B := ((fun _ => literal M (-340282346638528859811704183484516925440.0)),fun _ => literal M 0)
  let first := (walk (accumulate M base X Bool.false row N B) (previous N B) B 0 seed).2
  let v := (walk (accumulate M base X Bool.true row N B) N B (previous N B) first).2
  let mx := maxReduced M B v
  let z := sumReduced M B v
  let firstOut := (walk (emit M X Y Bool.true row N B mx z) B B 0 base).2
  (walk (emit M X Y Bool.false row N B mx z) N B B firstOut).2

theorem complete (M : Algebra α) (s : State α) (X Y : RegionName)
    (rows N B : Nat) (hB : 0<B) :
    ∃ final, exec M (Source.forward_inner Y X rows N B Bool.false) s=some final ∧
      final.mem=(graph M s X Y (s.pids 0) N B).mem := by
  let row := s.pids 0
  let seed : Acc α B := ((fun _ => literal M (-340282346638528859811704183484516925440.0)),fun _ => literal M 0)
  let f₀ := accumulate M s X Bool.false row N B
  let first := (walk f₀ (previous N B) B 0 seed).2
  obtain ⟨mid₀,he₀,hi₀⟩ := range_model M "start_n" (rangeBody (code Y X rows N B) 5)
    f₀ (fun _ v st => Ready s X Y row N B v st) (previous N B) B
    (fun cur v st _ hs => accumulate_step M s st X Y Bool.false row rows N B cur v hs)
    0 seed (initial M s X Y N B) (initial_ready M s X Y N B)
  have bridge₀ := between_accumulate M s mid₀ X Y row rows N B first hi₀
  let f₁ := accumulate M s X Bool.true row N B
  let v := (walk f₁ N B (previous N B) first).2
  obtain ⟨mid₁,he₁,hi₁⟩ := range_model M "start_n" (rangeBody (code Y X rows N B) 6)
    f₁ (fun _ v st => Ready s X Y row N B v st) N B
    (fun cur v st _ hs => accumulate_step M s st X Y Bool.true row rows N B cur v hs)
    (previous N B) first mid₀ hi₀
  obtain ⟨bridge₁,ready⟩ := after_accumulate M s mid₁ X Y row rows N B hB v hi₁
  let mx := maxReduced M B v
  let z := sumReduced M B v
  let out₀ := (walk (emit M X Y Bool.true row N B mx z) B B 0 s).2
  obtain ⟨mid₂,he₂,hi₂⟩ := range_model M "start_n" (rangeBody (code Y X rows N B) 11)
    (emit M X Y Bool.true row N B mx z)
    (fun _ base st => WriteReady base X Y row N B mx z st) B B
    (fun cur base st _ hs => emit_step M base st X Y Bool.true row rows N B cur mx z hs)
    0 s (writeInitial M mid₁ N B v) ready
  obtain ⟨final,he₃,hi₃⟩ := range_model M "start_n" (rangeBody (code Y X rows N B) 12)
    (emit M X Y Bool.false row N B mx z)
    (fun _ base st => WriteReady base X Y row N B mx z st) N B
    (fun cur base st _ hs => emit_step M base st X Y Bool.false row rows N B cur mx z hs)
    B out₀ mid₂ hi₂
  refine ⟨final,?_,hi₃.2.2.2.2.2⟩
  rw [initial_exec,he₀]
  simp only [Option.bind_some]
  rw [bridge₀,he₁]
  simp only [Option.bind_some]
  rw [bridge₁,he₂]
  simp only [Option.bind_some]
  exact (between_emit M mid₂ X Y rows N B).trans he₃

#axiomsClean complete

#axiomsClean accumulate_step
#axiomsClean emit_step

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPForwardInner
