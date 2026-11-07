import bench.tritonbench_g.softmax_flaggems.Proofs.FPConstants
import bench.tritonbench_g.softmax_flaggems.Proofs.SignedLoops

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPForwardNoninner
open VeriTile Triton FP.Structural FPReference FPLoop
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.FullSource
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def code (Y X : RegionName) (rows N K B C : Nat) :=
  branch (Source.forward_noninner Y X rows N K B C Bool.false) 3

def offset {B C : Nat} (row col N K cur : Nat) (i : TileIndex [B,C]) :=
  row*N*K+(cur+i.1.val)*K+(col*C+i.2.1.val)
def active {B C : Nat} (col N K cur : Nat) (i : TileIndex [B,C]) :=
  decide (cur+i.1.val<N) && decide (col*C+i.2.1.val<K)
def load (M : Algebra α) (base : State α) (X : RegionName)
    (row col N K B C cur : Nat) (i : TileIndex [B,C]) : α :=
  M.fp32Load (if active col N K cur i then FPReference.read base X (offset row col N K cur i)
    else M.negInf)

abbrev Acc (α : Type) (B C : Nat) := Values α .real [B,C] × Values α .real [B,C]

def accumulate (M : Algebra α) (base : State α) (X : RegionName)
    (row col N K B C cur : Nat) (v : Acc α B C) : Acc α B C :=
  let inp := load M base X row col N K B C cur
  let mx := fun i => FPReference.max M (v.1 i) (inp i)
  (mx, fun i => add M (mul M (v.2 i) (exp M (sub M (v.1 i) (mx i))))
    (exp M (sub M (inp i) (mx i))))

def Ready (base : State α) (row col B C : Nat) (v : Acc α B C) (s : State α) : Prop :=
  s.regs .int [] "pid_m"=some (fun _ => ((row : Nat) : Int)) ∧
  s.regs .int [C] "k_offsets"=some (fun i => ((col*C+i.1.val : Nat) : Int)) ∧
  s.regs .real [B,C] "m"=some v.1 ∧ s.regs .real [B,C] "z"=some v.2 ∧ s.mem=base.mem

theorem accumulate_step (M : Algebra α) (base s : State α) (X Y : RegionName)
    (row col rows N K B C cur : Nat) (v : Acc α B C)
    (hs : Ready base row col B C v s) :
    ∃ t, run M (rangeBody (code Y X rows N K B C) 2)
      (s.setReg "start_n" .int [] (fun _ => ((cur : Nat) : Int)))=some t ∧
      Ready base row col B C (accumulate M base X row col N K B C cur v) t := by
  rcases hs with ⟨hr,hk,hm,hz,hmem⟩
  simp [rangeBody,code,branch,Source.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner_kernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,numericLt,numericLe,natLt,natLe,bop,ofFloat,toFloat,Region.cast,
    ComputeDType.eraseDType,State.setReg,Option.bind,hr,hk,hm,hz,hmem,
    Ready,accumulate,load,active,offset,literal,add,mul,sub,exp,FPReference.max,FPReference.read]
  constructor <;> funext i
  all_goals simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,load,active,offset,
    FPReference.read,add,mul,sub,exp,FPReference.max]
  all_goals simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
  all_goals rcases i with ⟨a,b,u⟩; cases u; rfl

def maxReduced (M : Algebra α) (B C : Nat) (v : Acc α B C) : Values α .real [C] :=
  M.reduceMax (some .fp32) (shape:=[B,C]) ⟨0,by simp⟩ Bool.false v.1

def sumReduced (M : Algebra α) (B C : Nat) (v : Acc α B C) : Values α .real [C] :=
  M.reduceSum (some .fp32) (shape:=[B,C]) ⟨0,by simp⟩ Bool.false
    (fun i => mul M (v.2 i) (exp M (sub M (v.1 i) (maxReduced M B C v ⟨i.2.1,PUnit.unit⟩))))

def previous (N B : Nat) := (N+B-1)/B*B-B

def storedValue (M : Algebra α) (base : State α) (X : RegionName)
    (row col N K B C cur : Nat) (mx z : Values α .real [C]) (i : TileIndex [B,C]) : α :=
  (FPReference.div M
    (exp M (sub M (load M base X row col N K B C (previous N B-cur) i) (mx ⟨i.2.1,PUnit.unit⟩)))
    (z ⟨i.2.1,PUnit.unit⟩))

def emit (M : Algebra α) (X Y : RegionName) (row col N K B C : Nat)
    (mx z : Values α .real [C]) (cur : Nat) (base : State α) : State α :=
  scatter base Y (offset row col N K (previous N B-cur)) (active col N K (previous N B-cur))
    (storedValue M base X row col N K B C cur mx z)

def WriteReady (base : State α) (row col N B C : Nat)
    (mx z : Values α .real [C]) (s : State α) : Prop :=
  s.regs .int [] "pid_m"=some (fun _ => ((row : Nat) : Int)) ∧
  s.regs .int [C] "k_offsets"=some (fun i => ((col*C+i.1.val : Nat) : Int)) ∧
  s.regs .real [C] "m"=some mx ∧ s.regs .real [C] "z"=some z ∧
  s.regs .int [] "previous_multiple"=some (fun _ => ((previous N B : Nat) : Int)) ∧ s.mem=base.mem

theorem emit_step (M : Algebra α) (base s : State α) (X Y : RegionName)
    (row col rows N K B C cur : Nat) (mx z : Values α .real [C])
    (hcur : cur≤previous N B)
    (hs : WriteReady base row col N B C mx z s) :
    ∃ t, run M (rangeBody (code Y X rows N K B C) 10)
      (s.setReg "start_n" .int [] (fun _ => ((cur : Nat) : Int)))=some t ∧
      WriteReady (emit M X Y row col N K B C mx z cur base) row col N B C mx z t := by
  have subCast : (previous N B : Int)-(cur : Int) = ((previous N B-cur : Nat) : Int) :=
    (Int.ofNat_sub hcur).symm
  rcases hs with ⟨hr,hk,hm,hz,hprev,hmem⟩
  simp [rangeBody,code,branch,Source.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner_kernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,store,
    numeric,numericLt,numericLe,natLt,natLe,bop,ofFloat,toFloat,Region.cast,
    ComputeDType.eraseDType,State.setReg,Option.bind,hr,hk,hm,hz,hprev,hmem,
    WriteReady,emit,storedValue,load,active,offset,literal,add,mul,sub,exp,
    FPReference.cast,FPReference.div,FPReference.read,scatter]
  simp only [subCast,←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
  apply writeFold_mem_congr
  rfl

def initial (M : Algebra α) (s : State α) (B C : Nat) : State α :=
  let s₁ := s.setReg "pid_k" .int [] (fun _ => ((s.pids 1 : Nat) : Int))
  let s₂ := s₁.setReg "pid_m" .int [] (fun _ => ((s.pids 0 : Nat) : Int))
  let s₃ := s₂.setReg "k_offsets" .int [C] (fun i => ((s.pids 1*C+i.1.val : Nat) : Int))
  let s₄ := s₃.setReg "m" .real [B,C] (fun _ => literal M (-340282346638528859811704183484516925440.0))
  s₄.setReg "z" .real [B,C] (fun _ => literal M 0)

theorem initial_ready (M : Algebra α) (s : State α) (B C : Nat) :
    Ready s (s.pids 0) (s.pids 1) B C ((fun _ => literal M (-340282346638528859811704183484516925440.0)),fun _ => literal M 0)
      (initial M s B C) := by simp [Ready,initial,State.setReg]

theorem initial_exec (M : Algebra α) (s : State α) (X Y : RegionName)
    (rows N K B C : Nat) :
    exec M (Source.forward_noninner Y X rows N K B C Bool.false) s =
    (rangeInt M "start_n" (0 : Int) (N : Int) B (rangeBody (code Y X rows N K B C) 2) (initial M s B C)).bind
      (run M ((code Y X rows N K B C).drop 3)) := by
  simp [exec,run,step,Source.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner_kernel,code,branch,rangeBody,
    ComputeKernel.surfaceBody,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,
    bop,ofFloat,toFloat,Region.cast,ComputeDType.eraseDType,State.setReg,initial,literal,
    List.drop_succ_cons,List.drop_zero]
  delta bop
  simp [Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_assoc,Option.bind_some]
  simp only [FPConstants.decimal_zero]
  all_goals rfl

def writeInitial (M : Algebra α) (s : State α) (N B C : Nat) (v : Acc α B C) : State α :=
  let s₁ := s.setReg "m_reduced" .real [C] (maxReduced M B C v)
  let s₂ := s₁.setReg "z" .real [C] (sumReduced M B C v)
  let s₃ := s₂.setReg "m" .real [C] (maxReduced M B C v)
  let s₄ := s₃.setReg "__veritile_statement_jit_0" .int [] (fun _ => (N : Int))
  let s₅ := s₄.setReg "__veritile_statement_jit_1" .nat [] (fun _ => B)
  let s₆ := s₅.setReg "__veritile_statement_jit_2" .int [] (fun _ => (previous N B : Int))
  s₆.setReg "previous_multiple" .int [] (fun _ => (previous N B : Int))

theorem after_accumulate (M : Algebra α) (base mid : State α) (X Y : RegionName)
    (row col rows N K B C : Nat) (hN : 0<N) (hB : 0<B) (v : Acc α B C)
    (hs : Ready base row col B C v mid) :
    run M ((code Y X rows N K B C).drop 3) mid =
      rangeInt M "start_n" (0 : Int) (N : Int) B (rangeBody (code Y X rows N K B C) 10) (writeInitial M mid N B C v) ∧
    WriteReady base row col N B C (maxReduced M B C v) (sumReduced M B C v) (writeInitial M mid N B C v) := by
  have hn : ¬ (N : Int)+((B : Int)-1)<0 := by omega
  have hp : ((N : Int)+((B : Int)-1))/(B : Int)*(B : Int)-(B : Int) = (previous N B : Int) := by
    have hsum : (N : Int)+((B : Int)-1) = (N : Int)+(B : Int)-1 := by omega
    rw [hsum]
    exact SignedAddress.previous_eq N B hN hB
  rcases hs with ⟨hr,hk,hm,hz,hmem⟩
  simp [code,branch,rangeBody,Source.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner,VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.forward_noninner_kernel,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,bop,ofFloat,toFloat,
    integralFloorDiv,Region.cast,ComputeDType.eraseDType,State.setReg,hr,hk,hm,hz,hmem,
    writeInitial,maxReduced,sumReduced,previous,WriteReady,TileShape.axisDim,hB,
    mul,sub,exp]
  delta bop maxReduced
  simp [Broadcast.leftIndex,Broadcast.rightIndex,Option.bind_some]
  simp only [Option.bind,hr,hk,hm,hz,String.reduceEq,↓reduceIte,↓reduceDIte]
  have eta (i : TileIndex [B,C]) : (i.1,i.2.1,PUnit.unit) = i := by
    rcases i with ⟨a,b,u⟩; cases u; rfl
  simp only [Function.comp_apply,eta]
  simp only [if_neg hn,hp]
  rfl

def graph (M : Algebra α) (base : State α) (X Y : RegionName)
    (row col N K B C : Nat) : State α :=
  let v := (walk (accumulate M base X row col N K B C) N B 0 ((fun _ => literal M (-340282346638528859811704183484516925440.0)),fun _ => literal M 0)).2
  (walk (emit M X Y row col N K B C (maxReduced M B C v) (sumReduced M B C v)) N B 0 base).2

theorem complete (M : Algebra α) (s : State α) (X Y : RegionName)
    (rows N K B C : Nat) (hN : 0<N) (hB : 0<B) :
    ∃ final, exec M (Source.forward_noninner Y X rows N K B C Bool.false) s = some final ∧
      final.mem=(graph M s X Y (s.pids 0) (s.pids 1) N K B C).mem := by
  let f := accumulate M s X (s.pids 0) (s.pids 1) N K B C
  let seed : Acc α B C := ((fun _ => literal M (-340282346638528859811704183484516925440.0)),fun _ => literal M 0)
  obtain ⟨mid,he,hi⟩ := range_model_int M "start_n" (rangeBody (code Y X rows N K B C) 2)
    f (fun _ v st => Ready s (s.pids 0) (s.pids 1) B C v st) N B
    (fun cur v st _ hs => accumulate_step M s st X Y (s.pids 0) (s.pids 1) rows N K B C cur v hs)
    0 seed (initial M s B C) (initial_ready M s B C)
  obtain ⟨htail,hready⟩ := after_accumulate M s mid X Y (s.pids 0) (s.pids 1) rows N K B C hN hB
    (walk f N B 0 seed).2 hi
  let mx := maxReduced M B C (walk f N B 0 seed).2
  let z := sumReduced M B C (walk f N B 0 seed).2
  obtain ⟨final,hfinal,hmem⟩ := range_int_aligned M "start_n" (rangeBody (code Y X rows N K B C) 10)
    (emit M X Y (s.pids 0) (s.pids 1) N K B C mx z)
    (fun _ base st => WriteReady base (s.pids 0) (s.pids 1) N B C mx z st) N B
    (fun cur base st hc aligned hs => emit_step M base st X Y (s.pids 0) (s.pids 1) rows N K B C cur mx z
      (SignedAddress.tile_le_previous N B cur hN hB hc aligned) hs)
    0 s (writeInitial M mid N B C (walk f N B 0 seed).2) (dvd_zero B) hready
  refine ⟨final,?_,hmem.2.2.2.2.2⟩
  rw [initial_exec]
  exact (congrArg (fun result : Option (State α)=>result.bind
    (run M ((code Y X rows N K B C).drop 3))) he).trans
      (by simpa only [Option.bind_some] using htail.trans hfinal)

#axiomsClean accumulate_step
#axiomsClean emit_step
#axiomsClean complete

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPForwardNoninner
