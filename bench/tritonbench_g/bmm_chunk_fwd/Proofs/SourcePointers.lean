import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeBridge

namespace VeriTile.Bench.Repairs.BmmChunkFwd.SourcePointers
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open FPSuccess
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

structure Coordinates where
  batch : Nat
  chunk : Nat
  head : Nat
  m : Nat
  n : Nat

def coordinates (h : Host.Call) (c : Host.Config) (p : Geometry.PID h c) : Coordinates :=
  ⟨Geometry.p1 p,Geometry.p2 p/h.shape.groups,
    Geometry.p2 p-Geometry.p2 p/h.shape.groups*h.shape.groups,
    Geometry.p0 p/Host.cdiv h.shape.chunk c.N,Geometry.p0 p%Host.cdiv h.shape.chunk c.N⟩

def Controls {α : Type} (d : Coordinates) (s : State α) : Prop :=
  s.regs .nat [] "pid_b"=some (fun _ => d.batch) ∧
  s.regs .nat [] "pid_c"=some (fun _ => d.chunk) ∧
  s.regs .nat [] "pid_h"=some (fun _ => d.head) ∧
  s.regs .nat [] "pid_m"=some (fun _ => d.m) ∧
  s.regs .nat [] "pid_n"=some (fun _ => d.n)

theorem minimum (a b : Nat) : (if a<b then a else b)=min a b := by
  by_cases h : a<b
  · rw [if_pos h,Nat.min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h,Nat.min_eq_right (by omega)]

def row (c : Host.Config) (d : Coordinates) (i : Nat) := d.m*c.M+i
def col (c : Host.Config) (d : Coordinates) (i : Nat) := d.n*c.N+i
def limit (h : Host.Call) (d : Coordinates) := min h.shape.chunk (h.shape.seqlen-d.chunk*h.shape.chunk)
def aOffset (h : Host.Call) (c : Host.Config) (d : Coordinates) (q i e : Nat) :=
  FPGrid.inputOffset (Host.prepared h.shape h.aStrides) d.batch
    (d.chunk*h.shape.chunk+row c d i) d.head (q*c.K+e)
def bOffset (h : Host.Call) (c : Host.Config) (d : Coordinates) (q e j : Nat) :=
  FPGrid.inputOffset (Host.prepared h.shape h.bStrides) d.batch
    (d.chunk*h.shape.chunk+col c d j) d.head (q*c.K+e)
def seqBase (h : Host.Call) (d : Coordinates) := d.batch*h.seqBatch+d.chunk*h.shape.chunk*h.seqRow
def seqOffset (h : Host.Call) (d : Coordinates) (lane : Nat) := seqBase h d+lane*h.seqRow
def outOffset (h : Host.Call) (c : Host.Config) (d : Coordinates) (i j : Nat) :=
  (d.batch*((h.shape.chunks*h.shape.groups)*h.shape.chunk*h.shape.chunk)+
    d.chunk*(h.shape.groups*h.shape.chunk*h.shape.chunk)+
    d.head*(if h.shape.grouped then h.shape.chunk*h.shape.chunk else 0))+
    (h.shape.chunk*row c d i+col c d j*1)

/-- Exact source pointer registers before K block q, including the pointer
values after the final (possibly padded) advance when q equals the loop stop. -/
def StateAt {α : Type} (h : Host.Call) (c : Host.Config) (d : Coordinates)
    (q : Nat) (s : State α) : Prop :=
  Controls d s ∧
  s.regs .nat [c.M] "offs_m"=some (fun i => row c d i.1) ∧
  s.regs .nat [c.N] "offs_n"=some (fun i => col c d i.1) ∧
  s.regs .nat [c.K] "offs_k"=some (fun i => i.1.val) ∧
  s.regs .ptr [c.M,c.K] "a_ptrs"=some (fun i => (h.A,aOffset h c d q i.1 i.2.1)) ∧
  s.regs .ptr [c.K,c.N] "b_ptrs"=some (fun i => (h.B,bOffset h c d q i.1 i.2.1)) ∧
  s.regs .nat [] "chunk_size_limit"=some (fun _ => limit h d) ∧
  Has s .real [c.M,c.N] "acc" ∧
  (h.hasSeq=Bool.true → s.regs .ptr [] "seq_idx_ptr"=some (fun _ => (Region.cast h.Seq,seqBase h d)))

theorem ready {α : Type} (h : Host.Call) (c : Host.Config) (d : Coordinates)
    (q : Nat) (s : State α) (hs : StateAt h c d q s) : Ready h c s := by
  rcases hs with ⟨⟨b,ch,g,m,n⟩,hm,hn,hk,ha,hb,hl,hacc,hseq⟩
  simp only [Ready,Base,Has,b,ch,g,m,n,hm,hn,hk,ha,hb,hl,Option.isSome_some,true_and]
  exact ⟨hacc,fun h => by rw [hseq h]; rfl⟩

theorem setup {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (d : Coordinates) (s : State α) (hc : Controls d s)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, run (engine o.algebra) (parts h c).before s=some t ∧
      StateAt h c d 0 t ∧ t.mem=s.mem := by
  rcases hc with ⟨h0,h1,h2,h3,h4⟩
  cases seq : h.hasSeq
  all_goals
    simp (config := {maxSteps := 400000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,VeriTile.Bench.Repairs.BmmChunkFwd.Kernels.mainSource,
      ha,hb,seq,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
      numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,State.setReg,integralFloorDiv,integralMod,StateAt,Controls,Has,
      h0,h1,h2,h3,h4,aOffset,bOffset,FPGrid.inputOffset,row,col,limit,seqBase,
      Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals
    repeat' constructor
    all_goals
      funext i
      simp [bop, minimum, Nat.add_mul,Nat.mul_add,Nat.mul_assoc,
        Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
      all_goals omega

theorem body {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (d : Coordinates) (q : Nat) (s : State α) (hs : StateAt h c d q s)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, run (engine o.algebra) (parts h c).body
      (s.setReg "k" .nat [] (fun _ => q))=some t ∧
      StateAt h c d (q+1) t ∧ t.mem=s.mem := by
  rcases hs with ⟨⟨h0,h1,h2,h3,h4⟩,h5,h6,h7,h8,h9,h10,h11,hseq⟩
  obtain ⟨acc,h11⟩ := Option.isSome_iff_exists.mp h11
  cases seq : h.hasSeq
  all_goals
    simp (config := {maxSteps := 400000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,VeriTile.Bench.Repairs.BmmChunkFwd.Kernels.mainSource,
      ha,hb,seq,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
      numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,State.setReg,integralFloorDiv,integralMod,StateAt,Controls,Has,
      h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,hseq,
      aOffset,bOffset,FPGrid.inputOffset,row,col,limit,seqBase,
      Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals
    constructor <;> funext i <;>
      simp [bop,Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
    all_goals omega

theorem iterations {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (d : Coordinates) (cur stop : Nat) (hc : cur≤stop) (s : State α)
    (hs : StateAt h c d cur s) (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, range (engine o.algebra) "k" cur stop 1 (parts h c).body s=some t ∧
      StateAt h c d stop t ∧ t.mem=s.mem := by
  by_cases less : cur<stop
  · obtain ⟨mid,hm,hr,hmem⟩ := body o h c d cur s hs ha hb
    obtain ⟨t,ht,htr,htm⟩ := iterations o h c d (cur+1) stop (by omega) mid hr ha hb
    refine ⟨t,?_,htr,htm.trans hmem⟩
    rw [range,if_neg (by decide),if_pos less,hm]
    exact ht
  · have eq : cur=stop := by omega
    exact ⟨s,by rw [range,if_neg (by decide),if_neg less],eq ▸ hs,rfl⟩
termination_by stop-cur

#axiomsClean setup
#axiomsClean body
#axiomsClean iterations
end
end VeriTile.Bench.Repairs.BmmChunkFwd.SourcePointers

#guard_msgs (drop info) in
#auditModuleAxioms
