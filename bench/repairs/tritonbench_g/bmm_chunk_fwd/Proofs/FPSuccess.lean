import bench.repairs.tritonbench_g.bmm_chunk_fwd.Kernels
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.BmmChunkFwd.FPSuccess
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Only the locally invoked optional primitives are made available. Their
results remain arbitrary opaque floating values. This is backend availability,
not a numerical equation, program-success law, or output assumption. -/
structure Ops (α : Type) where
  scalar : Algebra α
  dot : Option ComputeDType → {batch : TileShape} → {m k n : Nat} →
    Values α .real (batch++[m,k]) → Values α .real (batch++[k,n]) → Values α .real (batch++[m,n])
  typedDot : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    Values α left.toTileDType (batch++[m,k]) → Values α right.toTileDType (batch++[k,n]) →
    Values α out.toTileDType (batch++[m,n])

def Ops.algebra {α : Type} (o : Ops α) : Algebra α :=
  { o.scalar with
    dot := fun p {_} {_} {_} {_} a b => some (o.dot p a b)
    dotTyped := fun p l r d cfg {_} {_} {_} {_} a b => some (o.typedDot p l r d cfg a b) }

def branch (h : Host.Call) (c : Host.Config) : Source.Conditional :=
  (Source.conditional? (Kernels.source h c).surfaceBody
    (fun body => (Source.dynamicRange? body "k").isSome)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩

def parts (h : Host.Call) (c : Host.Config) : Source.DynamicRange :=
  (Source.dynamicRange? (branch h c).yes "k").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 1,[],[]⟩

theorem source_split (h : Host.Call) (c : Host.Config) :
    (Kernels.source h c).surfaceBody=(branch h c).before ++
      [.ifThen (branch h c).condition (branch h c).yes] := by
  cases ha : h.aStorage <;> cases hb : h.bStorage <;>
    simp only [parts,branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize] <;> rfl

theorem branch_split (h : Host.Call) (c : Host.Config) :
    (branch h c).yes=(parts h c).before ++
      [.forRangeDyn "k" (parts h c).start (parts h c).stop (parts h c).stride (parts h c).body] ++
      (parts h c).after := by
  cases ha : h.aStorage <;> cases hb : h.bStorage <;>
    simp only [parts,branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize] <;> rfl

def Has {α : Type} (s : State α) (d : TileDType) (sh : TileShape) (n : RegName) := (s.regs d sh n).isSome

def Base {α : Type} (s : State α) : Prop :=
  Has s .nat [] "pid_b" ∧ Has s .nat [] "pid_c" ∧ Has s .nat [] "pid_h" ∧
  Has s .nat [] "pid_m" ∧ Has s .nat [] "pid_n"

def Ready {α : Type} (h : Host.Call) (c : Host.Config) (s : State α) : Prop :=
  Base s ∧ Has s .nat [c.M] "offs_m" ∧ Has s .nat [c.N] "offs_n" ∧
  Has s .nat [c.K] "offs_k" ∧ Has s .ptr [c.M,c.K] "a_ptrs" ∧
  Has s .ptr [c.K,c.N] "b_ptrs" ∧ Has s .nat [] "chunk_size_limit" ∧
  Has s .real [c.M,c.N] "acc" ∧ (h.hasSeq=Bool.true → Has s .ptr [] "seq_idx_ptr")

def active {α : Type} (h : Host.Call) (c : Host.Config) (s : State α) : Bool :=
  !h.causal || decide (s.pids 0 % Host.cdiv h.shape.chunk c.N*c.N <
    (s.pids 0 / Host.cdiv h.shape.chunk c.N+1)*c.M)

def Fixed {α : Type} (s t : State α) : Prop :=
  t.regs .nat [] "pid_b"=s.regs .nat [] "pid_b" ∧
  t.regs .nat [] "pid_c"=s.regs .nat [] "pid_c" ∧
  t.regs .nat [] "pid_h"=s.regs .nat [] "pid_h" ∧
  t.regs .nat [] "pid_m"=s.regs .nat [] "pid_m" ∧
  t.regs .nat [] "pid_n"=s.regs .nat [] "pid_n"

def Controls {α : Type} (h : Host.Call) (c : Host.Config) (s t : State α) : Prop :=
  t.regs .nat [] "pid_b"=some (fun _ => s.pids 1) ∧
  t.regs .nat [] "pid_c"=some (fun _ => s.pids 2/h.shape.groups) ∧
  t.regs .nat [] "pid_h"=some (fun _ => s.pids 2-s.pids 2/h.shape.groups*h.shape.groups) ∧
  t.regs .nat [] "pid_m"=some (fun _ => s.pids 0/Host.cdiv h.shape.chunk c.N) ∧
  t.regs .nat [] "pid_n"=some (fun _ => s.pids 0%Host.cdiv h.shape.chunk c.N)

theorem Fixed.refl {α : Type} (s : State α) : Fixed s s := ⟨rfl,rfl,rfl,rfl,rfl⟩
theorem Fixed.trans {α : Type} {s t u : State α} (a : Fixed s t) (b : Fixed t u) : Fixed s u := by
  rcases a with ⟨a0,a1,a2,a3,a4⟩; rcases b with ⟨b0,b1,b2,b3,b4⟩
  exact ⟨b0.trans a0,b1.trans a1,b2.trans a2,b3.trans a3,b4.trans a4⟩
theorem Controls.fixed {α : Type} {h : Host.Call} {c : Host.Config} {s t u : State α}
    (a : Controls h c s t) (b : Fixed t u) : Controls h c s u := by
  rcases a with ⟨a0,a1,a2,a3,a4⟩; rcases b with ⟨b0,b1,b2,b3,b4⟩
  exact ⟨b0.trans a0,b1.trans a1,b2.trans a2,b3.trans a3,b4.trans a4⟩

theorem prologue {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config) (s : State α) :
    ∃ t, run (engine o.algebra) (branch h c).before s=some t ∧ Base t ∧ t.mem=s.mem ∧
      FP.Structural.evalOp (engine o.algebra) none (branch h c).condition t=some (fun _ => active h c s) ∧
      Controls h c s t := by
  cases ha : h.aStorage <;> cases hb : h.bStorage <;> cases causal : h.causal
  all_goals
    simp (config := {maxSteps := 300000}) [branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,
      ha,hb,causal,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
      numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,State.setReg,integralFloorDiv,integralMod,Base,Has,active,Host.cdiv,Controls]
  all_goals repeat' constructor


theorem setup {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (s : State α) (ready : Base s) :
    ∃ t, run (engine o.algebra) (parts h c).before s=some t ∧ Ready h c t ∧ t.mem=s.mem ∧ Fixed s t := by
  rcases ready with ⟨h0,h1,h2,h3,h4⟩
  obtain ⟨v0,h0⟩ := Option.isSome_iff_exists.mp h0
  obtain ⟨v1,h1⟩ := Option.isSome_iff_exists.mp h1
  obtain ⟨v2,h2⟩ := Option.isSome_iff_exists.mp h2
  obtain ⟨v3,h3⟩ := Option.isSome_iff_exists.mp h3
  obtain ⟨v4,h4⟩ := Option.isSome_iff_exists.mp h4
  cases ha : h.aStorage <;> cases hb : h.bStorage <;> cases seq : h.hasSeq
  all_goals
    simp (config := {maxSteps := 300000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,
      ha,hb,seq,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
      numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,State.setReg,integralFloorDiv,integralMod,Ready,Base,Has,Fixed,h0,h1,h2,h3,h4]


end
end VeriTile.Bench.Repairs.BmmChunkFwd.FPSuccess

#guard_msgs (drop info) in
#auditModuleAxioms
