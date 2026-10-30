import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedLoopValues
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def keyTile (M : Algebra α) (c : Config) (s : State α) (region : RegionName)
    (i : TileIndex [64,c.BK]) : α :=
  if s.pids 1*64+i.1.val<c.T ∧ s.pids 0*c.BK+i.2.1.val<c.K then
    (s.mem region (s.pids 2*c.layout.s_k_h+(s.pids 1*64+i.1.val)*c.layout.s_k_t+(s.pids 0*c.BK+i.2.1.val))).read .fp32
  else M.literal (some .fp32) .fp32 0

def gatedQ (M : Algebra α) (c : Config) (s : State α) (a : Carries α c)
    (i : TileIndex [64,c.BK]) : α :=
  mul M (mul M (a.dq i) (exp M (gate M c s i.1))) (M.literal (some .fp32) .real c.scale)
def gatedK (M : Algebra α) (c : Config) (s : State α) (a : Carries α c)
    (i : TileIndex [64,c.BK]) : α :=
  mul M (a.dk i) (exp M (add M (M.unary (some .fp32) (.neg .real) (gate M c s i.1)) (gateLast M c s)))
def scores (M : Algebra α) (c : Config) (s : State α) (a : Carries α c)
    (i : TileIndex [64,64]) : α :=
  M.cast (some .fp32) .real .fp32
    (if i.2.1.val ≤ i.1.val then
      mul M (mul M (a.ds i) (M.literal (some .fp32) .real c.scale))
        (exp M (sub M (gate M c s i.1) (gate M c s i.2.1))) else zero M)
def answerQ (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : Carries α c)
    (i : TileIndex [64,c.BK]) : α :=
  add M (gatedQ M c s a i) (M.cast (some .fp32) .fp32 .real
    (dot.apply (some .fp32) .fp32 .fp32 .fp32 {} (batch:=[]) (scores (α:=α) M c s a) (keyTile (α:=α) M c s "k") i))
def answerK (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : Carries α c)
    (i : TileIndex [64,c.BK]) : α :=
  add M (gatedK M c s a i) (M.cast (some .fp32) .fp32 .real
    (dot.apply (some .fp32) .fp32 .fp32 .fp32 {} (batch:=[])
      (fun j=>scores (α:=α) M c s a (j.2.1,j.1,())) (keyTile (α:=α) M c s "q") i))

def lastCorrection (M : Algebra α) (c : Config) (s : State α) (a : Carries α c)
    (i : TileIndex [1]) : α :=
  add M (mul M (a.last i) (exp M (gateLast M c s)))
    (totalProduct M (gatedK M c s a) (fun j=>M.cast (some .fp32) .fp32 .real (keyTile M c s "k" j)))

def gateContribution (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (a : Carries α c) (i : TileIndex [64]) : α :=
  add M (zero M) (M.reduceSum (some .fp32) (shape:=[64,c.BK]) ⟨1,by simp⟩ Bool.false
    (fun j=>sub M
      (mul M (M.cast (some .fp32) .fp32 .real (keyTile M c s "q" j)) (answerQ M dot c s a j))
      (mul M (M.cast (some .fp32) .fp32 .real (keyTile M c s "k" j)) (answerK M dot c s a j))) i)

def answerG (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : Carries α c)
    (i : TileIndex [64]) : α :=
  if i.1.val<min 64 (c.T-s.pids 1*64)-1 then gateContribution M dot c s a i
  else add M (gateContribution M dot c s a i) (lastCorrection M c s a (⟨0,by decide⟩,()))

def qPointer (c : Config) (s : State α) (region : RegionName) : Values α .blockPtr [64,c.BK] :=
  fun _=>⟨region,s.pids 2*c.layout.s_k_h,[c.T,c.K],[64,c.BK],[c.layout.s_k_t,1],[s.pids 1*64,s.pids 0*c.BK]⟩
def gPointer (c : Config) (s : State α) : Values α .blockPtr [64] :=
  fun _=>⟨"dg",(s.pids 0*s.numPids 2+s.pids 2)*c.T,[c.T],[64],[1],[s.pids 1*64]⟩

structure AnswerContext (M : Algebra α) (dot : Dot α) (c : Config) (initial : State α)
    (a : Carries α c) (current : State α) : Prop where
  memory : current.mem=initial.mem
  dq : current.regs .real [64,c.BK] "b_dq"=some (answerQ M dot c initial a)
  dk : current.regs .real [64,c.BK] "b_dk"=some (answerK M dot c initial a)
  dg : current.regs .real [64] "b_dg"=some (answerG M dot c initial a)
  pq : current.regs .blockPtr [64,c.BK] "p_dq"=some (qPointer c initial "dq")
  pk : current.regs .blockPtr [64,c.BK] "p_dk"=some (qPointer c initial "dk")
  pg : current.regs .blockPtr [64] "p_dg"=some (gPointer c initial)

#specNonCircular answerQ avoiding [run,FP.Structural.exec,body,post,finalState]
#specNonCircular answerK avoiding [run,FP.Structural.exec,body,post,finalState]
#specNonCircular answerG avoiding [run,FP.Structural.exec,body,post,finalState]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
