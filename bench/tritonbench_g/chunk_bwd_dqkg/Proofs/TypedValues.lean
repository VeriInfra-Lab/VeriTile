import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedStores
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def add (M : Algebra α) := M.binary (some .fp32) .real .add
def mul (M : Algebra α) := M.binary (some .fp32) .real .mul
def sub (M : Algebra α) := M.binary (some .fp32) .real .sub
def exp (M : Algebra α) := M.unary (some .fp32) .exp
def zero (M : Algebra α) := M.literal (some .fp32) .real 0

def gate (M : Algebra α) (c : Config) (s : State α) (r : Fin 64) : α :=
  M.fp32Load (if s.pids 1*64+r.val<c.T then
    (s.mem "g" (s.pids 2*c.T+(s.pids 1*64+r.val))).read .real else zero M)
def gateLast (M : Algebra α) (c : Config) (s : State α) : α :=
  M.fp32Load ((s.mem "g" (s.pids 2*c.T+(min (s.pids 1*64+64) c.T-1))).read .real)

def valTile (M : Algebra α) (c : Config) (s : State α) (region : RegionName)
    (block : Nat) (i : TileIndex [64,c.BV]) : α :=
  if s.pids 1*64+i.1.val<c.T ∧ block*c.BV+i.2.1.val<c.V then
    (s.mem region (s.pids 2*c.layout.s_v_h+(s.pids 1*64+i.1.val)*c.layout.s_v_t+(block*c.BV+i.2.1.val))).read .fp32
  else M.literal (some .fp32) .fp32 0

def histTile (M : Algebra α) (c : Config) (s : State α) (region : RegionName)
    (block : Nat) (i : TileIndex [c.BV,c.BK]) : α :=
  M.fp32Load (if block*c.BV+i.1.val<c.V ∧ s.pids 0*c.BK+i.2.1.val<c.K then
    (s.mem region (s.pids 2*c.layout.s_h_h+s.pids 1*(c.K*c.layout.s_h_t)+
      (block*c.BV+i.1.val)+(s.pids 0*c.BK+i.2.1.val)*c.layout.s_h_t)).read .real else zero M)

structure Carries (α : Type) (c : Config) where
  dq : TileIndex [64,c.BK] → α
  dk : TileIndex [64,c.BK] → α
  ds : TileIndex [64,64] → α
  last : TileIndex [1] → α

def initialCarries (M : Algebra α) (c : Config) : Carries α c :=
  ⟨fun _=>zero M,fun _=>zero M,fun _=>zero M,fun _=>zero M⟩

def flatProduct {m n : Nat} (M : Algebra α) (a b : TileIndex [m,n] → α)
    (i : TileIndex [m*n]) : α :=
  let idx := TileShape.reshapeIndex (by simp [TileShape.product]) i
  mul M (a idx) (b idx)

def totalProduct {m n : Nat} (M : Algebra α) (a b : TileIndex [m,n] → α) : α :=
  M.reduceSum (some .fp32) (shape:=[m*n]) ⟨0,by simp⟩ Bool.false (flatProduct M a b) ()

/-- Independent finite FP recurrence in exact source order, including storage
casts and the flat history-product reduction. -/
def updateCarries (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (block : Nat) (a : Carries α c) : Carries α c :=
  let v := valTile M c s "v" block
  let dout := valTile M c s "do" block
  let h := histTile M c s "h" block
  let dh := histTile M c s "dh" block
  ⟨fun i=>add M (a.dq i) (M.cast (some .fp32) .fp32 .real
      (dot.apply (some .fp32) .fp32 .fp32 .fp32 {} (batch:=[]) dout
        (fun j=>M.cast (some .fp32) .real .fp32 (h j)) i)),
   fun i=>add M (a.dk i) (M.cast (some .fp32) .fp32 .real
      (dot.apply (some .fp32) .fp32 .fp32 .fp32 {} (batch:=[]) v
        (fun j=>M.cast (some .fp32) .real .fp32 (dh j)) i)),
   fun i=>add M (a.ds i) (M.cast (some .fp32) .fp32 .real
      (dot.apply (some .fp32) .fp32 .fp32 .fp32 {} (batch:=[]) dout
        (fun j=>v (j.2.1,j.1,())) i)),
   fun i=>add M (a.last i) (totalProduct M h dh)⟩

def accumulated (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) : Nat → Carries α c
  | 0=>initialCarries M c
  | n+1=>updateCarries M dot c s n (accumulated M dot c s n)

structure Context (M : Algebra α) (c : Config) (initial : State α)
    (a : Carries α c) (current : State α) : Prop where
  memory : current.mem=initial.mem
  key : current.regs .nat [] "i_k"=some (fun _=>initial.pids 0)
  time : current.regs .nat [] "i_t"=some (fun _=>initial.pids 1)
  head : current.regs .nat [] "i_bh"=some (fun _=>initial.pids 2)
  heads : current.regs .nat [] "n_bh"=some (fun _=>initial.numPids 2)
  rows : current.regs .nat [64] "o_i"=some (fun i=>i.1.val)
  gate : current.regs .real [64] "b_g"=some (fun i=>Typed.gate M c initial i.1)
  last : current.regs .real [] "b_g_last"=some (fun _=>gateLast M c initial)
  dq : current.regs .real [64,c.BK] "b_dq"=some a.dq
  dk : current.regs .real [64,c.BK] "b_dk"=some a.dk
  ds : current.regs .real [64,64] "b_ds"=some a.ds
  dgLast : current.regs .real [1] "b_dg_last"=some a.last
  dg : current.regs .real [64] "b_dg"=some (fun _=>zero M)

def Context.ready {M : Algebra α} {c : Config} {s t : State α} {a : Carries α c}
    (h : Context M c s a t) : Ready c t :=
  ⟨⟨_,h.key⟩,⟨_,h.time⟩,⟨_,h.head⟩,⟨_,h.heads⟩,⟨_,h.rows⟩,⟨_,h.gate⟩,
   ⟨_,h.last⟩,⟨_,h.dq⟩,⟨_,h.dk⟩,⟨_,h.ds⟩,⟨_,h.dgLast⟩,⟨_,h.dg⟩⟩

private theorem min_lt (a b : Nat) : min a b=(if a<b then a else b) := by split_ifs <;> omega

theorem entry_context (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) :
    Context M c s (initialCarries M c) (entry M dot c s).val := by
  constructor
  all_goals simp [entry,initialCarries,gate,gateLast,zero,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    TileShape.indexToList,min_lt]

#specNonCircular updateCarries avoiding [run,FP.Structural.exec,body,post,finalState]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
