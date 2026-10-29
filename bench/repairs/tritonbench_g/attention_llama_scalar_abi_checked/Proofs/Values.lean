import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.Source

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def base (c : Config) (s : State (Term α)) (sz sh : Nat) : Nat :=
  s.pids 1/c.NH*sz+s.pids 1%c.NH*sh

def rows (s : State (Term α)) (r : Fin 64) : Nat := s.pids 0*64+r.val

def pointers (c : Config) (s : State (Term α)) (R : RegionName) (sz sh sr sd n : Nat)
    (i : TileIndex [64,c.D]) : RegionName × Nat :=
  (R,base c s sz sh+i.1.val*sr+i.2.1.val*sd+n*64*sr)

def query (c : Config) (s : State (Term α)) (i : TileIndex [64,c.D]) : Term α :=
  if rows s i.1<c.H then
    (s.mem c.Q (base c s c.sqz c.sqh+rows s i.1*c.sqm+i.2.1.val*c.sqk)).read .fp16
  else .half (.literal 0)

def loaded (c : Config) (s : State (Term α)) (R : RegionName) (sz sh sr sd n : Nat)
    (i : TileIndex [64,c.D]) : Term α :=
  if c.fp8 then .half (.byte (UInt8.ofNat (Float8E5Bits.ofInt
    (if n*64+i.1.val<c.N then (s.mem R (pointers c s R sz sh sr sd n i).2).read .int else 0)).val))
  else if n*64+i.1.val<c.N then (s.mem R (pointers c s R sz sh sr sd n i).2).read .fp16
    else .half (.literal 0)

structure Carries (α : Type) (D : Nat) where
  maximum : Fin 64 → Term α
  denominator : Fin 64 → Term α
  accumulator : TileIndex [64,D] → Term α

def initial (D : Nat) : Carries α D :=
  ⟨fun _=>.add (.literal 0) .negativeInfinity,fun _=>.literal 0,fun _=>.literal 0⟩

def scores (c : Config) (s : State (Term α)) (scale : Term α) (n : Nat)
    (r k : Fin 64) : Term α :=
  let dot := Term.dot c.D (fun d=>query c s (r,d,()))
    (fun d=>loaded c s c.K c.skz c.skh c.skn c.skk n (k,d,()))
  let score := Term.mul (.add (.literal 0) dot) scale
  let masked := if n*64+k.val<c.N then score else .negativeInfinity
  if c.causal then if n*64+k.val+c.SP≤rows s r then masked else .negativeInfinity else masked

def advance (c : Config) (s : State (Term α)) (scale : Term α) (n : Nat)
    (a : Carries α c.D) : Carries α c.D :=
  let qk := scores c s scale n
  let m := fun r=>Term.maximum (.max 64 (qk r)) (a.maximum r)
  let prev := fun r=>Term.mul (a.denominator r) (.exp (.sub (a.maximum r) (m r)))
  let prob := fun r k=>Term.exp (.sub (qk r k) (m r))
  let denom := fun r=>Term.add (.sum 64 (prob r)) (prev r)
  let recip := fun r=>Term.div (.literal 1) (denom r)
  let hp := fun r k=>Term.half (.mul (prob r k) (recip r))
  ⟨m,denom,fun i=>.add
    (.mul (a.accumulator i) (.mul (prev i.1) (recip i.1)))
    (.dot 64 (hp i.1) (fun k=>loaded c s c.V c.svz c.svh c.svk c.svn n (k,i.2.1,())))⟩

def accumulated (c : Config) (s : State (Term α)) (scale : Term α) : Nat → Carries α c.D
  | 0=>initial c.D
  | n+1=>advance c s scale n (accumulated c s scale n)

structure Context (c : Config) (s : State (Term α)) (scale : Term α) (n : Nat)
    (a : Carries α c.D) (t : State (Term α)) : Prop where
  memory : t.mem=s.mem
  pids : t.pids=s.pids
  scalar : t.regs .real [] "sm_scale"=some (fun _=>scale)
  maximum : t.regs .real [64] "m_prev"=some (fun i=>a.maximum i.1)
  denominator : t.regs .real [64] "l_prev"=some (fun i=>a.denominator i.1)
  accumulator : t.regs .real [64,c.D] "acc"=some a.accumulator
  query : t.regs .fp16 [64,c.D] "q"=some (Source.query c s)
  rows : t.regs .nat [64] "offs_m"=some (fun i=>Source.rows s i.1)
  columns : t.regs .nat [64] "offs_n"=some (fun i=>i.1.val)
  batch : t.regs .nat [] "batch_id"=some (fun _=>s.pids 1/c.NH)
  head : t.regs .nat [] "off_hz"=some (fun _=>s.pids 1%c.NH)
  key : t.regs .ptr [64,c.D] "k_ptrs"=some (pointers c s c.K c.skz c.skh c.skn c.skk n)
  value : t.regs .ptr [64,c.D] "v_ptrs"=some (pointers c s c.V c.svz c.svh c.svk c.svn n)

 theorem entry_context (c : Config) (s : State (Term α)) (scale : Term α) :
    Context c s scale 0 (initial c.D) (entry c s scale).val := by
  rcases c with ⟨Q,K,V,Out,sqz,sqh,sqm,sqk,skz,skh,skn,skk,svz,svh,svk,svn,soz,soh,som,son,NH,H,N,SP,D,causal,fp8⟩
  cases fp8 <;> cases causal
  all_goals constructor <;> simp [entry,initial,query,pointers,rows,base,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,TileShape.indexToList]
  all_goals funext i; simp [query,pointers,rows,base,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,TileShape.indexToList]
  all_goals norm_num
  all_goals rfl

#axiomsClean entry_context
#specNonCircular accumulated avoiding [FP.Structural.exec,FP.Structural.run,entry]
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Source
