import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelNativeCarrier

/-! Independent finite native recurrence. Inputs are the entry-memory words;
there are no output loads, interpreter calls, or success/output premises.
All rounded arithmetic is the instruction carrier, including K8 dot chains,
complete reduction trees, scale bits, masks and the pre-attention correction. -/
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelNativeReference
open VeriTile.Triton FP.Structural Whole KernelNativeCarrier
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def gate (isa : Instructions) (route : Routing) (c : Config) (s : State Node) (r : Fin 64) : Node :=
  materialize isa route (if s.pids 1*64+r.val<c.T then
    (s.mem "g" (s.pids 2*c.T+s.pids 1*64+r.val)).read .real else .zero)
def lastGate (isa : Instructions) (route : Routing) (c : Config) (s : State Node) : Node :=
  materialize isa route ((s.mem "g" (s.pids 2*c.T+(min (s.pids 1*64+64) c.T-1))).read .real)
def value (c : Config) (s : State Node) (r : RegionName) (block : Nat) (i : TileIndex [64,c.BV]) : Node :=
  if s.pids 1*64+i.1.val<c.T ∧ block*c.BV+i.2.1.val<c.V then
    (s.mem r (s.pids 2*(c.T*c.V)+(s.pids 1*64+i.1.val)*c.V+(block*c.BV+i.2.1.val))).read .fp32 else .zero
def history (isa : Instructions) (route : Routing) (c : Config) (s : State Node)
    (r : RegionName) (block : Nat) (i : TileIndex [c.BV,c.BK]) : Node :=
  materialize isa route (if block*c.BV+i.1.val<c.V ∧ s.pids 0*c.BK+i.2.1.val<c.K then
    (s.mem r (s.pids 2*(c.NT*c.K*c.V)+s.pids 1*(c.K*c.V)+
      (block*c.BV+i.1.val)+(s.pids 0*c.BK+i.2.1.val)*c.V)).read .real else .zero)
def queryKey (c : Config) (s : State Node) (r : RegionName) (i : TileIndex [64,c.BK]) : Node :=
  if s.pids 1*64+i.1.val<c.T ∧ s.pids 0*c.BK+i.2.1.val<c.K then
    (s.mem r (s.pids 2*(c.T*c.K)+(s.pids 1*64+i.1.val)*c.K+(s.pids 0*c.BK+i.2.1.val))).read .fp32 else .zero
def total {m n : Nat} (route : Routing) (a b : TileIndex [m,n] → Node) : Node :=
  reduce route (shape:=[m*n]) ⟨0,by simp⟩ Bool.false
    (fun i=>let j:=TileShape.reshapeIndex (by simp [TileShape.product]) i; .mul (a j) (b j)) ()
theorem total_bridge (isa : Instructions) (route : Routing) (scale : Word) {m n : Nat}
    (a b : TileIndex [m,n] → Node) : Typed.totalProduct (algebra isa route scale) a b=total route a b := rfl

structure Carry (c : Config) where
  q : TileIndex [64,c.BK] → Node
  k : TileIndex [64,c.BK] → Node
  score : TileIndex [64,64] → Node
  terminal : Node
def seed (c : Config) : Carry c := ⟨fun _=>.zero,fun _=>.zero,fun _=>.zero,.zero⟩
def step (isa : Instructions) (route : Routing) (c : Config) (s : State Node) (block : Nat) (a : Carry c) : Carry c :=
  let v:=value c s "v" block
  let d:=value c s "do" block
  let h:=history isa route c s "h" block
  let f:=history isa route c s "dh" block
  ⟨fun i=>.add (a.q i) (.dot c.BV i.1.val i.2.1.val (fun j=>d (i.1,j,())) (fun j=>h (j,i.2.1,()))),
   fun i=>.add (a.k i) (.dot c.BV i.1.val i.2.1.val (fun j=>v (i.1,j,())) (fun j=>f (j,i.2.1,()))),
   fun i=>.add (a.score i) (.dot c.BV i.1.val i.2.1.val (fun j=>d (i.1,j,())) (fun j=>v (i.2.1,j,()))),
   .add a.terminal (total route h f)⟩
def accumulate (isa : Instructions) (route : Routing) (c : Config) (s : State Node) : Nat → Carry c
  | 0=>seed c
  | n+1=>step isa route c s n (accumulate isa route c s n)
def gatedQ (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node) (a : Carry c)
    (i : TileIndex [64,c.BK]) : Node := .mul (.mul (a.q i) (.exp (gate isa route c s i.1))) (.scale scale)
def gatedK (isa : Instructions) (route : Routing) (c : Config) (s : State Node) (a : Carry c)
    (i : TileIndex [64,c.BK]) : Node := .mul (a.k i) (.exp (.add (.neg (gate isa route c s i.1)) (lastGate isa route c s)))
def scores (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node) (a : Carry c)
    (i : TileIndex [64,64]) : Node := if i.2.1.val ≤ i.1.val then
      .mul (.mul (a.score i) (.scale scale)) (.exp (.sub (gate isa route c s i.1) (gate isa route c s i.2.1))) else .zero
def query (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node) (a : Carry c)
    (i : TileIndex [64,c.BK]) : Node :=
  .add (gatedQ isa route scale c s a i) (.dot 64 i.1.val i.2.1.val
    (fun j=>scores isa route scale c s a (i.1,j,())) (fun j=>queryKey c s "k" (j,i.2.1,())))
def key (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node) (a : Carry c)
    (i : TileIndex [64,c.BK]) : Node :=
  .add (gatedK isa route c s a i) (.dot 64 i.1.val i.2.1.val
    (fun j=>scores isa route scale c s a (j,i.1,())) (fun j=>queryKey c s "q" (j,i.2.1,())))
def correction (isa : Instructions) (route : Routing) (c : Config) (s : State Node) (a : Carry c) : Node :=
  .add (.mul a.terminal (.exp (lastGate isa route c s)))
    (total route (gatedK isa route c s a) (queryKey c s "k"))
def gateResult (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node) (a : Carry c)
    (i : TileIndex [64]) : Node :=
  let row:=Node.add .zero (reduce route (shape:=[64,c.BK]) ⟨1,by simp⟩ Bool.false
    (fun j=>.sub (.mul (queryKey c s "q" j) (query isa route scale c s a j))
      (.mul (queryKey c s "k" j) (key isa route scale c s a j))) i)
  if i.1.val<min 64 (c.T-s.pids 1*64)-1 then row else .add row (correction isa route c s a)

def typedCarry (a : Carry c) : Typed.Carries Node (nativeConfig c) :=
  ⟨a.q,a.k,a.score,fun _=>a.terminal⟩
theorem value_bridge (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node)
    (r : RegionName) (n : Nat) : Typed.valTile (algebra isa route scale) (nativeConfig c) s r n=value c s r n := by
  funext i
  simp [Typed.valTile,value,algebra,nativeConfig,Config.layout,scaleSlot]
  rfl
theorem history_bridge (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node)
    (r : RegionName) (n : Nat) : Typed.histTile (algebra isa route scale) (nativeConfig c) s r n=history isa route c s r n := by
  funext i
  simp [Typed.histTile,Typed.zero,history,algebra,nativeConfig,Config.layout,scaleSlot]
  rfl
theorem key_bridge (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node)
    (r : RegionName) : Typed.keyTile (algebra isa route scale) (nativeConfig c) s r=queryKey c s r := by
  funext i
  simp [Typed.keyTile,queryKey,algebra,nativeConfig,Config.layout,scaleSlot]
  rfl
theorem gate_bridge (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node) :
    Typed.gate (algebra isa route scale) (nativeConfig c) s=gate isa route c s := by
  funext r
  simp [Typed.gate,Typed.zero,gate,algebra,nativeConfig,scaleSlot,Nat.add_assoc]
theorem last_bridge (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node) :
    Typed.gateLast (algebra isa route scale) (nativeConfig c) s=lastGate isa route c s := rfl
theorem recurrence (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node) (n : Nat) :
    Typed.accumulated (algebra isa route scale) dotModel (nativeConfig c) s n=
      typedCarry (accumulate isa route c s n) := by
  induction n with
  | zero=>
    simp [Typed.accumulated,Typed.initialCarries,Typed.zero,algebra,scaleSlot,accumulate,seed,typedCarry]
    rfl
  | succ n ih=>
    rw [Typed.accumulated,ih]
    simp only [Typed.updateCarries,value_bridge,history_bridge,total_bridge]
    simp only [typedCarry,accumulate,step,Typed.add,algebra,dotModel,dotValues,
      FP.Structural.ofFloat,FP.Structural.toFloat]
    rfl
theorem answers (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node) (a : Carry c) :
    Typed.answerQ (algebra isa route scale) dotModel (nativeConfig c) s (typedCarry a)=query isa route scale c s a ∧
    Typed.answerK (algebra isa route scale) dotModel (nativeConfig c) s (typedCarry a)=key isa route scale c s a ∧
    Typed.answerG (algebra isa route scale) dotModel (nativeConfig c) s (typedCarry a)=gateResult isa route scale c s a := by
  refine ⟨?_,?_,?_⟩ <;> funext i
  all_goals simp only [Typed.answerQ,Typed.answerK,Typed.answerG,Typed.gatedQ,Typed.gatedK,Typed.scores,
    Typed.lastCorrection,Typed.gateContribution,gate_bridge,last_bridge,key_bridge,total_bridge]
  all_goals simp [
    Typed.add,Typed.mul,Typed.sub,Typed.exp,Typed.zero,typedCarry,query,key,gateResult,gatedQ,gatedK,scores,correction,
    gate,lastGate,queryKey,total,algebra,nativeConfig,Config.layout,dotModel,dotValues,
    FP.Structural.ofFloat,FP.Structural.toFloat,scaleSlot,ite_true,ne_eq,neg_eq_zero,one_ne_zero,if_false,
    Nat.add_assoc]
  all_goals simp [Typed.scores,Typed.gatedQ,Typed.gatedK,Typed.gate,Typed.gateLast,Typed.keyTile,
    Typed.zero,Typed.mul,Typed.add,Typed.sub,Typed.exp,algebra,scaleSlot,Config.layout,nativeConfig,Nat.add_assoc]
  all_goals rfl

#axiomsClean recurrence
#axiomsClean answers
#specNonCircular accumulate avoiding [Typed.accumulated,Typed.runGrid,FP.Structural.exec]
#specNonCircular gateResult avoiding [Typed.answerG,Typed.runGrid,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelNativeReference
