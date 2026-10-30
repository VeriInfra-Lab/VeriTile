import Mathlib.Tactic
namespace VeriTile.Bench.TritonBenchG.StreamK
structure HostInput where
 m : Nat
 n : Nat
 k : Nat
 bm : Nat
 bn : Nat
 bk : Nat
 programs : Nat
 twoTiles : Bool

/-- Python host integers, before native i32 argument materialization. No tail
padding, divisibility or whole-tile allocation is silently added. -/
def ceilDiv (a b : Nat) := (a+b-1)/b
structure HostPlan where
 tilesM : Nat
 tilesN : Nat
 iterations : Nat
 streamTiles : Nat
 blockingTiles : Nat
 fullIterations : Nat
 partialIterations : Nat

def hostPlan (h : HostInput) : HostPlan :=
 let tm:=ceilDiv h.m h.bm; let tn:=ceilDiv h.n h.bn
 let tiles:=tm*tn; let iters:=ceilDiv h.k h.bk
 let stream:=if h.programs>0 then
   let tail:=tiles%h.programs
   if h.twoTiles && tiles-tail>h.programs then tail+h.programs else tail
   else 0
 ⟨tm,tn,iters,stream,tiles-stream,(stream*iters)/h.programs,(stream*iters)%h.programs⟩

end VeriTile.Bench.TritonBenchG.StreamK
