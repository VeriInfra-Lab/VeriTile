import bench.tritonbench_g.softmax_flaggems.Proofs.SoftmaxFlaggemsFiniteSeed_Native_Native
import bench.tritonbench_g.softmax_flaggems.Proofs.FPTraceGrid
import bench.tritonbench_g.softmax_flaggems.Proofs.FPHost

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.Native
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000

def entry (memory : RegionName → Nat → Word) : State Term where
  mem := fun r o => .mk .real (.leaf r o (memory r o))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

def input (region : RegionName) (values : Nat → Word) : Nat → Term :=
  fun o => .leaf region o (values o)

/-- Complete host result at raw FP32 words, under the same selected scalar
reduction trees and shape-only local fusion choices. These parameters contain
no successful execution or whole-result equation. A concrete compiler/runtime
must supply its actual local selection; numerical primitive accuracy is local.
The Python's default fusion setting is retained. -/
theorem host_native (I : Instructions) (R : Reductions) (S : Schedule)
    (shape : List Nat) (dim : Int) (ax : Fin shape.length) (hnorm : normalizeDim shape dim=some ax)
    (device sms : Nat) (waves : WavesAboveOne) (layouts : LayoutPrimitives) (inputStrides : List Nat)
    (trials : List (Fin 6)) (choice : Fin 6) (X P G Y : RegionName)
    (memory : RegionName → Nat → Word) (x emptyP g emptyY : Nat → Word) (hPY : P≠Y) :
    let initial := entry memory
    let expected := FPHost.reference (algebra R) shape ax device sms waves layouts inputStrides
      trials choice X P G Y initial (input X x) (input P emptyP) (input G g) (input Y emptyY)
    ∃ result,
      FPHost.invoke (algebra R) shape dim device sms waves layouts inputStrides trials choice X P G Y initial
        (input X x) (input P emptyP) (input G g) (input Y emptyY)=.ok result ∧
      FPHost.Matches result expected ∧
      (∀ o, o<shape.prod → observe I S result.forwardState P o=observe I S expected.forwardState P o) ∧
      (∀ o, o<shape.prod → observe I S result.finalState Y o=observe I S expected.finalState Y o) ∧
      (∀ o, observe I S result.finalState P o=observe I S result.forwardState P o) ∧
      (∀ r o, ¬((r=X ∨ r=P ∨ r=G ∨ r=Y) ∧ o<shape.prod) →
        observe I S result.finalState r o=memory r o) := by
  dsimp only
  obtain ⟨result,he,hm,hs,hf⟩ := FPHost.whole_typed (algebra R) shape dim ax hnorm device sms waves layouts
    inputStrides trials choice X P G Y (entry memory) (input X x) (input P emptyP) (input G g) (input Y emptyY) hPY
  refine ⟨result,he,hm,?_,?_,?_,?_⟩
  · intro o ho; unfold observe; rw [hm.1]
  · intro o ho; unfold observe; rw [hm.2.1]
  · intro o; unfold observe; rw [hs o]
  · intro r o hn; unfold observe; rw [hf r o hn]; rfl

/-- Each output slot in the same host graph is an actual store event, and no
outside slot is a store event. This is not an assumed allocation completion. -/
theorem host_native_coverage (R : Reductions) (shape : List Nat) (ax : Fin shape.length)
    (hz : shape.prod≠0) (sms : Nat) (waves : WavesAboveOne)
    (trials : List (Fin 6)) (choice : Fin 6) (X P G Y : RegionName)
    (memory : RegionName → Nat → Word) (x emptyP g emptyY : Nat → Word) :
    let A := algebra R
    let d := dimensions shape ax
    let pre := FPHost.prepareForward (entry memory) X P shape.prod (input X x) (input P emptyP)
    let f := FPGrid.forwardGraph A X P d (forwardConfig waves d sms) pre
    let before := FPHost.prepareBackward f P G Y shape.prod (input G g) (input Y emptyY)
    let b := FPTuning.graph A P G Y d (trials++[choice]) before
    ∃ ft bt, f=FPTrace.execute ft pre ∧ b=FPTrace.execute bt before ∧
      (∀ a, a∈FPTrace.events ft ↔ a.1=P ∧ a.2<shape.prod) ∧
      (∀ a, a∈FPTrace.events bt ↔ a.1=Y ∧ a.2<shape.prod) :=
  FPHost.reference_traces (algebra R) shape ax hz sms waves trials choice X P G Y (entry memory)
    (input X x) (input P emptyP) (input G g) (input Y emptyY)

#axiomsClean host_native
#axiomsClean host_native_coverage
#specNonCircular entry avoiding [FP.Structural.exec,FPHost.invoke,FPHost.nonempty]
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.Native
