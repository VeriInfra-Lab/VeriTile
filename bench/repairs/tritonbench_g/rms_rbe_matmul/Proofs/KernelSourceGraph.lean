import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelPairIndex

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 7000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem foldRange_list {β : Type} (f : Nat → β → β) (start count : Nat) (v : β) :
    foldRange f start (start+count) 1 v=(List.range' start count).foldl (fun a n => f n a) v := by
  induction count generalizing start v with
  | zero => simp [foldRange]
  | succ count ih =>
    rw [foldRange]
    simp only [show (1:Nat)≠0 by decide,if_false,
      show start<start+(count+1) by omega,if_true,List.range'_succ,List.foldl_cons]
    simpa only [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using ih (start+1) (f start v)
private theorem foldRange_zero {β : Type} (f : Nat → β → β) (v : β) : foldRange f 0 0 1 v=v := by
  simp [foldRange]
private theorem foldRange_succ {β : Type} (f : Nat → β → β) (n : Nat) (v : β) :
    foldRange f 0 (n+1) 1 v=f n (foldRange f 0 n 1 v) := by
  have h0 := foldRange_list f 0 n v
  have h1 := foldRange_list f 0 (n+1) v
  simp only [Nat.zero_add] at h0 h1
  rw [h0,h1,List.range'_1_concat,List.foldl_append]
  simp

def inputs (p : Parameters) (s : State α) : KernelGraph.Input α where
  x := fun b m k => (s.mem "X" (b*(p.M*p.K)+m*p.K+k)).read .real
  weightHalf := fun n k => (s.mem "W" (k+n*p.K)).read .fp16
  rmsHalf := fun k => (s.mem "RMS" k).read .fp16
  weightByte := fun n k => (s.mem "W" (k+n*p.K)).read .int
  rmsByte := fun k => (s.mem "RMS" k).read .int
def tile (p : Parameters) (s : State α) : KernelGraph.Tile := ⟨s.pids 0,pidM p s,pidN p s⟩

theorem x_graph (a : Operations α) (p : Parameters) (s : State α) (block : Nat)
    (i : TileIndex [16,64]) : xValue a p s block i=KernelGraph.x a p (inputs p s) (tile p s) block i := by
  simp [xValue,KernelGraph.x,inputs,tile,zero,KernelGraph.zero,xOffset,row,Nat.add_assoc]
theorem rms_graph (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α)
    (block lane : Nat) : rmsHalf a fp8 p s block lane=KernelGraph.rms a fp8 p (inputs p s) block lane := by
  rfl
theorem weight_graph (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α)
    (block : Nat) : weight a fp8 p s block=KernelGraph.weight a fp8 p (inputs p s) (tile p s) block := by
  funext i
  simp [weight,wHalf,KernelGraph.weight,inputs,tile,halfZero,zero,KernelGraph.halfZero,
    KernelGraph.zero,wOffset,col,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
theorem scaled_graph (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α)
    (block : Nat) : scaledX a fp8 p s block=KernelGraph.scaled a fp8 p (inputs p s) (tile p s) block := by
  funext i
  simp only [scaledX,KernelGraph.scaled,x_graph,rms_graph]

theorem fold_graph (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α) (count : Nat) :
    foldRange (update a fp8 p s) 0 count 1 (initial a)=
      ⟨KernelGraph.products a fp8 p (inputs p s) (tile p s) count,
       KernelGraph.moments a p (inputs p s) (tile p s) count⟩ := by
  induction count with
  | zero => rw [foldRange_zero]; rfl
  | succ count ih =>
    rw [foldRange_succ,ih]
    simp only [update,KernelGraph.products,KernelGraph.moments,scaled_graph,weight_graph,x_graph]

theorem normalized_graph (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α) :
    normalized a p (accumulated a fp8 p s)=
      KernelGraph.projection a fp8 p (inputs p s) (tile p s) := by
  rw [accumulated,fold_graph]
  rfl

theorem output_graph (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α)
    (i : TileIndex [16,64]) :
    outputCell a fp8 (rotated a p s (normalized a p (accumulated a fp8 p s)) i)=
      .mk (outputDType fp8).toTileDType (ofFloat (outputDType fp8)
        (KernelGraph.output a fp8 p (inputs p s) (tile p s) i)) := by
  rw [normalized_graph]
  simp only [outputCell,KernelGraph.output,rotated]
  congr 2
  cases hr : p.rotary
  · rfl
  · simp only [hr,if_true,reshape_to_pairs,TileShape.splitLastIndex,rotatedPair,
      reshape_from_pairs,angle,KernelGraph.phase,tile,row,KernelGraph.evenLane,KernelGraph.oddLane]
    have order : 2*(i.2.1.val/2)=i.2.1.val/2*2 := Nat.mul_comm _ _
    simp only [order,Fin.ext_iff]
    simp [outputDType]

#axiomsClean fold_graph
#axiomsClean output_graph
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
