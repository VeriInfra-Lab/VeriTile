import bench.tritonbench_g.flash_attn.Proofs.IOResults
import bench.tritonbench_g.flash_attn.Proofs.RealResult

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
noncomputable section
set_option maxHeartbeats 1000000

def realMemory (s : State KernelReal.R) : Memory ℝ :=
  fun r off => ((s.mem (regionName r) off).read .real).getD 0

def liftedState (s : State KernelReal.R) : State KernelReal.R :=
  {s with mem := liftMemory (Real.lift (realMemory s))}

variable (x : Layout) (scale : ℝ) (xs : Inputs KernelReal.R x) (s : State KernelReal.R)
  (hin : (realIO x scale).InputBinding xs s)

include hin
theorem lifted_binding (domain : Logical.RealInputs x xs) :
    (realIO x scale).InputBinding xs (liftedState s) := by
  intro port i _
  have lift_value (v : KernelReal.R) (finite : ∃ value : ℝ,v=some value) : some (v.getD 0)=v := by
    obtain ⟨value,rfl⟩ := finite
    rfl
  fin_cases port
  · change some (((s.mem "Q" (FlashAttn.address x.q i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val)).read .real).getD 0)=xs (0 : Fin 3) i
    have h : (s.mem "Q" (FlashAttn.address x.q i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val)).read .real=xs (0 : Fin 3) i := hin (0 : Fin 3) i (by trivial)
    rw [h]
    exact lift_value _ (domain 0 i)
  · change some (((s.mem "K" (FlashAttn.address x.k i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val)).read .real).getD 0)=xs (1 : Fin 3) i
    have h : (s.mem "K" (FlashAttn.address x.k i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val)).read .real=xs (1 : Fin 3) i := hin (1 : Fin 3) i (by trivial)
    rw [h]
    exact lift_value _ (domain 1 i)
  · change some (((s.mem "V" (FlashAttn.address x.v i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val)).read .real).getD 0)=xs (2 : Fin 3) i
    have h : (s.mem "V" (FlashAttn.address x.v i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val)).read .real=xs (2 : Fin 3) i := hin (2 : Fin 3) i (by trivial)
    rw [h]
    exact lift_value _ (domain 2 i)

theorem real_memory_value (port : Fin 3)
    (i : TileIndex [x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim]) :
    realMemory s (sourceRegion (inputs x)[port].buf)
      (FlashAttn.address (inputStrides x port) i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val)=
      (array xs port i).getD 0 := by
  have h := input_value KernelReal.algebra KernelReal.dot id x (some scale) (fun _ _ => id) xs s hin port i
  fin_cases port <;> simpa [realMemory,inputs,sourceRegion,regionName] using congrArg (fun v => v.getD 0) h

theorem real_input_logical (port : Fin 3) (b h row col : Nat)
    (hb : b<x.shape.batches) (hh : h<x.shape.heads) (hr : row<x.shape.length) (hc : col<x.shape.dim) :
    logical (realMemory s) (sourceRegion (inputs x)[port].buf) (inputStrides x port) b h row col=
      Logical.realAt xs port b h row col := by
  have value := real_memory_value x scale xs s hin port (⟨b,hb⟩,⟨h,hh⟩,⟨row,hr⟩,⟨col,hc⟩,())
  simpa only [logical,Logical.realAt,inputAt,dif_pos hb,dif_pos hh,dif_pos hr,dif_pos hc] using value

theorem score_binding (b h row key : Nat) (hb : b<x.shape.batches) (hh : h<x.shape.heads)
    (hr : row<x.shape.length) (hk : key<x.shape.length) :
    Spec.score x (realMemory s) scale b h row key=Logical.score x scale xs b h row key := by
  unfold Spec.score Logical.score
  congr 1
  apply List.map_congr_left
  intro d hd
  have hd' := List.mem_range.mp hd
  have q := real_input_logical x scale xs s hin 0 b h row d hb hh hr hd'
  have k := real_input_logical x scale xs s hin 1 b h key d hb hh hk hd'
  change logical (realMemory s) .q x.q b h row d=Logical.realAt xs 0 b h row d at q
  change logical (realMemory s) .k x.k b h key d=Logical.realAt xs 1 b h key d at k
  rw [q,k]

theorem denominator_binding (b h row : Nat) (hb : b<x.shape.batches) (hh : h<x.shape.heads)
    (hr : row<x.shape.length) :
    Spec.denominator x (realMemory s) scale b h row=Logical.denominator x scale xs b h row := by
  unfold Spec.denominator Logical.denominator
  congr 1
  apply List.map_congr_left
  intro key hk
  have keyBound : key<x.shape.length := List.mem_range.mp (List.mem_filter.mp hk).1
  rw [score_binding x scale xs s hin b h row key hb hh hr keyBound]

theorem numerator_binding (b h row col : Nat) (hb : b<x.shape.batches) (hh : h<x.shape.heads)
    (hr : row<x.shape.length) (hc : col<x.shape.dim) :
    Spec.numerator x (realMemory s) scale b h row col=Logical.numerator x scale xs b h row col := by
  unfold Spec.numerator Logical.numerator
  congr 1
  apply List.map_congr_left
  intro key hk
  have keyBound : key<x.shape.length := List.mem_range.mp (List.mem_filter.mp hk).1
  rw [score_binding x scale xs s hin b h row key hb hh hr keyBound]
  have v := real_input_logical x scale xs s hin 2 b h key col hb hh keyBound hc
  simpa [inputs,sourceRegion,inputStrides] using congrArg (fun v => pow2 (Logical.score x scale xs b h row key)*v) v

theorem real_output_formula (hs : x.shape.Valid) (ho : OutputInjective x)
    (domain : Logical.RealInputs x xs)
    (i : TileIndex [x.shape.batches,x.shape.heads,x.shape.length,x.shape.dim]) :
    tensorOutput KernelReal.algebra KernelReal.dot id x (some scale) xs i=
      some (Logical.numerator x scale xs i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val /
        Logical.denominator x scale xs i.1.val i.2.1.val i.2.2.1.val) := by
  have generic := grid_output_injective KernelReal.algebra KernelReal.dot id x (some scale) (fun _ _ => id)
    xs (liftedState s) (lifted_binding x scale xs s hin domain) hs ho i
  have old := KernelReal.grid_output x hs ho (realMemory s) scale i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val
    ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt,i.2.2.2.1.isLt⟩
  dsimp only [liftedState] at generic
  rw [generic] at old
  have value := congrArg (fun c : Cell KernelReal.R => c.read .real) old
  simpa only [Cell.read,Spec.output,numerator_binding x scale xs s hin _ _ _ _
    i.1.isLt i.2.1.isLt i.2.2.1.isLt i.2.2.2.1.isLt,
    denominator_binding x scale xs s hin _ _ _ i.1.isLt i.2.1.isLt i.2.2.1.isLt] using value

theorem real_log_formula (hs : x.shape.Valid) (domain : Logical.RealInputs x xs)
    (i : TileIndex [x.shape.batches,x.shape.heads,x.shape.length]) :
    Logical.logPayload KernelReal.algebra KernelReal.dot id x (some scale) xs i=
      some (Real.log (Logical.denominator x scale xs i.1.val i.2.1.val i.2.2.1.val)/Real.log 2) := by
  have generic := grid_log_tensor KernelReal.algebra KernelReal.dot id x (some scale) (fun _ _ => id)
    xs (liftedState s) (lifted_binding x scale xs s hin domain) i
  have old := KernelReal.grid_log x hs (realMemory s) scale i.1.val i.2.1.val i.2.2.1.val
    ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt⟩
  dsimp only [liftedState] at generic
  rw [generic] at old
  have value := congrArg (fun c : Cell KernelReal.R => c.read .real) old
  simpa only [Cell.read,Spec.lse,
    denominator_binding x scale xs s hin _ _ _ i.1.isLt i.2.1.isLt i.2.2.1.isLt] using value

#axiomsClean lifted_binding
#axiomsClean real_output_formula
#axiomsClean real_log_formula
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
