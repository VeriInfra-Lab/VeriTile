import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelLoop
import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelStore
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.SourceContract

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def loopCode (fp8 : Bool) (p : Parameters) := ((kernel fp8 p).surfaceBody.drop 12).take 1

theorem source_split (fp8 : Bool) (p : Parameters) :
    (kernel fp8 p).surfaceBody=prelude fp8 p++loopCode fp8 p++statistics fp8 p++
      rotation fp8 p++stores fp8 p := by cases fp8 <;> rfl

theorem loopCode_run (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α) :
    run (engine a) (loopCode fp8 p) s=
      range (engine a) "step" 0 (steps p) 1 (loop fp8 p) s := by
  cases fp8 <;> simp [loopCode,loop,kernel,TypedSource.kernel,TypedSource.source,
    arguments,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,numeric,bop,integralFloorDiv,integralMod,steps,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals split <;> simp_all only

def result (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α) :=
  rotated a p s (normalized a p (accumulated a fp8 p s))

/-- Success of the actual canonical source, including every dynamic K block,
optional rotary branch and final typed masked store. No execution premise. -/
theorem source_executes_reads (a : Operations α) (fp8 : Bool) (p : Parameters)
    (s : State α) (bytes : ∀ cur, cur<steps p → BlockBytes fp8 p s cur) :
    ∃ final, exec (engine a) (kernel fp8 p) s=some final ∧
      final.mem=(scatter a fp8 p s s (result a fp8 p s)).mem := by
  obtain ⟨u,hu,ru⟩ := prefix_run a fp8 p s
  obtain ⟨v,hv,rv⟩ := loop_run_reads a fp8 p s u 0 (steps p) (initial a) (fun cur _ hcur => bytes cur hcur) ru
  obtain ⟨w,hw,rw⟩ := statistics_run a fp8 p s v _ _ rv
  obtain ⟨x,hx,rx⟩ := rotation_run a fp8 p s w _ rw
  obtain ⟨final,hf,mf⟩ := store_run a fp8 p s x _ rx
  refine ⟨final,?_,mf⟩
  change run (engine a) (kernel fp8 p).surfaceBody s=some final
  rw [source_split]
  simp only [run_append,hu,Option.bind_some,loopCode_run,hv,hw,hx,hf]

/-- The original all-offset ByteMemory endpoint remains available. -/
theorem source_executes (a : Operations α) (fp8 : Bool) (p : Parameters)
    (s : State α) (bytes : ByteMemory s) :
    ∃ final, exec (engine a) (kernel fp8 p) s=some final ∧
      final.mem=(scatter a fp8 p s s (result a fp8 p s)).mem :=
  source_executes_reads a fp8 p s (fun cur _ => blockBytes_of_byteMemory fp8 p s cur bytes)

def Outside (p : Parameters) (s : State α) (region : RegionName) (off : Nat) : Prop :=
  region≠"OUT" ∨ ∀ i, active p s i=Bool.true → off≠outputOffset p s i

theorem scatter_frame (a : Operations α) (fp8 : Bool) (p : Parameters)
    (s base : State α) (v : TileIndex [16,64] → α) (r : RegionName) (off : Nat)
    (outside : Outside p s r off) : (scatter a fp8 p s base v).mem r off=base.mem r off := by
  apply State.masked_scatter_frame
  rcases outside with hr|ho
  · exact .inl hr
  · exact .inr (fun i _ hi => ho i hi)

theorem outputOffset_injective (p : Parameters) (s : State α)
    (i j : TileIndex [16,64]) (hi : active p s i=Bool.true) (hj : active p s j=Bool.true)
    (he : outputOffset p s i=outputOffset p s j) : i=j := by
  have ai : row p s i.1<p.M ∧ col p s i.2.1<p.N := by simpa [active] using hi
  have aj : row p s j.1<p.M ∧ col p s j.2.1<p.N := by simpa [active] using hj
  have ci := ai.2
  have cj := aj.2
  have equal : row p s i.1*p.N+col p s i.2.1=row p s j.1*p.N+col p s j.2.1 := by
    unfold outputOffset at he
    omega
  have rows : row p s i.1=row p s j.1 := by
    by_contra hn
    rcases lt_or_gt_of_ne hn with h|h
    · have hle := Nat.succ_le_of_lt h; nlinarith
    · have hle := Nat.succ_le_of_lt h; nlinarith
  have columns : col p s i.2.1=col p s j.2.1 := by rw [rows] at equal; omega
  rcases i with ⟨i,k,⟨⟩⟩
  rcases j with ⟨j,l,⟨⟩⟩
  have rij : i=j := Fin.ext (by simpa [row] using rows)
  have ckl : k=l := Fin.ext (by simpa [col] using columns)
  simp [rij,ckl]

theorem scatter_readback (a : Operations α) (fp8 : Bool) (p : Parameters)
    (s base : State α) (v : TileIndex [16,64] → α)
    (i : TileIndex [16,64]) (hi : active p s i=Bool.true) :
    (scatter a fp8 p s base v).mem "OUT" (outputOffset p s i)=outputCell a fp8 (v i) :=
  LayernormFwdTriton.Source.masked_readback "OUT" (outputOffset p s)
    (fun j => outputCell a fp8 (v j)) (fun j => active p s j=Bool.true) base i hi
    (outputOffset_injective p s)

theorem source_complete (a : Operations α) (fp8 : Bool) (p : Parameters)
    (s : State α) (bytes : ByteMemory s) :
    ∃ final, exec (engine a) (kernel fp8 p) s=some final ∧
      (∀ i, active p s i=Bool.true →
        final.mem "OUT" (outputOffset p s i)=outputCell a fp8 (result a fp8 p s i)) ∧
      (∀ r off, Outside p s r off → final.mem r off=s.mem r off) := by
  obtain ⟨final,run,mem⟩ := source_executes a fp8 p s bytes
  exact ⟨final,run,fun i hi => by rw [mem]; exact scatter_readback a fp8 p s s _ i hi,
    fun r off ho => by rw [mem]; exact scatter_frame a fp8 p s s _ r off ho⟩

theorem source_complete_reads (a : Operations α) (fp8 : Bool) (p : Parameters)
    (s : State α) (bytes : ∀ cur, cur<steps p → BlockBytes fp8 p s cur) :
    ∃ final, exec (engine a) (kernel fp8 p) s=some final ∧
      (∀ i, active p s i=Bool.true →
        final.mem "OUT" (outputOffset p s i)=outputCell a fp8 (result a fp8 p s i)) ∧
      (∀ r off, Outside p s r off → final.mem r off=s.mem r off) := by
  obtain ⟨final,run,mem⟩ := source_executes_reads a fp8 p s bytes
  exact ⟨final,run,fun i hi => by rw [mem]; exact scatter_readback a fp8 p s s _ i hi,
    fun r off ho => by rw [mem]; exact scatter_frame a fp8 p s s _ r off ho⟩

#axiomsClean source_executes_reads
#axiomsClean source_complete_reads

#axiomsClean source_executes
#axiomsClean source_complete
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
