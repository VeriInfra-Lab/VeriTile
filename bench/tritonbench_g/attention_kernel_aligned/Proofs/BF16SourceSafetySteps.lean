import bench.tritonbench_g.attention_kernel_aligned.Proofs.SourceSafetyRun
import bench.tritonbench_g.attention_kernel_aligned.Proofs.BF16Complete

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.SourceSafety
open VeriTile Triton FP.Structural Safety Source Full FPValue
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

alias pure_safe := VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety.pure_safe
alias safeBlockPtr2 := VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety.safeBlockPtr2
abbrev queryPrefix := VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety.queryPrefix
def prefixResult (P : Primitives α) (p : Parameters) (s : State α) : State α :=
  let t := s.setReg "start_m" .nat [] (fun _ => s.pids 0)
  let t := t.setReg "off_hz" .nat [] (fun _ => s.pids 1)
  let t := t.setReg "off_b" .nat [] (fun _ => s.pids 1/p.heads)
  let t := t.setReg "off_h" .nat [] (fun _ => s.pids 1%p.heads)
  let t := t.setReg "q_offset" .nat [] (fun _ => headBase p s)
  let t := t.setReg "k_offset" .nat [] (fun _ => kvBase p s)
  let t := t.setReg "v_offset" .nat [] (fun _ => valueBase p s)
  let t := t.setReg "o_offset" .nat [] (fun _ => outputBase p s)
  let t := t.setReg "Q_block_ptr" .blockPtr [p.blockM,p.head] (fun _ => qPointer p s)
  let t := t.setReg "K_block_ptr" .blockPtr [p.head,p.blockN] (fun _ => kPointer p s 0)
  let t := t.setReg "V_block_ptr" .blockPtr [p.blockN,p.head] (fun _ => vPointer p s 0)
  let t := t.setReg "m_i" .real [p.blockM] (initialAccumulator P p).maximum
  let t := t.setReg "l_i" .real [p.blockM] (initialAccumulator P p).denominator
  let t := t.setReg "acc" .real [p.blockM,p.head] (initialAccumulator P p).numerator
  let t := t.setReg "qk_scale" .real [] (fun _ => scaleValue P p)
  t


theorem queryPrefix_source (p : Parameters) : queryPrefix p=(prologue p).take 15 := rfl

def queryLoad (p : Parameters) := ((prologue p).drop 15).take 1
def biasPrefix (p : Parameters) := ((prologue p).drop 16).take 6
def biasLoad (p : Parameters) := (prologue p).drop 22

def middleResult (P : Primitives α) (p : Parameters) (s current : State α) : State α :=
  let t := current.setReg "q" .bf16 [p.blockM,p.head] (qScaled P p s)
  let t := t.setReg "lo" .nat [] (fun _ => 0)
  let t := t.setReg "hi" .nat [] (fun _ => p.length)
  let t := t.setReg "b_ptr_offsets_m" .nat [p.blockM] (fun i => i.1.val)
  let t := t.setReg "b_offset" .nat [] (fun _ => biasBase p s)
  t.setReg "b_ptr_offsets_n_1" .nat [p.blockN] (fun i => i.1.val%p.biasLast+p.biasLast)

theorem prefix_safe (A : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (backend A P) bounds (queryPrefix p) s :=
  VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety.prefix_safe A P p bounds s

theorem prefix_run (A : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    run (backend A P) (queryPrefix p) s=some (prefixResult P p s) :=
  VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety.prefix_run A P p s

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.SourceSafety
