import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.RealIO
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.RealBackingTransport

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealFiniteContext
open VeriTile Triton Source
open VeriTile.Bench.Repairs.AttentionLlama LoopInvariant HostGrid HostResult MaskCoverage
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- Reuse the accepted mathematical proof context without changing state. -/
def context (c : Config) (scale : ℝ) (s : BlockState) : Context :=
  ⟨s,c.Q,c.K,c.V,scale,c.sqz,c.sqh,c.sqm,c.sqk,c.skz,c.skh,c.skn,c.skk,
    c.svz,c.svh,c.svk,c.svn,c.NH,c.H,c.N,c.SP,c.D,c.causal⟩

@[simp] theorem config_context (c : Config) (scale : ℝ) (s : BlockState) :
    Backing.realConfig (context c scale s) c.Out c.soz c.soh c.som c.son c.fp8=c := by
  cases c
  rfl

def tile (i : Index B NH H D) : TileIndex [B,NH,H,D] :=
  (i.batch,i.head,i.query,i.channel,())
def index (i : TileIndex [B,NH,H,D]) : Index B NH H D :=
  ⟨i.1,i.2.1,i.2.2.1,i.2.2.2.1⟩

@[simp] theorem tile_index (i : TileIndex [B,NH,H,D]) : tile (index i)=i := by
  rcases i with ⟨b,h,q,d,u⟩
  cases u
  rfl
@[simp] theorem index_tile (i : Index B NH H D) : index (tile i)=i := by cases i; rfl

theorem base_eq (c : Config) (scale : ℝ) (s : BlockState)
    (i : Index B c.NH c.H c.D) (sz sh : Nat) :
    alBase (contextFor (context c scale s) i).initial c.NH sz sh=
      i.batch.val*sz+i.head.val*sh := by
  have hb:=batch_head_roundtrip c.NH i.batch.val i.head.val i.head.isLt
  simp only [alBase,contextFor,context,↓reduceIte,Nat.one_ne_zero,hb.1,hb.2]

theorem inputBinding_of_mem (B : Nat) (c : Config) (vals : (RealLayout.memory B c).Ctx)
    (s t : BlockState) (hm : t.mem=s.mem)
    (hin : (RealLayout.memory B c).InputBinding s vals) :
    (RealLayout.memory B c).InputBinding t vals := by
  intro i j active
  have h:=hin i j active
  fin_cases i <;> simpa [RealLayout.memory,ChanTy.read,BlockState.readMem,
    BlockState.readMemValue,BlockState.readMemTyped,hm] using h

end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealFiniteContext
