import bench.tritonbench_g.flash_attn.Proofs.SourceSyntax
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.FlashAttn.ActualSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Private total primitive adapter. Public native endpoints instantiate this
with explicit fixed-width K8 instruction chains. -/
structure Dot (α : Type) where
  apply : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    Values α left.toTileDType (batch++[m,k]) →
    Values α right.toTileDType (batch++[k,n]) →
    Values α out.toTileDType (batch++[m,n])

def withDot (M : Algebra α) (D : Dot α) : Algebra α :=
  { M with
    dot := fun p {_} {_} {_} {_} a b=>some (D.apply p .real .real .real {} a b)
    dotWithOptions := fun p opts {_} {_} {_} {_} a b=>some (D.apply p .real .real .real opts a b)
    dotTyped := fun p l r o opts {_} {_} {_} {_} a b=>some (D.apply p l r o opts a b) }

def bindScale (scale : α) (s : State α) : State α :=
  s.setReg "sm_scale" .real [] (fun _=>scale)

def zero (M : Algebra α) : α := M.literal (some .fp32) .real 0

def base (p : Parameters) (s : State α) (bs hs : Nat) : Nat :=
  s.pids 1/p.HEAD*bs+s.pids 1%p.HEAD*hs

def qptr (p : Parameters) (s : State α) : BlockPtr :=
  ⟨"Q",base p s p.stride_q_bs p.stride_q_head,[p.SEQLEN,p.DIM],[128,p.DIM],
    [p.stride_q_seqlen,p.stride_q_dim],[s.pids 0*128,0]⟩
def kptr (p : Parameters) (s : State α) (row : Nat) : BlockPtr :=
  ⟨"K",base p s p.stride_k_bs p.stride_k_head,[p.DIM,p.SEQLEN],[p.DIM,64],
    [p.stride_k_dim,p.stride_k_seqlen],[0,row]⟩
def vptr (p : Parameters) (s : State α) (row : Nat) : BlockPtr :=
  ⟨"V",base p s p.stride_v_bs p.stride_v_head,[p.SEQLEN,p.DIM],[64,p.DIM],
    [p.stride_v_seqlen,p.stride_v_dim],[row,0]⟩
def optr (p : Parameters) (s : State α) : BlockPtr :=
  ⟨"O",base p s p.stride_o_bs p.stride_o_head,[p.SEQLEN,p.DIM],[128,p.DIM],
    [p.stride_o_seqlen,p.stride_o_dim],[s.pids 0*128,0]⟩

def query (M : Algebra α) (p : Parameters) (s : State α) (scale : α)
    (i : TileIndex [128,p.DIM]) : α :=
  M.cast (some .fp32) .real .fp16
    (M.binary (some .fp32) .real .mul
      (M.fp32Load (if (qptr p s).inBounds (TileShape.indexToList _ i) [0] then
        (s.mem "Q" (base p s p.stride_q_bs p.stride_q_head+
          (s.pids 0*128+i.1.val)*p.stride_q_seqlen+i.2.1.val*p.stride_q_dim)).read .real
       else zero M))
      (M.binary (some .fp32) .real .mul scale (M.literal (some .fp32) .real 1.44269504)))

structure Acc (p : Parameters) (α : Type) where
  maximum : Fin 128→α
  normalizer : Fin 128→α
  output : TileIndex [128,p.DIM]→α

def initialAcc (M : Algebra α) (p : Parameters) : Acc p α :=
  ⟨fun _=>M.binary (some .fp32) .real .add (zero M) M.negInf,
   fun _=>zero M,fun _=>zero M⟩

structure Context (M : Algebra α) (p : Parameters) (initial : State α) (scale : α)
    (row : Nat) (a : Acc p α) (current : State α) : Prop where
  memory : current.mem=initial.mem
  startM : current.regs .nat [] "start_m"=some (fun _=>initial.pids 0)
  head : current.regs .nat [] "off_bs_head"=some (fun _=>initial.pids 1)
  rows : current.regs .nat [128] "off_m"=some (fun i=>initial.pids 0*128+i.1.val)
  columns : current.regs .nat [64] "off_n"=some (fun i=>i.1.val)
  outputOffset : current.regs .nat [] "o_base_offset"=
    some (fun _=>base p initial p.stride_o_bs p.stride_o_head)
  query : current.regs .fp16 [128,p.DIM] "q"=some (query M p initial scale)
  keyPointer : current.regs .blockPtr [p.DIM,64] "K_block_ptr"=some (fun _=>kptr p initial row)
  valuePointer : current.regs .blockPtr [64,p.DIM] "V_block_ptr"=some (fun _=>vptr p initial row)
  maximum : current.regs .real [128] "max"=some (fun i=>a.maximum i.1)
  normalizer : current.regs .real [128] "denom"=some (fun i=>a.normalizer i.1)
  output : current.regs .real [128,p.DIM] "out_buffer"=some a.output

end
end VeriTile.Bench.Repairs.FlashAttn.ActualSource
