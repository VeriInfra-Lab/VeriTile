import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Prefix

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def batch {α : Type} (p : Parameters) (s : State α) := s.pids 1 / p.heads
def head {α : Type} (p : Parameters) (s : State α) := s.pids 1 % p.heads
def kv {α : Type} (p : Parameters) (s : State α) := head p s / (p.heads / p.kvHeads)
def layout {α : Type} (p : Parameters) (s : State α) := head p s % p.layouts
def row {α : Type} (s : State α) (i : Fin 16) := s.pids 0 * 16 + i.val
def channel {α : Type} (s : State α) (j : Fin 16) := s.pids 2 * 16 + j.val
def qPointer {α : Type} (p : Parameters) (s : State α) (i : TileIndex [16,16]) :=
  (batch p s * p.qb + head p s * p.qh) + (row s i.1 * p.qm + i.2.1.val * p.qd)
def kPointer {α : Type} (p : Parameters) (s : State α) (i : TileIndex [16,16]) :=
  (batch p s * p.kb + kv p s * p.kh) + (i.2.1.val * p.kn + i.1.val * p.kd)
def vPointer {α : Type} (p : Parameters) (s : State α) (i : TileIndex [16,16]) :=
  (batch p s * p.vb + kv p s * p.vh) + (i.1.val * p.vn + channel s i.2.1 * p.vd)
def rowPointer {α : Type} (p : Parameters) (s : State α) :=
  layout p s * p.rowStride + s.pids 0
def first {α : Type} [Inhabited α] (p : Parameters) (s : State α) :=
  (s.mem "rows" (rowPointer p s)).read .int
def last {α : Type} [Inhabited α] (p : Parameters) (s : State α) :=
  (s.mem "rows" (rowPointer p s + 1)).read .int

structure Indices {α : Type} [Inhabited α] (a : Arithmetic α) (p : Parameters)
    (initial current : State (Replicas α)) : Prop where
  scale : current.regs .real [] "softmax_scale" = some (fun _ => broadcast (a.argument p.scale))
  memory : current.mem = initial.mem
  length : current.regs .nat [] "q_seq_len" = some (fun _ => p.length)
  batch : current.regs .nat [] "off_b" = some (fun _ => batch p initial)
  head : current.regs .nat [] "off_h" = some (fun _ => head p initial)
  channel : current.regs .nat [] "channel_block" = some (fun _ => initial.pids 2)
  rows : current.regs .nat [16] "offs_m" = some (fun i => row initial i.1)
  keys : current.regs .nat [16] "offs_n" = some (fun i => i.1.val)
  dimensions : current.regs .nat [16] "offs_d" = some (fun i => i.1.val)
  qptrs : current.regs .ptr [16,16] "q_ptrs" = some (fun i => (("Q" : RegionName),qPointer p initial i))
  kptrs : current.regs .ptr [16,16] "k_ptrs" = some (fun i => (("K" : RegionName),kPointer p initial i))
  vptrs : current.regs .ptr [16,16] "v_ptrs" = some (fun i => (("V" : RegionName),vPointer p initial i))
  layout : current.regs .nat [] "layout_h" = some (fun _ => layout p initial)
  first : current.regs .int [] "start_l" = some (fun _ => first p initial)
  last : current.regs .int [] "end_l" = some (fun _ => last p initial)

theorem prefixIndices {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (s : State (Replicas α)) (integers : Int32Memory s) :
    Indices a p s (prefixExecution a.model p (bindScale (broadcast (a.argument p.scale)) s) integers).val := by
  constructor <;>
    simp [prefixExecution,bindScale,batch,head,kv,layout,row,channel,qPointer,kPointer,vPointer,
      first,last,rowPointer,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    funext i
    simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

#axiomsClean prefixIndices
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
