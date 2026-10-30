import bench.tritonbench_g.layer_norm_liger.Kernels
import bench.tritonbench_g.layer_norm_liger.Proofs.Settings
import VeriTile.Torch.FP32
import VeriTile.Triton.Float.Structural

/-! Kernel-entry contract. These are the actual prepared arguments at the two
Triton calls. Contiguous preparation, allocation, launch delivery and the Torch
sum after backward belong to the retained host result. No input copy is made. -/
namespace VeriTile.Bench.Repairs.LayerNormLiger.KernelContract
open VeriTile.Triton FP.Structural
open VeriTile.Torch
noncomputable section

structure Buffer where
  tensor : Torch.FP32.Tensor
  region : RegionName
  /-- Actual stride(0) API observation; singleton strides are not normalized. -/
  stride0 : Nat

def Buffer.size (b : Buffer) := b.tensor.metadata.numel
def Buffer.pointer (b : Buffer) := b.tensor.dataPtr.toNat
def Buffer.logical (b : Buffer) (memory : Nat → α) (i : Nat) : α :=
  memory (b.tensor.storage.address (b.tensor.physicalIndex i))

/-- Direct prepared argument. Alignment is checked on data_ptr; no
backing-allocation alignment or element-finiteness restriction is added. -/
structure Valid (b : Buffer) (shape : List Nat) : Prop where
  tensor : Torch.FP32.TensorLaws b.tensor
  shape_eq : b.tensor.metadata.shape = shape
  contiguous : b.tensor.metadata.isContiguous = Bool.true
  noNeg : b.tensor.metadata.isNeg = Bool.false
  cuda : b.tensor.metadata.device.kind = "cuda"
  aligned : 0 < b.size → b.pointer % 4 = 0

/-- Ordinary matrix-view indexing law for the observed row stride. The source
never queries column stride: contiguity supplies its active unit-column map.
This permits arbitrary singleton row strides, including zero and huge values. -/
def RowStride (b : Buffer) (rows columns : Nat) : Prop :=
  ∀ row col, row < rows → col < columns →
    b.tensor.physicalIndex (row*columns+col) =
      b.tensor.storageOffset + row*b.stride0 + col

abbrev Location := RegionName × Nat
def address (pointers : RegionName → Nat) (l : Location) := pointers l.1 + 4*l.2

/-- One physical heap supplies every pointer, preserving read-only aliases. -/
def entry [Inhabited α] (pointers : RegionName → Nat) (memory : Nat → α) (pid : Nat) : State α where
  regs := fun _ _ _ => none
  mem := fun r o => .mk .fp32 (memory (address pointers (r,o)))
  pids := fun _ => pid
  numPids := fun _ => 0
  undef := fun dtype _ _ => defaultValue dtype

structure Bound (pointers : RegionName → Nat) (b : Buffer) : Prop where
  pointer : pointers b.region = b.pointer

/-- Read-only specification overlay. The binding theorem proves it is the
same initial state; it is not a materialization or input copy. -/
def logicalEntry (inputs : List Buffer) (memory : Nat → α) (s : State α) : State α :=
  { s with mem := fun r o =>
      match inputs.find? (fun b => b.region == r && o < b.size) with
      | some b => .mk .fp32 (b.logical memory o)
      | none => s.mem r o }

/-- Physical realization of precisely the actual source writes. Address
injectivity is an ordinary memory-separation obligation. -/
def commit [Inhabited α] (pointers : RegionName → Nat) (writes : Location → Prop)
    (initial : Nat → α) (final : State α) (a : Nat) : α := by
  classical
  exact if h : ∃ l, writes l ∧ address pointers l = a
    then (final.mem h.choose.1 h.choose.2).read .fp32 else initial a

structure ForwardCall where
  (x weight bias y mean rstd : Buffer)
  (rows columns block pid : Nat)
  epsilon : ℝ

def ForwardCall.parameters (c : ForwardCall) : Forward.Parameters :=
  ⟨c.y.region,c.x.region,c.weight.region,c.bias.region,c.mean.region,c.rstd.region,
    c.y.stride0,c.x.stride0,c.weight.stride0,c.bias.stride0,c.mean.stride0,c.rstd.stride0,c.columns,c.block,c.epsilon⟩
def ForwardCall.inputs (c : ForwardCall) := [c.x,c.weight,c.bias]
def ForwardCall.buffers (c : ForwardCall) := c.inputs ++ [c.y,c.mean,c.rstd]
def ForwardCall.writes (c : ForwardCall) (l : Location) : Prop :=
  l = (c.mean.region,c.pid) ∨ l = (c.rstd.region,c.pid) ∨
    ∃ col, col < c.columns ∧ l = (c.y.region,c.pid*c.columns+col)

structure ForwardDomain (c : ForwardCall) (pointers : RegionName → Nat) : Prop where
  x : Valid c.x [c.rows,c.columns]
  weight : Valid c.weight [c.columns]
  bias : Valid c.bias [c.columns]
  y : Valid c.y [c.rows,c.columns]
  mean : Valid c.mean [c.rows]
  rstd : Valid c.rstd [c.rows]
  xStride : RowStride c.x c.rows c.columns
  /-- These buffers are the actual canonical torch.empty allocations. -/
  yStride : c.y.stride0 = c.columns
  meanStride : c.mean.stride0 = 1
  rstdStride : c.rstd.stride0 = 1
  bound : ∀ b ∈ c.buffers, Bound pointers b
  sameDevice : ∀ b ∈ c.buffers, b.tensor.metadata.device = c.x.tensor.metadata.device
  columns_pos : 0 < c.columns
  selectedBlock : c.block = Settings.block c.columns
  block_covers : c.columns ≤ c.block
  block_bound : c.block ≤ 65536
  pid_bound : c.pid < c.rows
  epsilon_pos : 0 < c.epsilon
  distinct : c.mean.region ≠ c.y.region ∧ c.mean.region ≠ c.rstd.region ∧ c.rstd.region ≠ c.y.region
  physicalWrites : ∀ a b, c.writes a → c.writes b → address pointers a = address pointers b → a = b

structure BackwardCall where
  (x weight mean rstd dx dw db dy : Buffer)
  (rows columns block group programs pid : Nat)

def BackwardCall.parameters (c : BackwardCall) : Backward.Parameters :=
  ⟨c.x.region,c.weight.region,c.mean.region,c.rstd.region,c.dx.region,c.dw.region,c.db.region,c.dy.region,
    c.x.stride0,c.dx.stride0,c.dw.stride0,c.db.stride0,c.dy.stride0,c.rows,c.columns,c.group,c.block⟩
def BackwardCall.first (c : BackwardCall) := c.pid*c.group
def BackwardCall.last (c : BackwardCall) := min ((c.pid+1)*c.group) c.rows
def BackwardCall.inputs (c : BackwardCall) := [c.x,c.weight,c.dy,c.mean,c.rstd]
def BackwardCall.buffers (c : BackwardCall) := c.inputs ++ [c.dx,c.dw,c.db]
def BackwardCall.writes (c : BackwardCall) (l : Location) : Prop :=
  (∃ col, col < c.columns ∧ l = (c.dw.region,c.pid*c.columns+col)) ∨
  (∃ col, col < c.columns ∧ l = (c.db.region,c.pid*c.columns+col)) ∨
  ∃ row col, c.first ≤ row ∧ row < c.last ∧ col < c.columns ∧ l = (c.dx.region,row*c.columns+col)

structure BackwardDomain (c : BackwardCall) (pointers : RegionName → Nat) : Prop where
  x : Valid c.x [c.rows,c.columns]
  weight : Valid c.weight [c.columns]
  dy : Valid c.dy [c.rows,c.columns]
  mean : Valid c.mean [c.rows]
  rstd : Valid c.rstd [c.rows]
  dx : Valid c.dx [c.rows,c.columns]
  dw : Valid c.dw [c.programs,c.columns]
  db : Valid c.db [c.programs,c.columns]
  xStride : RowStride c.x c.rows c.columns
  dyStride : RowStride c.dy c.rows c.columns
  /-- Actual torch.empty allocation strides; no input-stride restriction. -/
  dxStride : c.dx.stride0 = c.columns
  dwStride : c.dw.stride0 = c.columns
  dbStride : c.db.stride0 = c.columns
  bound : ∀ b ∈ c.buffers, Bound pointers b
  sameDevice : ∀ b ∈ c.buffers, b.tensor.metadata.device = c.x.tensor.metadata.device
  columns_pos : 0 < c.columns
  selectedBlock : c.block = Settings.block c.columns
  block_covers : c.columns ≤ c.block
  block_bound : c.block ≤ 65536
  rows_pos : 0 < c.rows
  programs_pos : 0 < c.programs
  pid_bound : c.pid < c.programs
  grouping : c.group = (c.rows+c.programs-1)/c.programs
  distinct : c.dw.region ≠ c.db.region ∧ c.dx.region ≠ c.dw.region ∧ c.dx.region ≠ c.db.region
  fresh : ∀ b ∈ c.inputs, b.region ≠ c.dx.region
  physicalWrites : ∀ a b, c.writes a → c.writes b → address pointers a = address pointers b → a = b
  physicalInputs : ∀ b ∈ c.inputs, ∀ i, i < b.size → ∀ l, c.writes l →
    address pointers l ≠ b.pointer + 4*i

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.LayerNormLiger.KernelContract
