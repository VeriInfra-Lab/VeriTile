import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Contract

/-! One readable quotation of the unchanged bmm_kernel. Host constexpr mask
choices are specialized before expansion because Python None is not a runtime
mask register. Dead None assignments are omitted; the corresponding load/store
has no mask. Every nontrivial mask retains its original shape and broadcasting.
The tuple pid assignment is split. All source pointers, grid swizzle, loops,
zero fill, live FP32 accumulator and allow_tf32=False remain explicit. -/
open Lean VeriTile.Triton
namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.LegacySource
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000
syntax "bmmCheckedSource%" "[" ident "," ident "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(bmmCheckedSource%[$dm:ident, $dn:ident, $dk:ident]) => do
    for flag in [dm, dn, dk] do
      unless flag.getId == `yes || flag.getId == `no do
        Macro.throwError "divisibility flags must be yes or no"
    let m := dm.getId == `yes
    let n := dn.getId == `yes
    let k := dk.getId == `yes
    let mut before : Array (TSyntax `tritonStmt) := #[]
    if !m then before := before.push (← `(tritonStmt| mask_m = offs_m < $(M)))
    if !n then before := before.push (← `(tritonStmt| mask_n = offs_n < $(N)))
    let mut masks : Array (TSyntax `tritonStmt) := #[]
    if !k then masks := masks.push (← `(tritonStmt| mask_k = offs_k < $(K)))
    if !m && !k then masks := masks.push (← `(tritonStmt| mask_a = mask_m[:, None] & mask_k[None, :]))
    else if !m then masks := masks.push (← `(tritonStmt| mask_a = mask_m[:, None]))
    else if !k then masks := masks.push (← `(tritonStmt| mask_a = mask_k[None, :]))
    if !k && !n then masks := masks.push (← `(tritonStmt| mask_b = mask_k[:, None] & mask_n[None, :]))
    else if !k then masks := masks.push (← `(tritonStmt| mask_b = mask_k[:, None]))
    else if !n then masks := masks.push (← `(tritonStmt| mask_b = mask_n[None, :]))
    let loadA : TSyntax `tritonStmt ← if m && k then `(tritonStmt| a = tl.load(a_ptrs))
      else `(tritonStmt| a = tl.load(a_ptrs, mask=mask_a, other=0.0))
    let loadB : TSyntax `tritonStmt ← if k && n then `(tritonStmt| b = tl.load(b_ptrs))
      else `(tritonStmt| b = tl.load(b_ptrs, mask=mask_b, other=0.0))
    let add : TSyntax `tritonStmt ← `(tritonStmt| o += tl.dot(a, b, allow_tf32=false))
    let mut after : Array (TSyntax `tritonStmt) := #[]
    if !m && !n then after := after.push (← `(tritonStmt| mask_c = mask_m[:, None] & mask_n[None, :]))
    else if !m then after := after.push (← `(tritonStmt| mask_c = mask_m[:, None]))
    else if !n then after := after.push (← `(tritonStmt| mask_c = mask_n[None, :]))
    let store : TSyntax `tritonStmt ← if m && n then `(tritonStmt| tl.store(o_ptrs, o))
      else `(tritonStmt| tl.store(o_ptrs, o, mask=mask_c))
    `(fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      ComputeKernel.mk [A,B] [O] (triton {
        tl.region A = tl.float32, B = tl.float32, O = tl.float32
        pid_b = tl.program_id(2)
        A += pid_b * $(M) * $(K)
        B += pid_b * $(K) * $(N)
        O += pid_b * $(M) * $(N)
        pidx = tl.program_id(0)
        pidy = tl.program_id(1)
        if $(GROUP_M) == $(1) {
          pid_m = pidx
          pid_n = pidy
        } else {
          gridx = tl.num_programs(0)
          gridy = tl.num_programs(1)
          pid = pidx + pidy * gridx
          num_CTA_per_group = gridy * $(GROUP_M)
          group_id = pid // num_CTA_per_group
          inner_group_id = pid % num_CTA_per_group
          if (group_id * $(GROUP_M) + $(GROUP_M)) > gridx {
            GROUP_SIZE = gridx % $(GROUP_M)
          } else {
            GROUP_SIZE = $(GROUP_M)
          }
          pid_m = group_id * $(GROUP_M) + inner_group_id % GROUP_SIZE
          pid_n = inner_group_id // GROUP_SIZE
        }
        offs_m = pid_m * $(TILE_M) + tl.arange(0, $(TILE_M))
        offs_n = pid_n * $(TILE_N) + tl.arange(0, $(TILE_N))
        offs_k = tl.arange(0, $(TILE_K))
        $before*
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          $masks*
          $loadA
          $loadB
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          $add
        }
        $after*
        $store
      }).surfaceBody)

structure Parameters where
  (A B O : RegionName)
  (M N K TM TN TK GROUP : Nat)
  (divM divN divK : Bool)

def kernel (p : Parameters) : ComputeKernel :=
  let build := match p.divM,p.divN,p.divK with
    | Bool.false,Bool.false,Bool.false => bmmCheckedSource%[no, no, no]
    | Bool.false,Bool.false,Bool.true => bmmCheckedSource%[no, no, yes]
    | Bool.false,Bool.true,Bool.false => bmmCheckedSource%[no, yes, no]
    | Bool.false,Bool.true,Bool.true => bmmCheckedSource%[no, yes, yes]
    | Bool.true,Bool.false,Bool.false => bmmCheckedSource%[yes, no, no]
    | Bool.true,Bool.false,Bool.true => bmmCheckedSource%[yes, no, yes]
    | Bool.true,Bool.true,Bool.false => bmmCheckedSource%[yes, yes, no]
    | Bool.true,Bool.true,Bool.true => bmmCheckedSource%[yes, yes, yes]
  build p.A p.B p.O p.M p.N p.K p.TM p.TN p.TK p.GROUP

def parameters (s : BmmOptimized.Shape) (c : BmmOptimized.Config) : Parameters :=
  ⟨"A","B","O",s.m,s.n,s.k,c.tm,c.tn,32,c.group,
    decide (s.m%c.tm=0),decide (s.n%c.tn=0),decide (s.k%32=0)⟩

def parts (p : Parameters) : VeriTile.Triton.Source.DynamicRange :=
  (VeriTile.Triton.Source.dynamicRange? (kernel p).surfaceBody "_i").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
def prelude (p : Parameters) := (parts p).before
def body (p : Parameters) := (parts p).body
def epilogue (p : Parameters) := (parts p).after

theorem source_decomposition (p : Parameters) :
    (kernel p).surfaceBody = prelude p ++
      [.forRangeDyn "_i" (.constNat 0) (.ref .nat [] "num_iters") (.constNat 1) (body p)] ++ epilogue p := by
  rcases p with ⟨A,B,O,M,N,K,TM,TN,TK,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk <;> rfl

#axiomsClean source_decomposition
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.LegacySource
