import VeriTile.Triton.DSL
import bench.tritonbench_g.bmm_optimized.Contract

/-! Complete ordinary quotations for all eight constexpr mask choices.
The integer-CSE candidate exposes the quotient already used for group_id in
its remainder expression. The selected live-FMA profile and all FP operations,
loops, pointers, mask shapes and allow_tf32=False are preserved verbatim. -/
namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Source
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

structure Parameters where
  (A B O : RegionName)
  (M N K TM TN TK GROUP : Nat)
  (divM divN divK : Bool)

def kernel (p : Parameters) : ComputeKernel :=
  let build := match p.divM,p.divN,p.divK with
    | Bool.false,Bool.false,Bool.false =>
      fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      triton {
        tl.ports inputs=[A,B], outputs=[O]
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
        mask_m = offs_m < $(M)
        mask_n = offs_n < $(N)
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          mask_k = offs_k < $(K)
          mask_a = mask_m[:, None] & mask_k[None, :]
          mask_b = mask_k[:, None] & mask_n[None, :]
          a = tl.load(a_ptrs, mask=mask_a, other=0.0)
          b = tl.load(b_ptrs, mask=mask_b, other=0.0)
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          o += tl.dot(a, b, allow_tf32=false)
        }
        mask_c = mask_m[:, None] & mask_n[None, :]
        tl.store(o_ptrs, o, mask=mask_c)
      }
    | Bool.false,Bool.false,Bool.true =>
      fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      triton {
        tl.ports inputs=[A,B], outputs=[O]
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
        mask_m = offs_m < $(M)
        mask_n = offs_n < $(N)
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          mask_a = mask_m[:, None]
          mask_b = mask_n[None, :]
          a = tl.load(a_ptrs, mask=mask_a, other=0.0)
          b = tl.load(b_ptrs, mask=mask_b, other=0.0)
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          o += tl.dot(a, b, allow_tf32=false)
        }
        mask_c = mask_m[:, None] & mask_n[None, :]
        tl.store(o_ptrs, o, mask=mask_c)
      }
    | Bool.false,Bool.true,Bool.false =>
      fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      triton {
        tl.ports inputs=[A,B], outputs=[O]
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
        mask_m = offs_m < $(M)
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          mask_k = offs_k < $(K)
          mask_a = mask_m[:, None] & mask_k[None, :]
          mask_b = mask_k[:, None]
          a = tl.load(a_ptrs, mask=mask_a, other=0.0)
          b = tl.load(b_ptrs, mask=mask_b, other=0.0)
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          o += tl.dot(a, b, allow_tf32=false)
        }
        mask_c = mask_m[:, None]
        tl.store(o_ptrs, o, mask=mask_c)
      }
    | Bool.false,Bool.true,Bool.true =>
      fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      triton {
        tl.ports inputs=[A,B], outputs=[O]
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
        mask_m = offs_m < $(M)
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          mask_a = mask_m[:, None]
          a = tl.load(a_ptrs, mask=mask_a, other=0.0)
          b = tl.load(b_ptrs)
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          o += tl.dot(a, b, allow_tf32=false)
        }
        mask_c = mask_m[:, None]
        tl.store(o_ptrs, o, mask=mask_c)
      }
    | Bool.true,Bool.false,Bool.false =>
      fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      triton {
        tl.ports inputs=[A,B], outputs=[O]
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
        mask_n = offs_n < $(N)
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          mask_k = offs_k < $(K)
          mask_a = mask_k[None, :]
          mask_b = mask_k[:, None] & mask_n[None, :]
          a = tl.load(a_ptrs, mask=mask_a, other=0.0)
          b = tl.load(b_ptrs, mask=mask_b, other=0.0)
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          o += tl.dot(a, b, allow_tf32=false)
        }
        mask_c = mask_n[None, :]
        tl.store(o_ptrs, o, mask=mask_c)
      }
    | Bool.true,Bool.false,Bool.true =>
      fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      triton {
        tl.ports inputs=[A,B], outputs=[O]
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
        mask_n = offs_n < $(N)
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          mask_b = mask_n[None, :]
          a = tl.load(a_ptrs)
          b = tl.load(b_ptrs, mask=mask_b, other=0.0)
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          o += tl.dot(a, b, allow_tf32=false)
        }
        mask_c = mask_n[None, :]
        tl.store(o_ptrs, o, mask=mask_c)
      }
    | Bool.true,Bool.true,Bool.false =>
      fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      triton {
        tl.ports inputs=[A,B], outputs=[O]
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
        
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          mask_k = offs_k < $(K)
          mask_a = mask_k[None, :]
          mask_b = mask_k[:, None]
          a = tl.load(a_ptrs, mask=mask_a, other=0.0)
          b = tl.load(b_ptrs, mask=mask_b, other=0.0)
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          o += tl.dot(a, b, allow_tf32=false)
        }
        
        tl.store(o_ptrs, o)
      }
    | Bool.true,Bool.true,Bool.true =>
      fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      triton {
        tl.ports inputs=[A,B], outputs=[O]
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
        
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          
          a = tl.load(a_ptrs)
          b = tl.load(b_ptrs)
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          o += tl.dot(a, b, allow_tf32=false)
        }
        
        tl.store(o_ptrs, o)
      }
  build p.A p.B p.O p.M p.N p.K p.TM p.TN p.TK p.GROUP

def quotientRemainder (p : Parameters) : ComputeKernel :=
  let build := match p.divM,p.divN,p.divK with
    | Bool.false,Bool.false,Bool.false =>
      fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      triton {
        tl.ports inputs=[A,B], outputs=[O]
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
          inner_group_id = pid - (pid // num_CTA_per_group) * num_CTA_per_group
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
        mask_m = offs_m < $(M)
        mask_n = offs_n < $(N)
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          mask_k = offs_k < $(K)
          mask_a = mask_m[:, None] & mask_k[None, :]
          mask_b = mask_k[:, None] & mask_n[None, :]
          a = tl.load(a_ptrs, mask=mask_a, other=0.0)
          b = tl.load(b_ptrs, mask=mask_b, other=0.0)
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          o += tl.dot(a, b, allow_tf32=false)
        }
        mask_c = mask_m[:, None] & mask_n[None, :]
        tl.store(o_ptrs, o, mask=mask_c)
      }
    | Bool.false,Bool.false,Bool.true =>
      fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      triton {
        tl.ports inputs=[A,B], outputs=[O]
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
          inner_group_id = pid - (pid // num_CTA_per_group) * num_CTA_per_group
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
        mask_m = offs_m < $(M)
        mask_n = offs_n < $(N)
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          mask_a = mask_m[:, None]
          mask_b = mask_n[None, :]
          a = tl.load(a_ptrs, mask=mask_a, other=0.0)
          b = tl.load(b_ptrs, mask=mask_b, other=0.0)
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          o += tl.dot(a, b, allow_tf32=false)
        }
        mask_c = mask_m[:, None] & mask_n[None, :]
        tl.store(o_ptrs, o, mask=mask_c)
      }
    | Bool.false,Bool.true,Bool.false =>
      fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      triton {
        tl.ports inputs=[A,B], outputs=[O]
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
          inner_group_id = pid - (pid // num_CTA_per_group) * num_CTA_per_group
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
        mask_m = offs_m < $(M)
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          mask_k = offs_k < $(K)
          mask_a = mask_m[:, None] & mask_k[None, :]
          mask_b = mask_k[:, None]
          a = tl.load(a_ptrs, mask=mask_a, other=0.0)
          b = tl.load(b_ptrs, mask=mask_b, other=0.0)
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          o += tl.dot(a, b, allow_tf32=false)
        }
        mask_c = mask_m[:, None]
        tl.store(o_ptrs, o, mask=mask_c)
      }
    | Bool.false,Bool.true,Bool.true =>
      fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      triton {
        tl.ports inputs=[A,B], outputs=[O]
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
          inner_group_id = pid - (pid // num_CTA_per_group) * num_CTA_per_group
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
        mask_m = offs_m < $(M)
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          mask_a = mask_m[:, None]
          a = tl.load(a_ptrs, mask=mask_a, other=0.0)
          b = tl.load(b_ptrs)
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          o += tl.dot(a, b, allow_tf32=false)
        }
        mask_c = mask_m[:, None]
        tl.store(o_ptrs, o, mask=mask_c)
      }
    | Bool.true,Bool.false,Bool.false =>
      fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      triton {
        tl.ports inputs=[A,B], outputs=[O]
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
          inner_group_id = pid - (pid // num_CTA_per_group) * num_CTA_per_group
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
        mask_n = offs_n < $(N)
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          mask_k = offs_k < $(K)
          mask_a = mask_k[None, :]
          mask_b = mask_k[:, None] & mask_n[None, :]
          a = tl.load(a_ptrs, mask=mask_a, other=0.0)
          b = tl.load(b_ptrs, mask=mask_b, other=0.0)
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          o += tl.dot(a, b, allow_tf32=false)
        }
        mask_c = mask_n[None, :]
        tl.store(o_ptrs, o, mask=mask_c)
      }
    | Bool.true,Bool.false,Bool.true =>
      fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      triton {
        tl.ports inputs=[A,B], outputs=[O]
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
          inner_group_id = pid - (pid // num_CTA_per_group) * num_CTA_per_group
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
        mask_n = offs_n < $(N)
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          mask_b = mask_n[None, :]
          a = tl.load(a_ptrs)
          b = tl.load(b_ptrs, mask=mask_b, other=0.0)
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          o += tl.dot(a, b, allow_tf32=false)
        }
        mask_c = mask_n[None, :]
        tl.store(o_ptrs, o, mask=mask_c)
      }
    | Bool.true,Bool.true,Bool.false =>
      fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      triton {
        tl.ports inputs=[A,B], outputs=[O]
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
          inner_group_id = pid - (pid // num_CTA_per_group) * num_CTA_per_group
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
        
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          mask_k = offs_k < $(K)
          mask_a = mask_k[None, :]
          mask_b = mask_k[:, None]
          a = tl.load(a_ptrs, mask=mask_a, other=0.0)
          b = tl.load(b_ptrs, mask=mask_b, other=0.0)
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          o += tl.dot(a, b, allow_tf32=false)
        }
        
        tl.store(o_ptrs, o)
      }
    | Bool.true,Bool.true,Bool.true =>
      fun (A B O : RegionName) (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) =>
      triton {
        tl.ports inputs=[A,B], outputs=[O]
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
          inner_group_id = pid - (pid // num_CTA_per_group) * num_CTA_per_group
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
        
        a_ptrs = A + offs_m[:, None] * $(K) + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * $(N) + offs_n[None, :]
        o_ptrs = O + offs_m[:, None] * $(N) + offs_n[None, :]
        num_iters = tl.cdiv($(K), $(TILE_K))
        o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
        for _i in range($(0), num_iters, $(1)) {
          
          a = tl.load(a_ptrs)
          b = tl.load(b_ptrs)
          offs_k += $(TILE_K)
          a_ptrs += $(TILE_K)
          b_ptrs += $(TILE_K) * $(N)
          o += tl.dot(a, b, allow_tf32=false)
        }
        
        tl.store(o_ptrs, o)
      }
  build p.A p.B p.O p.M p.N p.K p.TM p.TN p.TK p.GROUP

def parameters (s : BmmOptimized.Shape) (c : BmmOptimized.Config) : Parameters :=
  ⟨"A","B","O",s.m,s.n,s.k,c.tm,c.tn,32,c.group,
    decide (s.m%c.tm=0),decide (s.n%c.tn=0),decide (s.k%32=0)⟩

end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.Source
