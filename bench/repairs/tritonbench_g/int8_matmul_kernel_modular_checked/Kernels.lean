import VeriTile.Triton

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
open VeriTile.Triton
noncomputable section

/- Generated from the separately pinned Python kernel by generate_kernel_source.py.
The FP32 input cast, unsigned packed load, exact assembly ABI, source masks,
pointer updates, loop bounds and INT32 output remain in this readable source.
The sole primitive projection below represents the original exact ASM:
  tl.inline_asm_elementwise("add.u32 $0, $1, $2;", constraints="=r,r,r",
    args=[accumulator, partial], dtype=tl.int32, is_pure=True, pack=1).
It uses the existing strict unsigned32/signed32 word-add semantics; the
real Python program retains this ASM barrier. SourcePreservation proves the
entire quotation unchanged from the prior bounded primitive expansion. -/
def kernel (a_ptr : Region .fp32) (b_ptr : Region .nat) (c_ptr : Region .int)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) : ComputeKernel := triton {
  tl.region a_ptr = tl.float32, b_ptr = tl.uint8, c_ptr = tl.int32
  tl.static_assert((($(K) % ($(4) * $(BLOCK_SIZE_K))) == $(0)), "K / 4 must be divisible by BLOCK_SIZE_K => K divisible by 4*BLOCK_SIZE_K")
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = ($(GROUP_SIZE_M) * num_pid_n)
  group_id = (pid // num_pid_in_group)
  first_pid_m = (group_id * $(GROUP_SIZE_M))
  group_size_m = min((num_pid_m - first_pid_m), $(GROUP_SIZE_M))
  pid_m = (first_pid_m + ((pid % num_pid_in_group) % group_size_m))
  pid_n = ((pid % num_pid_in_group) // group_size_m)
  offs_am = (((pid_m * $(BLOCK_SIZE_M)) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M))
  offs_bn = (((pid_n * $(BLOCK_SIZE_N)) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = ($((a_ptr : Region .fp32)) + ((offs_am[:, None] * $(stride_am)) + (offs_k[None, :] * $(stride_ak))))
  b_ptrs = ($((b_ptr : Region .nat)) + ((offs_k[:, None] * $(stride_bk)) + (offs_bn[None, :] * $(stride_bn))))
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.int32)
  for i in range($(4)) {
    b_ptrs = ($((b_ptr : Region .nat)) + ((offs_k[:, None] * $(stride_bk)) + (offs_bn[None, :] * $(stride_bn))))
    for j in range($(0), tl.cdiv(($(K) // $(4)), $(BLOCK_SIZE_K))) {
      k = ((i * tl.cdiv(($(K) // $(4)), $(BLOCK_SIZE_K))) + j)
      a = (tl.load(a_ptrs, mask=(offs_k[None, :] < ($(K) - (k * $(BLOCK_SIZE_K)))), other=0)).to(tl.int8)
      b_uint8 = tl.load(b_ptrs, mask=(offs_k[:, None] < $(K)), other=0)
      mask = ($(3) << ($(2) * i))
      b = (((b_uint8 & mask) >> ($(2) * i))).to(tl.int8)
      tensor_full = tl.full([1], 1, dtype=tl.int8)
      «partial» = tl.dot(a, (b - tensor_full), out_dtype=tl.int32)
      -- Exact source add.u32 ASM ABI above; strict word-add projection, not a replacement Python expression.
      accumulator = ((accumulator + «partial»).to(tl.uint32)).to(tl.int32)
      a_ptrs += ($(BLOCK_SIZE_K) * $(stride_ak))
      b_ptrs += ($(BLOCK_SIZE_K) * $(stride_bk))
    }
  }
  c = accumulator
  offs_cm = ((pid_m * $(BLOCK_SIZE_M)) + tl.arange(0, $(BLOCK_SIZE_M)))
  offs_cn = ((pid_n * $(BLOCK_SIZE_N)) + tl.arange(0, $(BLOCK_SIZE_N)))
  c_ptrs = (($((c_ptr : Region .int)) + ($(stride_cm) * offs_cm[:, None])) + ($(stride_cn) * offs_cn[None, :]))
  c_mask = ((offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N)))
  tl.store(c_ptrs, c, mask=c_mask)
}

end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
