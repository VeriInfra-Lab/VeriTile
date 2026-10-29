@torch.no_grad()
def token_softmax_fwd(Logics, B_Start_Loc, B_Seqlen, Prob_Out, max_input_len):
    BLOCK_SIZE = triton.next_power_of_2(max_input_len)
    batch, head_num = B_Start_Loc.shape[0], Logics.shape[0]

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    _fwd_kernel_token_softmax[(batch, head_num)](
        Logics, B_Start_Loc, B_Seqlen,
        Prob_Out,
        Logics.stride(0), Logics.stride(1),
        Prob_Out.stride(0), Prob_Out.stride(1),
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return
