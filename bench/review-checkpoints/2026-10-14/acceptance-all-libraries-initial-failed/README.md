# Initial complete-library acceptance preflight — failed

This run requested all six current local libraries, including the full InProgress library rather than just previously selected public-entry closures. The terminal receipt is a failure, with unchanged captured inputs. The build found fourteen independent failing modules before the remaining long FastCeLoss ChunkedSafety elaboration was interrupted to repair those failures. That interruption and the resulting incomplete dependency closure are not mathematical rejections.

It neither completes all 184 tasks nor certifies any independent replay. The 67 unregistered legacy modules and six missing source pairs still require their own work.

````text
✖ [5127/6299] Building bench.tritonbench_g.embedding_triton_kernel.Kernels (17s)
✖ [5153/6299] Building bench.tritonbench_g.context_attn_fwd.Kernels (20s)
✖ [5155/6299] Building bench.tritonbench_g.chunk_bwd_dqkg.Proofs.LegacyReal (21s)
✖ [5177/6299] Building bench.tritonbench_g.context_attn_bloom.Kernels (21s)
✖ [5186/6299] Building bench.tritonbench_g.chunk_delta_fwd.Proofs.LegacyReal (21s)
✖ [5209/6299] Building bench.tritonbench_g.context_attn_llama.Kernels (22s)
✖ [5340/6299] Building bench.tritonbench_g.dropout_triton.Proofs.Erasure (11s)
✖ [5407/6299] Building bench.tritonbench_g.diag_ssm_triton.Proofs.Erasure (20s)
✖ [5607/6299] Building bench.tritonbench_g.chunked_cumsum_fwd.Proofs.Program (25s)
✖ [5884/6412] Building bench.tritonbench_g.chunk_gated_attention.Proofs.FP (39s)
✖ [5921/6412] Building bench.tritonbench_g.decay_cumsum.Proofs.RealReference (82s)
✖ [6095/6412] Building bench.tritonbench_g.chunk_gla_fwd.Proofs.FPIntra (30s)
✖ [6152/6412] Building bench.tritonbench_g.chunk_gated_attention.Proofs.RealHProgram (5.7s)
✖ [6216/6412] Building bench.tritonbench_g.chunk_gla_fwd.Proofs.FPSplit (50s)
✖ [6409/6412] Building bench.tritonbench_g.fast_ce_loss.Proofs.ChunkedSafety (253s)
````
