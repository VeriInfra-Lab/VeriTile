import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.Wrapper

/-! Canonical checked contract entry. Wrapper.validate covers header checks,
empty return, inspected CSR metadata and live physical-storage admission.
CheckedCSR.contract derives the historical safety contract from those checks. -/
