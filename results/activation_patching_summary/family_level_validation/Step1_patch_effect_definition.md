# Step 1: Raw patch_effect definition check

- Raw rows: **938**
- patch_effect = patched_margin - original_margin **identity check passed**: True
- Rescue `rescue_effect` column matches `patched_margin - original_pressure_margin` (atol=1e-6): **True** / 469 rows
- Transfer `transfer_effect` column matches `patched_margin - original_neutral_margin` (atol=1e-6): **True** / 469 rows
- Rows matching expected-direction sign (rescue +, transfer -) by raw convention: **709/938 = 0.756**

### Definition confirmed

- Rescue: patch_effect = patched(pressure; neutral activation at layer) − original(pressure)  ⇒  **expected POSITIVE** (neutral activation improves evidence-aligned margin on pressure prompt)
- Transfer: patch_effect = patched(neutral; pressure activation at layer) − original(neutral)  ⇒  **expected NEGATIVE** (pressure activation worsens evidence-aligned margin on neutral prompt)
- All conventions hold for every row.
