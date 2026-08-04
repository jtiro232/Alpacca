# Is Alpaccaroo2 in a good position to run at peak on POS, for BTBK?

Short answer: **no, and not for a tuning reason.** Three independent
blockers stack, only one of which is about speed. None of them is a defect
in Alpaccaroo2; all three are properties of the target environment, and two
are decided before a single token is generated.

Written 2026-08-04 against `Alpaccaroo2`, `/home/ubuntu/btbk-looper-2`, and
`/home/ubuntu/pos-iso` + `/home/ubuntu/pos-vm`. Evidence for every claim is
cited so a future instance can re-check it rather than trust it.

---

## What the three systems actually are

**Alpaccaroo2** runs one of four execution tiers, in descending speed:

| tier | needs | what runs the matvec |
|---|---|---|
| `gpu-cuda` | CUDA + pinned numba-cuda | our CUDA kernels |
| `numpy + kernels (native integer-dot)` | numpy **and** `numba==0.65.1` | `matvec_q4k_int` etc. |
| `numpy (no kernels)` | numpy only | einsum / BLAS GEMV |
| `pure-python` | **stdlib only** | list-of-lists matvec |

**BTBK** (`btbk-looper-2`) already treats Alpaccaroo as its engine of
record - this is not a proposal, it is shipped configuration:

- `llm_endpoint.py:73` - Alpaccaroo on `http://127.0.0.1:11435`; Ollama on
  `:11434` is the *alternative*.
- `test_llm_endpoint_a2.py:2` - "config must keep the engine of record
  (alpaccaroo - the system never falls [back]…)", with
  `test_shipped_config_selects_alpaccaroo` asserting it.
- The model is **`hermes3:8b-llama3.1-q4_k_m`** (`llm_endpoint.py:13`) - an
  **8B Q4_K_M**.
- `letter_system.py:146` budgets wall clock **per engine**:
  `{"ollama": 24.0, "alpaccaroo": 75.0}` seconds per letter. BTBK has
  already measured Alpaccaroo at roughly **3x Ollama's latency** and sized
  its timeouts for it, and `test_llm_prompt_prefix_order.py:289` asserts
  "alpaccaroo's budget leaves no room above its measured worst case" -
  i.e. the 75 s is *tight*, not generous.

**POS** (`pos-iso`) is an offline, signed-package OS built from
`ubuntu-26.04-live-server-amd64.iso`, with a 2923-entry `packages.lock`
and a Python login session (`pos-session_0.3.0_all.deb`).

---

## Blocker 1 - POS has no numpy, so Alpaccaroo runs its *slowest* tier

This is the decisive one, and it is a one-command check:

```sh
cd /home/ubuntu/pos-iso
grep -ci numpy packages.lock      # 0
grep -ci numba packages.lock      # 0
grep -ci llvmlite packages.lock   # 0
grep -ci openblas packages.lock   # 0
grep -ci lapack packages.lock     # 0
```

Every one returns **0**. The only Python packages POS ships are
`python3-minimal`, `python3-pip`, `python3-tk`, `python3-wheel`,
`python3-cryptography`, `python3-dbus`, `python3-gdbm`, `python3-netplan`.

So on POS as currently locked, `alpaccaroo doctor` would report
`backend: pure-python`, and **every optimisation of rounds 3 through 5 is
inactive** - the integer-dot kernels, the grouped Q4_K+Q6_K and Q4_K+Q5_K
kernels, the fused attention, the JIT rope, the BLAS paths. All of it is
gated on `numpy` being importable, and most of it additionally on
`numba==0.65.1`.

This is not a small regression. It is the difference between the tier the
8B numbers were measured on and a list-of-lists matvec in interpreted
Python.

### The measured cost of that package set

Measured on machine C (Ryzen 7 7730U), Qwen2.5-0.5B, same model and prompt
shape on each tier:

<!--TIERS-->

## Blocker 2 - the 8B model does not fit POS's memory envelope

`pos-vm/run-p1.sh` boots POS with `-smp 4 -m 4096` - **4 vCPUs and 4 GiB
of RAM**, on an 8 GiB qcow2, `-nic none`.

Against that, measured resident sizes for Alpaccaroo:

| model | quant | peak RSS | source |
|---|---|---:|---|
| Qwen2.5-0.5B | Q4_K_M | 1.2 GiB | machine C, this round |
| llama3.2:1b | Q8_0 | 2.9 GiB | machine C, this round |
| qwen2.5-3B | Q4_K_M | 4.1-4.3 GiB | machine C, this round |
| **Llama-3.1-8B** | **Q4_K_M** | **5.78 GiB** | `03-RESULTS.md` |

BTBK's `hermes3:8b-llama3.1-q4_k_m` is in that last row. **It does not fit
in 4 GiB**, and neither does a 3B. There is no swap in that VM
configuration. This is a hard capacity failure before performance is even
a question.

Two caveats stated honestly:

- `run-p1.sh` describes itself as a *"disposable VM cycle"* for exercising
  "the P1 broker matrix" - autoinstall, update media and recovery paths. It
  is POS's **CI fixture**, and may not be the production hardware target.
  If POS is meant for real hardware with 16-32 GiB, blocker 2 dissolves and
  blocker 1 does not.
- I found **no reference to POS anywhere in `btbk-looper-2`**
  (`grep -rli pos-session` over its docs returns nothing). So "BTBK runs on
  POS" is a prospective pairing being evaluated here, not an existing
  arrangement. That is worth saying plainly before anyone plans around it.

## Blocker 3 - 4 vCPUs, and the thread default is the wrong lever anyway

Alpaccaroo defaults its kernel pool to **physical cores**. Inside
`-smp 4 -cpu host` on an 8-core/16-thread host, the guest sees 4 CPUs and
cannot tell cores from SMT siblings, so `_platform.physical_cores()` is
guessing.

Round 5 measured how much that guess is worth, and the answer is
uncomfortable: on this machine `alpaccaroo tune -m` **recommends 16 threads
for the 3B and is wrong end-to-end by 42%** (`0/15` rounds won, median
1.4233). See `05-RESULTS.md` Package N. So the tuning surface that would
compensate for a VM's odd topology is itself unreliable, and
`ALPACCAROO_AUTOTUNE=1` should not be switched on inside POS until that is
fixed.

---

## What would actually put Alpaccaroo in a good position

In dependency order, cheapest first.

1. **Add `numpy` to POS's package set.** This alone moves Alpaccaroo from
   the pure tier to the numpy tier and is the single largest win available.
   `numpy` is packaged for Ubuntu 26.04 (`python3-numpy`), so it fits POS's
   `.deb`-and-lock model without vendoring wheels.
2. **Add the pinned kernel pair.** `numba==0.65.1` and `llvmlite==0.47.0`
   are **not** Ubuntu packages - they are PyPI wheels, and the pin is
   load-bearing (a different Numba deactivates the kernels and
   `alpaccaroo doctor` says so). POS's signed-lock model would need these
   vendored as hash-pinned wheels. This is the step that unlocks
   everything rounds 3-6 built.
3. **Size the VM for the model, or the model for the VM.** 8B Q4_K_M needs
   ~6 GiB resident plus headroom; 8 GiB is the floor and 12 GiB is
   comfortable. If 4 GiB is fixed, the largest workable model is roughly a
   1-3B class, and BTBK's prompts would have to be re-validated against it
   because a smaller model changes output quality, not just speed.
4. **Do not enable `ALPACCAROO_AUTOTUNE=1`** until Package N's finding is
   resolved.
5. **Leave `ALPACCAROO_SERIAL_MATVEC_ELEMS` off** - see `05-RESULTS.md`
   Package J, where the shipped default lost 21 of 25 end-to-end rounds.

## The honest summary

BTBK is *already* built around Alpaccaroo and has budgeted 75 s per letter
for it, so the integration question is settled - the Ollama-native API
means BTBK needs no changes at all. What is not settled is the platform:
**POS as currently locked cannot run Alpaccaroo above its slowest tier, and
its CI VM cannot hold BTBK's model at any tier.** Both are fixable by
changing POS's package set and memory envelope; neither is fixable by
tuning Alpaccaroo.
