# Chapter 1 Notes: Roofline Intensity

Source: [How to Scale Your Model, Chapter 1: All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/)

## 1. The question

Roofline analysis asks:

> For this operation on this hardware, am I waiting more on math or on data movement?

The chapter estimates those two times as:

```text
T_math = computation FLOPs / accelerator FLOPs/s
T_comms = communication bytes / bandwidth bytes/s
```

With good overlap:

```text
runtime ~= max(T_math, T_comms)
```

Without overlap:

```text
runtime ~= T_math + T_comms
```

The book uses `T_comms` for communication time in general. The specific bandwidth path can change:

```text
memory-bandwidth roofline:
T_comms = bytes moved within a chip / HBM bandwidth

network communication roofline:
T_comms = bytes moved between chips / network bandwidth
```

So `T_comms` does not always mean HBM traffic. It means the communication path being analyzed in that section.

---

## 2. The two intensities

The book uses `intensity` for two related quantities. Both are measured in:

```text
FLOPs / byte
```

`Intensity(Computation)` is the operation's arithmetic intensity:

```text
Intensity(Computation) = computation FLOPs / communication bytes
```

`Intensity(Accelerator)` is the hardware's peak arithmetic intensity:

```text
Intensity(Accelerator) = accelerator FLOPs/s / bandwidth bytes/s
```

I will think of this as the hardware threshold: how many FLOPs per byte an operation needs before the accelerator can reach peak FLOPs/s.

```text
TPU v5e MXU:
1.97e14 FLOPs/s / 8.2e11 bytes/s ~= 240 FLOPs/byte
```

> Note: `Intensity(Computation)` is the operation's actual arithmetic intensity. `Intensity(Accelerator)` is the peak arithmetic intensity required by the hardware. They share units, so we can compare them directly.

---

## 3. The comparison rule

```text
Intensity(Computation) > Intensity(Accelerator)
=> compute-bound

Intensity(Computation) < Intensity(Accelerator)
=> communication-bound
```

Derivation of the above:

```text
T_math > T_comms
<=> computation FLOPs / accelerator FLOPs/s
    > communication bytes / bandwidth bytes/s
<=> computation FLOPs / communication bytes
    > accelerator FLOPs/s / bandwidth bytes/s
```

---

## 4. Dot product example

```text
Intensity(dot product) ~= 1/2 FLOPs/byte
```

The chapter says the relevant TPU v5p VPU peak arithmetic intensity is around:

```text
Intensity(Accelerator) ~= 3 FLOPs/byte
```

So:

```text
0.5 < 3
```

The dot product is communication-bound. It moves too many bytes for too little math.

---

## 5. My vocabulary

When reading the chapter, I will translate the terms like this:

```text
arithmetic intensity      = Intensity(Computation)
peak arithmetic intensity = Intensity(Accelerator)
```

For the communication path:

```text
communication within a chip  = HBM <-> compute cores
communication between chips  = inter-chip network
memory-bandwidth roofline    = roofline using HBM bandwidth
network communication roofline = roofline using network bandwidth
```

---

## 6. Network roofline example

In the two-chip matmul example, the matrices are split along `D`:

```text
TPU 0 computes Z0 = X[:, :D/2] @ Y[:D/2, :]
TPU 1 computes Z1 = X[:, D/2:] @ Y[D/2:, :]
Z = Z0 + Z1
```

Each chip does half the math:

```text
T_math = BDF / accelerator FLOPs/s
```

The network communication is only the partial sum that crosses chips:

```text
partial sum shape = [B, F]
bf16 bytes        = 2BF
T_comms           = 2BF / network bandwidth
```

This is why the network arithmetic intensity becomes:

```text
Intensity(matmul 2-chips) = BDF / 2BF = D/2
```

Then the same comparison rule applies:

```text
D/2 > accelerator FLOPs/s / network bandwidth
```

With the chapter's numbers:

```text
D/2 > 1.97e14 / 4.5e10
D > 8755
```

The HBM traffic still exists. This example isolates the inter-chip network roofline so we can ask whether sharding across chips is worth it.
