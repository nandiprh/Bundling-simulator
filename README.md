A simulator implementing static bundling against dynamic scheduling for a stripped down ISA based on 32-bit RISCV.
The ISA implemented here, strictly strips any floating point arithmetic to phase out complexity.
Then implementing DAG-based dependency analysis, height-ranked list scheduling, and Tomasulo algorithm with reorder
buffer and reservation stations.

The simulator is still in its early stages, though I plan it to integrate with kvm/qemu, but for now the approach is no heap allocation, fixed-size arrays throughout,
designed for portability.

Goal: is to benchmark performance, the criteria here is to benchmark across 4 workload classes (dot product, matrix multiply, sieve, Fibonacci) thus, demonstrating IPC
tradeoffs between scheduling strategies
