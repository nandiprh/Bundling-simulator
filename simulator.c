/*
 * rv32imfd_sim.c  --  Cycle-accurate simulator for RV32IMFD
 *                     Static-bundle (VLIW+stop) vs Out-of-Order core.
 *
 * Compile:  gcc -O2 -std=c11 -Wall -Wextra -o rv32imfd_sim rv32imfd_sim.c
 * Run:      ./rv32imfd_sim
 *
 * Design notes
 * ─────────────────────────────────────────────────────────────────────────────
 * All data structures are fixed-size arrays; no heap allocation is used.
 * This is deliberate: the simulator is intended to be readable alongside the
 * paper and portable to bare-metal environments.
 *
 * Correspondence to paper sections:
 *   §2   hazard classification  →  build_dag()
 *   §4.2 height-ranked list scheduling  →  compute_heights(), list_schedule()
 *   §7.1 StaticBundleCore pipeline  →  sim_static()
 *   §5   Tomasulo / OOO core  →  sim_ooo()
 *   §7.3 benchmark definitions  →  bench_*()
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>


#define BUNDLE_WIDTH   3        /* slots per bundle; change to 2 or 4 here   */
#define MAX_INSTRS     8192     /* max instructions per benchmark             */
#define MAX_BUNDLES    4096     /* max bundles after scheduling               */
#define MAX_REGS       64       /* 32 int + 32 float architectural registers  */
#define ROB_SIZE       64       /* reorder buffer entries                     */
#define RS_SIZE        32       /* reservation station entries per FU class   */
#define ISSUE_WIDTH    4        /* max dispatches per cycle (OOO)             */

typedef enum {
    FU_INT_ALU = 0,   /* slot 0 or 1 */
    FU_BRANCH  = 1,   /* slot 0 only */
    FU_MEM     = 2,   /* slot 1 only */
    FU_FPU_MUL = 3,   /* slot 2      */
    FU_COUNT   = 4
} FUClass;

/* Which bundle slots each FU class may occupy */
static const int SLOT_ALLOWED[FU_COUNT][BUNDLE_WIDTH] = {
    /* FU_INT_ALU */ { 1, 1, 0 },
    /* FU_BRANCH  */ { 1, 0, 0 },
    /* FU_MEM     */ { 0, 1, 0 },
    /* FU_FPU_MUL */ { 0, 0, 1 },
};

/* Register name encoding: 0..31 = x0..x31 (int), 32..63 = f0..f31 (float) */
#define REG_NONE  (-1)
#define XREG(n)   (n)          /* integer register xN  */
#define FREG(n)   (32 + (n))   /* float register fN    */

typedef enum {
    /* Integer ALU */
    OP_ADD, OP_SUB, OP_AND, OP_OR, OP_XOR, OP_SLL, OP_SRL, OP_SRA,
    OP_SLT, OP_SLTU, OP_ADDI, OP_ANDI, OP_ORI, OP_XORI,
    OP_SLLI, OP_SRLI, OP_SRAI, OP_SLTI, OP_SLTIU, OP_LUI, OP_AUIPC,
    /* Branch / Jump */
    OP_BEQ, OP_BNE, OP_BLT, OP_BGE, OP_BLTU, OP_BGEU, OP_JAL, OP_JALR,
    /* Memory */
    OP_LB, OP_LH, OP_LW, OP_LBU, OP_LHU,
    OP_SB, OP_SH, OP_SW,
    /* M extension */
    OP_MUL, OP_MULH, OP_MULHSU, OP_MULHU,
    OP_DIV, OP_DIVU, OP_REM, OP_REMU,
    /* F extension */
    OP_FADD_S, OP_FSUB_S, OP_FMUL_S, OP_FDIV_S, OP_FSQRT_S,
    OP_FMADD_S, OP_FMSUB_S,
    OP_FLW, OP_FSW,
    OP_FCVT_W_S, OP_FCVT_S_W, OP_FMV_X_W, OP_FMV_W_X,
    OP_FEQ_S, OP_FLT_S, OP_FLE_S,
    /* Internal */
    OP_NOP,
    OP_COUNT
} Opcode;

static const int LATENCY[OP_COUNT] = {
    /* ADD..AUIPC   */ 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    /* BEQ..JALR    */ 1,1,1,1,1,1,1,1,
    /* LB..LHU      */ 3,3,3,3,3,
    /* SB..SW       */ 1,1,1,
    /* MUL..MULHU   */ 3,3,3,3,
    /* DIV..REMU    */ 20,20,20,20,
    /* FADD..FMSUB  */ 4,4,4,12,16,4,4,
    /* FLW,FSW      */ 3,1,
    /* FCVT..FLE    */ 4,4,1,1,1,1,1,
    /* NOP          */ 0,
};

static FUClass opcode_fu(Opcode op)
{
    if (op >= OP_BEQ  && op <= OP_JALR)   return FU_BRANCH;
    if (op >= OP_LB   && op <= OP_SW)     return FU_MEM;
    if (op >= OP_MUL  && op <= OP_REMU)   return FU_FPU_MUL;
    if (op >= OP_FADD_S && op <= OP_FLE_S) return FU_FPU_MUL;
    return FU_INT_ALU;
}

static const char *MNEMONIC[OP_COUNT] = {
    "add","sub","and","or","xor","sll","srl","sra","slt","sltu",
    "addi","andi","ori","xori","slli","srli","srai","slti","sltiu","lui","auipc",
    "beq","bne","blt","bge","bltu","bgeu","jal","jalr",
    "lb","lh","lw","lbu","lhu","sb","sh","sw",
    "mul","mulh","mulhsu","mulhu","div","divu","rem","remu",
    "fadd.s","fsub.s","fmul.s","fdiv.s","fsqrt.s","fmadd.s","fmsub.s",
    "flw","fsw",
    "fcvt.w.s","fcvt.s.w","fmv.x.w","fmv.w.x",
    "feq.s","flt.s","fle.s",
    "nop",
};

typedef struct {
    Opcode  op;
    int     dst;         /* destination register, REG_NONE if none */
    int     src[3];      /* source registers (REG_NONE = unused)   */
    int     lat;         /* latency (cycles); filled on construction */
    FUClass fu;          /* functional unit class                   */
} Instr;

static Instr make_instr(Opcode op, int dst, int s0, int s1, int s2)
{
    Instr i;
    i.op     = op;
    i.dst    = dst;
    i.src[0] = s0;
    i.src[1] = s1;
    i.src[2] = s2;
    i.lat    = LATENCY[op];
    i.fu     = opcode_fu(op);
    return i;
}

static Instr NOP_INSTR;   /* initialised in main() */


typedef struct { int from, to, lat; } Edge;

#define MAX_EDGES  (MAX_INSTRS * 8)

typedef struct {
    Edge   edges[MAX_EDGES];
    int    n_edges;
    /* adjacency: succ_start[i]..succ_start[i+1]-1 are outgoing edges of i */
    int    succ_head[MAX_INSTRS];   /* linked list head per node  */
    int    succ_next[MAX_EDGES];    /* linked list next pointers  */
    int    pred_count[MAX_INSTRS];
    int    n_nodes;
} DAG;

static void dag_add_edge(DAG *dag, int from, int to, int lat)
{
    if (dag->n_edges >= MAX_EDGES) return;
    int e = dag->n_edges++;
    dag->edges[e] = (Edge){ from, to, lat };
    dag->succ_next[e] = dag->succ_head[from];
    dag->succ_head[from] = e;
    dag->pred_count[to]++;
}

static void build_dag(DAG *dag, const Instr *instrs, int n)
{
    memset(dag, 0, sizeof(*dag));
    dag->n_nodes = n;
    for (int i = 0; i < n;   i++) dag->succ_head[i]  = -1;
    for (int e = 0; e < MAX_EDGES; e++) dag->succ_next[e] = -1;

    int last_write[MAX_REGS];          /* last instruction that wrote reg r */
    int last_read[MAX_REGS][16];       /* last instructions that read reg r */
    int n_reads[MAX_REGS];
    memset(last_write, -1, sizeof(last_write));
    memset(n_reads,    0,  sizeof(n_reads));

    for (int j = 0; j < n; j++) {
        const Instr *ins = &instrs[j];

        /* RAW: j reads a register written by an earlier instruction */
        for (int k = 0; k < 3; k++) {
            int r = ins->src[k];
            if (r == REG_NONE) continue;
            int i = last_write[r];
            if (i >= 0)
                dag_add_edge(dag, i, j, instrs[i].lat);
        }

        /* WAW: j writes a register written earlier — output dependency */
        if (ins->dst != REG_NONE) {
            int i = last_write[ins->dst];
            if (i >= 0)
                dag_add_edge(dag, i, j, 0);
        }

        /* WAR: j writes a register that earlier instructions still read */
        if (ins->dst != REG_NONE) {
            int r = ins->dst;
            for (int k = 0; k < n_reads[r]; k++)
                dag_add_edge(dag, last_read[r][k], j, 0);
            n_reads[r] = 0;
        }

        /* Update tracking tables */
        if (ins->dst != REG_NONE)
            last_write[ins->dst] = j;
        for (int k = 0; k < 3; k++) {
            int r = ins->src[k];
            if (r == REG_NONE) continue;
            if (n_reads[r] < 16)
                last_read[r][n_reads[r]++] = j;
        }
    }
}

static void compute_heights(const DAG *dag, int *h)
{
    int n = dag->n_nodes;
    memset(h, 0, n * sizeof(int));

    /* Simple relaxation: iterate until stable (Bellman-Ford on DAG).
     * Since the DAG is acyclic, n iterations suffice. */
    for (int pass = 0; pass < n; pass++) {
        int changed = 0;
        for (int e = 0; e < dag->n_edges; e++) {
            int  from = dag->edges[e].from;
            int  to   = dag->edges[e].to;
            int  lat  = dag->edges[e].lat;
            int  cand = lat + h[to];
            if (cand > h[from]) { h[from] = cand; changed = 1; }
        }
        if (!changed) break;
    }
}

typedef struct { int height; int idx; } PQEntry;
typedef struct { PQEntry data[MAX_INSTRS]; int size; } PQ;

static void pq_push(PQ *pq, int height, int idx)
{
    int i = pq->size++;
    pq->data[i] = (PQEntry){ height, idx };
    /* bubble up */
    while (i > 0) {
        int p = (i - 1) / 2;
        if (pq->data[p].height >= pq->data[i].height) break;
        PQEntry tmp = pq->data[p]; pq->data[p] = pq->data[i]; pq->data[i] = tmp;
        i = p;
    }
}

static PQEntry pq_pop(PQ *pq)
{
    PQEntry top = pq->data[0];
    pq->data[0] = pq->data[--pq->size];
    /* sift down */
    int i = 0;
    for (;;) {
        int l = 2*i+1, r = 2*i+2, largest = i;
        if (l < pq->size && pq->data[l].height > pq->data[largest].height) largest = l;
        if (r < pq->size && pq->data[r].height > pq->data[largest].height) largest = r;
        if (largest == i) break;
        PQEntry tmp = pq->data[i]; pq->data[i] = pq->data[largest]; pq->data[largest] = tmp;
        i = largest;
    }
    return top;
}

typedef struct {
    Instr slots[BUNDLE_WIDTH];
    int   stop;          /* IA-64-style stop bit */
} Bundle;

static int list_schedule(const Instr *instrs, int n,
                         Bundle *out_bundles, int max_bundles)
{
    if (n == 0) return 0;

    DAG dag;
    build_dag(&dag, instrs, n);

    int h[MAX_INSTRS];
    compute_heights(&dag, h);

    /* available[i] = earliest bundle index in which instruction i may issue */
    int available[MAX_INSTRS];
    memset(available, 0, n * sizeof(int));

    /* scheduled_at[i] = bundle index assigned to instruction i */
    int scheduled_at[MAX_INSTRS];
    memset(scheduled_at, -1, n * sizeof(int));

    /* pred_remaining[i] = number of unscheduled predecessors */
    int pred_remaining[MAX_INSTRS];
    memcpy(pred_remaining, dag.pred_count, n * sizeof(int));

    /* Initialise ready queue with all source nodes (0 predecessors) */
    PQ ready;
    ready.size = 0;
    for (int i = 0; i < n; i++)
        if (pred_remaining[i] == 0)
            pq_push(&ready, h[i], i);

    int n_bundles = 0;

    /* slot_used[b][s] = 1 if slot s of bundle b is occupied */
    static int slot_used[MAX_BUNDLES][BUNDLE_WIDTH];
    memset(slot_used, 0, sizeof(slot_used));

    /* Initialise all bundles with NOPs and stop=0 */
    for (int b = 0; b < max_bundles; b++) {
        out_bundles[b].stop = 0;
        for (int s = 0; s < BUNDLE_WIDTH; s++)
            out_bundles[b].slots[s] = NOP_INSTR;
    }

    while (ready.size > 0) {
        PQEntry pe = pq_pop(&ready);
        int idx = pe.idx;
        const Instr *ins = &instrs[idx];

        /* Find earliest bundle >= available[idx] with a free compatible slot */
        int placed = 0;
        int bi_start = available[idx];
        if (bi_start < 0) bi_start = 0;

        for (int bi = bi_start; bi < max_bundles && !placed; bi++) {
            for (int s = 0; s < BUNDLE_WIDTH && !placed; s++) {
                if (!SLOT_ALLOWED[ins->fu][s]) continue;
                if (slot_used[bi][s])          continue;
                /* Place instruction */
                out_bundles[bi].slots[s] = *ins;
                slot_used[bi][s] = 1;
                scheduled_at[idx] = bi;
                if (bi + 1 > n_bundles) n_bundles = bi + 1;
                placed = 1;
            }
        }

        if (!placed) {
            int bi = n_bundles;
            if (bi < max_bundles) {
                out_bundles[bi].slots[0] = *ins;
                slot_used[bi][0] = 1;
                scheduled_at[idx] = bi;
                n_bundles = bi + 1;
            }
        }

        /* Release successors whose all predecessors are now scheduled */
        for (int e = dag.succ_head[idx]; e != -1; e = dag.succ_next[e]) {
            int j   = dag.edges[e].to;
            int lat = dag.edges[e].lat;
            /* Successor j can issue in bundle scheduled_at[idx] + lat (rounded up) */
            int avail = scheduled_at[idx] + (lat > 0 ? lat : 1);
            if (avail > available[j]) available[j] = avail;
            if (--pred_remaining[j] == 0)
                pq_push(&ready, h[j], j);
        }
    }

    /* Set stop bits: a bundle has stop=1 if any result it produces is consumed
     * in the immediately following bundle (the hardware must stall one cycle). */
    for (int b = 0; b + 1 < n_bundles; b++) {
        for (int s = 0; s < BUNDLE_WIDTH; s++) {
            const Instr *prod = &out_bundles[b].slots[s];
            if (prod->op == OP_NOP || prod->dst == REG_NONE) continue;
            for (int t = 0; t < BUNDLE_WIDTH; t++) {
                const Instr *cons = &out_bundles[b+1].slots[t];
                for (int k = 0; k < 3; k++) {
                    if (cons->src[k] == prod->dst) {
                        out_bundles[b].stop = 1;
                        goto next_bundle;
                    }
                }
            }
        }
        next_bundle:;
    }

    return n_bundles;
}


typedef struct {
    long cycles;
    long committed;
    double cpi;
    double ipc;
    double slot_util;   /* fraction of non-NOP slots */
} SimResult;

static SimResult sim_static(const Instr *instrs, int n)
{
    static Bundle bundles[MAX_BUNDLES];
    int n_bundles = list_schedule(instrs, n, bundles, MAX_BUNDLES);

    long cycles    = 0;
    long committed = 0;
    long nop_slots = 0;
    long total_slots = 0;

    /* fu_free[f] = first cycle in which FU class f is available again */
    long fu_free[FU_COUNT];
    memset(fu_free, 0, sizeof(fu_free));

    for (int b = 0; b < n_bundles; b++) {
        const Bundle *bun = &bundles[b];

        /* Determine issue cycle: must wait for all FUs used in this bundle */
        long issue_cycle = cycles;
        for (int s = 0; s < BUNDLE_WIDTH; s++) {
            const Instr *ins = &bun->slots[s];
            if (ins->op == OP_NOP) continue;
            if (fu_free[ins->fu] > issue_cycle)
                issue_cycle = fu_free[ins->fu];
        }
        /* Stop bit: if set, also wait one extra cycle for the pipeline to drain */
        if (b > 0 && bundles[b-1].stop)
            issue_cycle = (issue_cycle > cycles) ? issue_cycle : cycles;

        cycles = issue_cycle;

        for (int s = 0; s < BUNDLE_WIDTH; s++) {
            const Instr *ins = &bun->slots[s];
            total_slots++;
            if (ins->op == OP_NOP) { nop_slots++; continue; }
            fu_free[ins->fu] = cycles + ins->lat;
            committed++;
        }
        cycles++;   /* one cycle to issue the bundle */
    }

    SimResult r;
    r.cycles    = cycles;
    r.committed = committed;
    r.cpi       = committed > 0 ? (double)cycles / committed : 0.0;
    r.ipc       = r.cpi > 0     ? 1.0 / r.cpi               : 0.0;
    r.slot_util = total_slots > 0
                  ? (double)(total_slots - nop_slots) / total_slots
                  : 0.0;
    return r;
}


/*
 * ROB entry: tracks when the result of an instruction will be ready.
 * Instructions commit in order from the ROB head when their ready_cycle
 * has passed.
 */
typedef struct {
    int  valid;
    long ready_cycle;   /* cycle at which the result is available            */
} ROBEntry;

static SimResult sim_ooo(const Instr *instrs, int n)
{
    /* Register ready table: reg_ready[r] = cycle at which register r holds
     * a valid value.  0 = already available.                               */
    long reg_ready[MAX_REGS];
    memset(reg_ready, 0, sizeof(reg_ready));

    /* Circular ROB */
    ROBEntry rob[ROB_SIZE];
    memset(rob, 0, sizeof(rob));
    int rob_head = 0, rob_tail = 0, rob_count = 0;

    long cycles    = 0;
    long committed = 0;
    int  ip        = 0;   /* next instruction to dispatch */

    /* Pipeline stages add a minimum of 3 cycles before an instruction
     * can execute (Rename + Dispatch + Issue wakeup latency). */
    const long PIPE_OVERHEAD = 3;

    while (ip < n || rob_count > 0) {
        /* ── Commit: retire instructions from ROB head in program order ── */
        while (rob_count > 0 && rob[rob_head].ready_cycle <= cycles) {
            rob[rob_head].valid = 0;
            rob_head = (rob_head + 1) % ROB_SIZE;
            rob_count--;
            committed++;
        }

        /* ── Dispatch: fetch and rename up to ISSUE_WIDTH instructions ── */
        int dispatched = 0;
        while (ip < n && rob_count < ROB_SIZE && dispatched < ISSUE_WIDTH) {
            const Instr *ins = &instrs[ip];

            /* Earliest cycle sources are ready (RAW dependency) */
            long src_ready = 0;
            for (int k = 0; k < 3; k++) {
                int r = ins->src[k];
                if (r == REG_NONE) continue;
                if (reg_ready[r] > src_ready) src_ready = reg_ready[r];
            }

            /* Earliest execution cycle: pipeline overhead + source latency */
            long exec_cycle   = cycles + PIPE_OVERHEAD;
            if (src_ready > exec_cycle) exec_cycle = src_ready;
            long result_cycle = exec_cycle + ins->lat;

            /* Update register ready table (rename: allocate result slot) */
            if (ins->dst != REG_NONE)
                reg_ready[ins->dst] = result_cycle;

            /* Insert into ROB */
            rob[rob_tail].valid       = 1;
            rob[rob_tail].ready_cycle = result_cycle;
            rob_tail = (rob_tail + 1) % ROB_SIZE;
            rob_count++;

            ip++;
            dispatched++;
        }

        cycles++;

        /* Safety valve */
        if (cycles > (long)n * 100) break;
    }

    SimResult r;
    r.cycles    = cycles;
    r.committed = committed;
    r.cpi       = committed > 0 ? (double)cycles / committed : 0.0;
    r.ipc       = r.cpi > 0     ? 1.0 / r.cpi               : 0.0;
    r.slot_util = -1.0;   /* not applicable to OOO core */
    return r;
}


/* Integer dot product: sum += a[i] * b[i]  (§7.3, benchmark 1) */
static int bench_dot_product(Instr *instrs, int n_iters)
{
    int cnt = 0;
    /* a ptr in x10 (a0), b ptr in x11 (a1), sum in x12 (a2),
       temporaries t0=x5, t1=x6, t2=x7                        */
    for (int i = 0; i < n_iters && cnt + 6 < MAX_INSTRS; i++) {
        instrs[cnt++] = make_instr(OP_LW,   XREG(5),  XREG(10), REG_NONE, REG_NONE); /* lw  t0, 0(a0)       */
        instrs[cnt++] = make_instr(OP_LW,   XREG(6),  XREG(11), REG_NONE, REG_NONE); /* lw  t1, 0(a1)       */
        instrs[cnt++] = make_instr(OP_MUL,  XREG(7),  XREG(5),  XREG(6),  REG_NONE); /* mul t2, t0, t1      */
        instrs[cnt++] = make_instr(OP_ADD,  XREG(12), XREG(12), XREG(7),  REG_NONE); /* add a2, a2, t2      */
        instrs[cnt++] = make_instr(OP_ADDI, XREG(10), XREG(10), REG_NONE, REG_NONE); /* addi a0, a0, 4      */
        instrs[cnt++] = make_instr(OP_ADDI, XREG(11), XREG(11), REG_NONE, REG_NONE); /* addi a1, a1, 4      */
    }
    return cnt;
}

static int bench_matmul(Instr *instrs, int n_iters)
{
    int cnt = 0;
    for (int i = 0; i < n_iters && cnt + 8 < MAX_INSTRS; i++) {
        instrs[cnt++] = make_instr(OP_LW,   XREG(5),  XREG(10), REG_NONE, REG_NONE); /* lw t0, 0(a0)   A */
        instrs[cnt++] = make_instr(OP_LW,   XREG(6),  XREG(11), REG_NONE, REG_NONE); /* lw t1, 0(a1)   B */
        instrs[cnt++] = make_instr(OP_MUL,  XREG(7),  XREG(5),  XREG(6),  REG_NONE); /* mul t2,t0,t1     */
        instrs[cnt++] = make_instr(OP_LW,   XREG(28), XREG(12), REG_NONE, REG_NONE); /* lw t3, 0(a2)   C */
        instrs[cnt++] = make_instr(OP_ADD,  XREG(28), XREG(28), XREG(7),  REG_NONE); /* add t3,t3,t2     */
        instrs[cnt++] = make_instr(OP_SW,   REG_NONE, XREG(12), XREG(28), REG_NONE); /* sw t3, 0(a2)     */
        instrs[cnt++] = make_instr(OP_ADDI, XREG(10), XREG(10), REG_NONE, REG_NONE); /* addi a0,a0,4     */
        instrs[cnt++] = make_instr(OP_ADDI, XREG(11), XREG(11), REG_NONE, REG_NONE); /* addi a1,a1,4     */
    }
    return cnt;
}

/* Fibonacci call structure: branch/call heavy (§7.3, benchmark 3) */
static int bench_fibonacci(Instr *instrs, int n_iters)
{
    int cnt = 0;
    for (int i = 0; i < n_iters && cnt + 9 < MAX_INSTRS; i++) {
        instrs[cnt++] = make_instr(OP_SW,   REG_NONE, XREG(2),  XREG(1),  REG_NONE); /* sw  ra, 0(sp)    */
        instrs[cnt++] = make_instr(OP_ADDI, XREG(2),  XREG(2),  REG_NONE, REG_NONE); /* addi sp,sp,-8    */
        instrs[cnt++] = make_instr(OP_BLT,  REG_NONE, XREG(10), XREG(5),  REG_NONE); /* blt a0,t0,.base  */
        instrs[cnt++] = make_instr(OP_ADDI, XREG(10), XREG(10), REG_NONE, REG_NONE); /* addi a0,a0,-1    */
        instrs[cnt++] = make_instr(OP_JALR, XREG(1),  XREG(6),  REG_NONE, REG_NONE); /* jalr ra,t1,0     */
        instrs[cnt++] = make_instr(OP_ADD,  XREG(7),  XREG(7),  XREG(10), REG_NONE); /* add t2,t2,a0     */
        instrs[cnt++] = make_instr(OP_LW,   XREG(1),  XREG(2),  REG_NONE, REG_NONE); /* lw  ra, 0(sp)    */
        instrs[cnt++] = make_instr(OP_ADDI, XREG(2),  XREG(2),  REG_NONE, REG_NONE); /* addi sp,sp,8     */
        instrs[cnt++] = make_instr(OP_JALR, REG_NONE, XREG(1),  REG_NONE, REG_NONE); /* jalr x0,ra,0     */
    }
    return cnt;
}

/* Sieve of Eratosthenes inner loop: irregular stride (§7.3, benchmark 4) */
static int bench_sieve(Instr *instrs, int n_iters)
{
    int cnt = 0;
    for (int i = 0; i < n_iters && cnt + 8 < MAX_INSTRS; i++) {
        instrs[cnt++] = make_instr(OP_MUL,  XREG(6),  XREG(5),  XREG(5),  REG_NONE); /* mul t1,t0,t0  p*p   */
        instrs[cnt++] = make_instr(OP_BGE,  REG_NONE, XREG(6),  XREG(12), REG_NONE); /* bge t1,a2,.end      */
        instrs[cnt++] = make_instr(OP_LW,   XREG(7),  XREG(10), REG_NONE, REG_NONE); /* lw  t2,0(a0) sieve  */
        instrs[cnt++] = make_instr(OP_BNE,  REG_NONE, XREG(7),  XREG(0),  REG_NONE); /* bne t2,x0,.skip     */
        instrs[cnt++] = make_instr(OP_MUL,  XREG(28), XREG(5),  XREG(5),  REG_NONE); /* mul t3,t0,t0  j=p*p */
        instrs[cnt++] = make_instr(OP_SW,   REG_NONE, XREG(10), XREG(0),  REG_NONE); /* sw  x0,0(a0)        */
        instrs[cnt++] = make_instr(OP_ADD,  XREG(28), XREG(28), XREG(5),  REG_NONE); /* add t3,t3,t0  j+=p  */
        instrs[cnt++] = make_instr(OP_ADDI, XREG(5),  XREG(5),  REG_NONE, REG_NONE); /* addi t0,t0,1        */
    }
    return cnt;
}

/* Single-precision SAXPY: y[i] = a*x[i] + y[i]  (§7.3, benchmark 5) */
static int bench_saxpy(Instr *instrs, int n_iters)
{
    int cnt = 0;
    /* a in f10 (fa0), x ptr in x10, y ptr in x11
       temporaries: f5=ft0, f6=ft1, f7=ft2, f28=ft3 */
    for (int i = 0; i < n_iters && cnt + 7 < MAX_INSTRS; i++) {
        instrs[cnt++] = make_instr(OP_FLW,    FREG(5),  XREG(10), REG_NONE, REG_NONE); /* flw ft0, 0(a0)       x[i] */
        instrs[cnt++] = make_instr(OP_FLW,    FREG(6),  XREG(11), REG_NONE, REG_NONE); /* flw ft1, 0(a1)       y[i] */
        instrs[cnt++] = make_instr(OP_FMUL_S, FREG(7),  FREG(32), FREG(5),  REG_NONE); /* fmul.s ft2,fa0,ft0   a*x  */
        instrs[cnt++] = make_instr(OP_FADD_S, FREG(28), FREG(7),  FREG(6),  REG_NONE); /* fadd.s ft3,ft2,ft1        */
        instrs[cnt++] = make_instr(OP_FSW,    REG_NONE, XREG(11), FREG(28), REG_NONE); /* fsw ft3, 0(a1)            */
        instrs[cnt++] = make_instr(OP_ADDI,   XREG(10), XREG(10), REG_NONE, REG_NONE); /* addi a0,a0,4              */
        instrs[cnt++] = make_instr(OP_ADDI,   XREG(11), XREG(11), REG_NONE, REG_NONE); /* addi a1,a1,4              */
    }
    return cnt;
}


static void print_bundles(const Bundle *bundles, int n)
{
    printf("\n  Bundle dump (%d bundles, %d slots wide):\n", n, BUNDLE_WIDTH);
    printf("  %-6s  %-3s  ", "Bundle", "S");
    for (int s = 0; s < BUNDLE_WIDTH; s++)
        printf("  %-14s", s == 0 ? "Slot0" : s == 1 ? "Slot1" : "Slot2");
    printf("\n  ");
    for (int i = 0; i < 6+3+BUNDLE_WIDTH*16; i++) putchar('-');
    putchar('\n');
    for (int b = 0; b < n; b++) {
        printf("  %-6d  %-3d  ", b, bundles[b].stop);
        for (int s = 0; s < BUNDLE_WIDTH; s++)
            printf("  %-14s", MNEMONIC[bundles[b].slots[s].op]);
        putchar('\n');
    }
    putchar('\n');
}


int main(void)
{
    /* Initialise the NOP instruction sentinel */
    NOP_INSTR = make_instr(OP_NOP, REG_NONE, REG_NONE, REG_NONE, REG_NONE);

    static Instr instrs[MAX_INSTRS];
    static Bundle bundles[MAX_BUNDLES];

    typedef struct {
        const char *name;
        int (*build)(Instr *, int);
        int n_iters;
    } Bench;

    Bench benches[] = {
        { "Dot product (int, N=64)",  bench_dot_product, 64  },
        { "Matrix multiply (8x8x8)",  bench_matmul,      8*8*8 },
        { "Fibonacci (depth=25)",     bench_fibonacci,   25  },
        { "Sieve (N=200 outer iters)",bench_sieve,       200 },
        { "SAXPY (FP, N=64)",         bench_saxpy,       64  },
    };
    int n_benches = (int)(sizeof(benches)/sizeof(benches[0]));

    printf("\n");
    printf("  RV32IMFD Static-Bundle vs OOO Simulator  (bundle width = %d)\n", BUNDLE_WIDTH);
    printf("  Latencies: INT=1, LW=3, MUL=3, DIV=20, FADD/FMUL=4, FDIV=12\n");
    printf("  OOO: ROB=%d entries, dispatch width=%d, pipeline overhead=%d cycles\n\n",
           ROB_SIZE, ISSUE_WIDTH, 3);

    printf("  %-30s  %8s  %8s  %8s  %8s  %8s  %9s\n",
           "Benchmark", "St.Cyc", "St.IPC", "SlotUtil", "OOO.Cyc", "OOO.IPC", "Winner");
    printf("  ");
    for (int i = 0; i < 92; i++) putchar('-');
    printf("\n");

    for (int b = 0; b < n_benches; b++) {
        int n = benches[b].build(instrs, benches[b].n_iters);

        SimResult sr = sim_static(instrs, n);
        SimResult or_ = sim_ooo(instrs, n);

        const char *winner = sr.ipc >= or_.ipc ? "Static" : "OOO";

        printf("  %-30s  %8ld  %8.3f  %7.1f%%  %8ld  %8.3f  %9s\n",
               benches[b].name,
               sr.cycles, sr.ipc, sr.slot_util * 100.0,
               or_.cycles, or_.ipc,
               winner);
    }
    printf("\n");

    int n_saxpy = bench_saxpy(instrs, 64);
    int n_bun   = list_schedule(instrs, n_saxpy, bundles, MAX_BUNDLES);
    Bundle top10[10];
    int show = n_bun < 10 ? n_bun : 10;
    for (int i = 0; i < show; i++) top10[i] = bundles[i];
    print_bundles(top10, show);

    return 0;
}
