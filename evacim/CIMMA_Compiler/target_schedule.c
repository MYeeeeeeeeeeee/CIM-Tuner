#include "target_schedule.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

/* 固定 F7 的 metadata 上限；不配置 tensor／scratch payload。 */
enum { LIMIT = 64, FILL, PROGRAM, READ, COMPUTE, EVICT };
typedef struct {
    int id, kind, m, p, a, batch, set, version, k0, k1, n0, n1;
    int direct, input, program, previous, overwrite, atos, owner, final;
    int dependencies[LIMIT], dependencies_count;
    const char *function;
    int line;
} Op;
static Op ops[LIMIT];
static int count, failed, batch, current[2], fills[5], partial[10], last_read;
static int blocks[4][2], sizes[2], a_count, p_count, batch_count, tiling_line;
static const char *tiling_function;
static const TargetConfig *configuration;
static int batch_lines[6];
static unsigned coverage[10];

static Op *append(int kind, const char *function, int line)
{
    if (count >= LIMIT) { failed = 1; return NULL; }
    Op *o = &ops[count++];
    memset(o, 0, sizeof(*o));
    o->id = count; o->kind = kind; o->m = o->p = o->a = o->set = -1;
    o->batch = batch; o->function = function; o->line = line;
    return o;
}

static void dependency(Op *o, int id)
{
    if (!id) return;
    for (int i = 0; i < o->dependencies_count; ++i)
        if (o->dependencies[i] == id) return;
    if (o->dependencies_count == LIMIT) { failed = 1; return; }
    o->dependencies[o->dependencies_count++] = id;
}

static void region(Op *o, int p, int a)
{
    o->p = p; o->a = a;
    o->k0 = a * 8; o->k1 = o->k0 + 8 < 9 ? o->k0 + 8 : 9;
    o->n0 = p * 4; o->n1 = o->n0 + 4 < 7 ? o->n0 + 4 : 7;
    if (p < 0 || p >= 2 || a < 0 || a >= 2) failed = 1;
}

void schedule_begin(const TargetConfig *target)
{
    memset(ops, 0, sizeof(ops)); memset(current, 0, sizeof(current));
    memset(fills, 0, sizeof(fills)); memset(partial, 0, sizeof(partial));
    memset(batch_lines, 0, sizeof(batch_lines));
    memset(coverage, 0, sizeof(coverage));
    count = failed = last_read = 0; batch = -1; configuration = target;
}

void schedule_tiling(int a, int p, int batches, const int *batch_sizes,
                     int **sets, int **atos, const char *function, int line)
{
    a_count = a; p_count = p; batch_count = batches;
    tiling_function = function; tiling_line = line;
    if (a != 2 || p != 2 || batches != 2) { failed = 1; return; }
    for (int i = 0; i < batches; ++i) sizes[i] = batch_sizes[i];
    for (int ip = 0; ip < p; ++ip)
        for (int ia = 0; ia < a; ++ia) {
            blocks[ip * a + ia][0] = sets[ip][ia];
            blocks[ip * a + ia][1] = atos[ip][ia];
        }
}

void schedule_cache(int vectors, const char *function, int line)
{
    if (vectors != 2) { failed = 1; return; }
    for (int m = 0; m < vectors; ++m) {
        Op *o = append(FILL, function, line); if (!o) return;
        o->m = m; o->k1 = 9; fills[m] = o->id;
    }
}

void schedule_ip_cache(int first, int vectors, int evict, const char *function, int line)
{
    if(first<0 || first+vectors>5 || vectors<1 || vectors>2) {failed=1;return;}
    for(int m=first;m<first+vectors;++m) {
        Op *o=append(evict?EVICT:FILL,function,line);if(!o)return;
        o->batch=-1;o->m=m;o->k1=9;
        if(evict) {
            o->input=fills[m];dependency(o,o->input);
            for(int i=0;i<count-1;++i)if(ops[i].kind==READ && ops[i].input==o->input)dependency(o,ops[i].id);
            fills[m]=0;
        } else fills[m]=o->id;
    }
}

void schedule_batch(int next, const char *function, int line)
{
    (void)function;
    if (next < 0 || next >= (configuration->ip_af_export?6:2)) { failed = 1; return; }
    batch = next; batch_lines[next] = line;
}

void schedule_program(int p, int a, int set, const char *function, int line)
{
    if (set < 0 || set >= 2) { failed = 1; return; }
    int old = current[set];
    Op *o = append(PROGRAM, function, line); if (!o) return;
    region(o, p, a); o->set = set; o->overwrite = old;
    o->version = old ? ops[old - 1].version + 1 : 1;
    /* 相依從已記錄 uses 得出，不預先產生預期排程。 */
    for (int i = 0; i < count - 1; ++i)
        if (ops[i].kind == COMPUTE && ops[i].program == old && old)
            dependency(o, ops[i].id);
    current[set] = o->id;
}

void schedule_read(int m, int p, int a, int direct, const char *function, int line)
{
    if (m < 0 || m >= 5) { failed = 1; return; }
    Op *o = append(READ, function, line); if (!o) return;
    region(o, p, a); o->m = m; o->direct = direct;
    o->input = direct ? 0 : fills[m]; dependency(o, o->input);
    last_read = o->id;
}

void schedule_compute(int m, int p, int a, int set, int atos, int owner,
                      const char *function, int line)
{
    if (set < 0 || set >= 2 || owner < 0 || owner >= 10) { failed = 1; return; }
    Op *o = append(COMPUTE, function, line); if (!o) return;
    region(o, p, a); o->m = m; o->set = set; o->atos = atos; o->owner = owner;
    o->input = last_read; o->program = current[set];
    if (!o->program) { failed = 1; return; }
    o->version = ops[o->program - 1].version; o->previous = partial[owner];
    /* ATOS 保留來源編碼；完成取決於同 owner 的有效 K 聯集。 */
    const unsigned contribution = ((1u << (o->k1 - o->k0)) - 1u) << o->k0;
    if (coverage[owner] & contribution) failed = 1;
    coverage[owner] |= contribution;
    o->final = coverage[owner] == 511u;
    dependency(o, o->input); dependency(o, o->program); dependency(o, o->previous);
    partial[owner] = o->id;
}

static void source(const char *function, int line, const char *decision)
{
    printf("{\"file\":\"compiler_count.c\",\"function\":\"%s\",\"line\":%d,"
           "\"decision\":\"%s\"}", function, line, decision);
}

static void uses(int producer)
{
    int comma = 0;
    putchar('[');
    for (int i = 0; i < count; ++i)
        if ((ops[i].kind == COMPUTE && ops[i].program == producer) ||
            (ops[i].kind == READ && ops[i].input == producer)) {
            printf("%s%d", comma ? "," : "", ops[i].id); comma = 1;
        }
    putchar(']');
}

int schedule_write(void)
{
    if (failed || !tiling_function || !batch_lines[0] || !batch_lines[1]) {
        fprintf(stderr, "target-f7-export: 決策記錄不完整或超出 metadata 上限\n");
        return 2;
    }
    const TargetConfig *t = configuration;
    printf("{\"format\":\"cim-tuner-logical-schedule\",\"version\":%d,"
           "\"request\":\"request-0\",\"provenance_manifest\":\"provenance.json\",\"config\":{", t->ip_af_export?3:t->pf_export ? 2 : 1);
#define NUMBER(name, expected) printf("\"" #name "\":%" PRIu64 ",", t->name);
    TARGET_F7_NUMBERS(NUMBER)
#undef NUMBER
    printf("\"placement\":\"NR\",\"binding\":[0,1,2,3,4,5,6,7],\"weightStationary\":true,"
           "\"bitSerial\":true,\"profile\":\"behavioral\",\"operation\":\"proj\","
           "\"dataflow\":\"%s\",\"ppaEnabled\":false},"
           "\"identities\":{"
           "\"caller\":\"0ce51e3ab99c9d02e7e0647f8dd159243f23f096e17cdcd3d9d3cb7912f6407a\","
           "\"candidate\":\"%s\","
           "\"tensors\":\"fa2769811b18e9ebf91ba6053a8b009fe2f17f2a6008b1d3b31d3e71fc12adad\","
           "\"costs\":\"c4c08d0908699751ab4223d9edcccce743079ab2004a1be06096ff0a7445cb99\","
           "\"proposal\":\"%s\"},"
           "\"candidate_origin\":\"%s\",\"seed\":null,\"rng\":\"%s\","
           "\"tensors\":{\"X\":{\"role\":\"activation\",\"shape\":[5,9],\"bits\":8,\"layout\":\"row_major\",\"virtual_shape\":[5,16]},"
           "\"W\":{\"role\":\"stored_weight\",\"shape\":[9,7],\"bits\":8,\"layout\":\"row_major\",\"virtual_shape\":[8,16],\"virtual_layout\":\"n_k\"},"
           "\"Y\":{\"role\":\"output\",\"shape\":[5,7],\"bits\":32,\"layout\":\"row_major\"}},"
           "\"tiling\":{\"A\":%d,\"P\":%d,\"batch_sizes\":[%d,%d],\"source\":",
           t->ip_af_export?"isap":t->pf_export ? "wspp" : "wsap",
           t->pf_export || t->ip_af_export ? t->candidate_sha256 : "cc7e60234578373f8af5755749069b0648d984b5c5f606bbe45163b7b8b47213",
           t->pf_export || t->ip_af_export ? t->proposal_sha256 : "a9704b71d22bdc88ff857951074088e2ce8cd2d3913a6ea534f9ef2406f57ea9",
           t->pf_export || t->ip_af_export ? "controlled_generator_probe" : "historical_initial_proposal",
           t->pf_export || t->ip_af_export ? "controlled_values_not_search" : "not_applicable",
           a_count, p_count, sizes[0], sizes[1]);
    source(tiling_function, tiling_line, t->pf_export ? "PF_SCR_ATOS" : "AF_SCR_ATOS");
    printf(",\"blocks\":[");
    if (t->pf_export) {
        /* 只列出真實 program 記錄順序，不在序列化時重建 PF 策略。 */
        int index = 0;
        for (int i = 0; i < count; ++i) if (ops[i].kind == PROGRAM) {
            const Op *o = &ops[i];
            printf("%s{\"block\":%d,\"p\":%d,\"a\":%d,\"set\":%d,\"atos\":%d}",
                   index ? "," : "", index, o->p, o->a, o->set, blocks[o->p * 2 + o->a][1]);
            ++index;
        }
    } else {
        for (int i = 0; i < 4; ++i)
            printf("%s{\"block\":%d,\"p\":%d,\"a\":%d,\"set\":%d,\"atos\":%d}",
                   i ? "," : "", i, i / 2, i % 2, blocks[i][0], blocks[i][1]);
    }
    printf("]},\"operations\":[");
    const char *kinds[] = {"cache_fill", "weight_load_program", "input_read", "compute", "cache_evict"};
    for (int i = 0; i < count; ++i) {
        const Op *o = &ops[i];
        printf("%s{\"id\":%d,\"source_order\":%d,\"kind\":\"%s\",\"source\":",
               i ? "," : "", o->id, i, kinds[o->kind - FILL]);
        const char *decision = o->kind == FILL ? (t->ip_af_export?"IP_CACHE_FILL":"WP_CACHE_PREFIX") :
            o->kind==EVICT?"IP_CACHE_EVICT":o->kind == PROGRAM ? "PROGRAM_BLOCK" :
            o->kind == READ ? (o->direct ? "DIRECT_SOURCE" : "CACHE_SOURCE") : (o->atos == 0 ? "AOR" : "FINAL_TOS");
        if (t->pf_export && o->kind == COMPUTE) decision = o->atos == 1 ? "STORE_TOS" : "FINAL_AOS";
        source(o->function, o->line, decision);
        int last = 0, overwritten_by = 0;
        for (int j = 0; j < count; ++j) {
            if ((ops[j].kind == COMPUTE && ops[j].program == o->id) ||
                (ops[j].kind == READ && ops[j].input == o->id)) last = ops[j].id;
            if (ops[j].kind == PROGRAM && ops[j].overwrite == o->id) overwritten_by = ops[j].id;
        }
        int payload = o->kind==EVICT?0:(o->k1 - o->k0) * (o->kind == PROGRAM ? o->n1 - o->n0 : 1);
        if(t->ip_af_export)printf(",\"input_group\":%d,\"cache_slot\":%d",o->m>=0?o->m/2:o->batch/2,
            o->kind==FILL || o->kind==EVICT || o->kind==READ ? o->m%2:-1);
        printf(",\"request\":\"request-0\",\"batch\":%d,\"m\":%d,\"p\":%d,\"a\":%d,"
               "\"k\":[%d,%d],\"n\":[%d,%d],\"virtual_k\":[%d,%d],\"virtual_n\":[%d,%d],"
               "\"tensor\":\"%s\",\"logical_payload_bytes\":%d,\"source_mode\":\"%s\","
               "\"input_producer\":%d,\"program_producer\":%d,\"set\":%d,\"version\":%d,"
               "\"previous_partial\":%d,\"owner\":%d,\"atos\":%d,\"psum_action\":\"%s\","
               "\"final\":%s,\"overwrites\":%d,\"overwritten_by\":%d,\"last_use\":%d,"
               "\"cache_offset\":%d,\"contribution_k\":[%d,%d],\"uses\":",
               o->batch, o->m, o->p, o->a, o->k0, o->k1, o->n0, o->n1,
               o->k0, o->kind == FILL || o->kind==EVICT ? 16 : o->k0 + 8, o->n0, o->kind == FILL || o->kind==EVICT ? 0 : o->n0 + 4,
               o->kind == PROGRAM ? "W" : o->kind == COMPUTE ? "Y" : "X",
               o->kind == COMPUTE ? 0 : payload,
               o->kind == READ ? (o->direct ? "direct" : "cached") : "external_or_logical",
               o->input, o->program, o->set, o->version, o->previous, o->owner, o->atos,
               o->kind != COMPUTE ? "none" : o->previous ? "accumulate_finalize" : "initialize_keep",
               o->kind == COMPUTE && (t->pf_export ? o->final : o->atos == 1) ? "true" : "false",
               o->overwrite, overwritten_by, last,
               o->kind == FILL || (o->kind == READ && !o->direct) ? (t->ip_af_export?o->m%2:o->m) * 9 + o->k0 : -1,
               o->kind == COMPUTE ? o->k0 : 0, o->kind == COMPUTE ? o->k1 : 0);
        uses(o->id); printf(",\"depends_on\":[");
        for (int j = 0; j < o->dependencies_count; ++j)
            printf("%s%d", j ? "," : "", o->dependencies[j]);
        printf("]}");
    }
    printf("],\"omissions\":[");
    for (int i = 0; i < count; ++i) {
        const Op *o = &ops[i];
        printf("%s{\"id\":\"omit-%d\",\"operation\":%d,\"source\":", i ? "," : "", o->id, o->id);
        source(o->function, o->line, "TARGET_REPRESENTATION");
        printf(",\"rule\":\"effective_ranges_and_target_encoding\","
               "\"reason\":\"虛擬範圍扣除有效範圍沒有 payload；原 packing、reward 與 read-slot 由目標表示取代，未產生原 ISA 實例\","
               "\"physical_children_expanded\":false}");
    }
    for (int b = 0; b < (t->ip_af_export?6:batch_count); ++b) {
        printf(",{\"id\":\"nop-batch-%d\",\"batch\":%d,\"source\":", b, b);
        source(t->ip_af_export?"is_process":"ws_process", batch_lines[b], "LEGACY_PROGRAM_WAIT");
        printf(",\"rule\":\"replace_packing_wait_with_program_dependency\","
               "\"reason\":\"原 bus pipeline 等待不轉成目標 NOP；保留 program-to-use 相依\",\"legacy_instruction_count\":null}");
    }
    printf("],\"tensor_payload_read\":false,\"numeric_execution\":false,\"operations_exported\":true,"
           "\"physical_generation_created\":false,\"transaction_boundaries_assigned\":false,"
           "\"counts_supported\":false,\"backend_cycles\":null,\"costs\":null}\n");
    return ferror(stdout) ? 2 : 0;
}
