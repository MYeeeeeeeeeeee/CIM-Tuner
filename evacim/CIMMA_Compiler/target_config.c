#include "target_config.h"
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    const char *name;
    size_t offset;
    uint64_t expected;
    const char *text;
} Field;

#define NUMBER_FIELD(name, expected) {#name, offsetof(TargetConfig, name), expected, NULL},
static const Field fields[] = {
    TARGET_F7_NUMBERS(NUMBER_FIELD)
    {"weightStationary", 0, 0, "true"}, {"bitSerial", 0, 0, "true"},
    {"placement", 0, 0, "NR"}, {"binding", 0, 0, "0,1,2,3,4,5,6,7"},
    {"profile", 0, 0, "behavioral"}, {"operation", 0, 0, "proj"},
    {"dataflow", 0, 0, "wsap"}, {"ppaEnabled", 0, 0, "false"}
};
#undef NUMBER_FIELD

static int reject(const char *reason, const char *field)
{
    fprintf(stderr, "target-f7: %s: %s\n", reason, field);
    return 2;
}

static int decimal(const char *s, uint64_t *value)
{
    uint64_t number = 0;
    if (!*s) return 0;
    for (; *s; ++s) {
        if (*s < '0' || *s > '9') return 0;
        const unsigned digit = (unsigned)(*s - '0');
        if (number > (UINT64_MAX - digit) / 10) return 0;
        number = number * 10 + digit;
    }
    *value = number;
    return 1;
}

static int parse(int argc, char **argv, TargetConfig *target, int pf)
{
    unsigned char seen[sizeof(fields) / sizeof(fields[0])] = {0};
    memset(target, 0, sizeof(*target));
    target->ip_export = pf >= 2;
    target->pf_export = pf == 1 || pf == 3;
    for (int arg = 0; arg < argc; ++arg) {
        const char *equals = strchr(argv[arg], '=');
        if (!equals || equals == argv[arg]) return reject("需要 key=value", argv[arg]);
        const size_t length = (size_t)(equals - argv[arg]);
        char *hash = pf && length == strlen("candidateSha256") &&
            !strncmp(argv[arg], "candidateSha256", length) ? target->candidate_sha256 :
            pf && length == strlen("proposalSha256") && !strncmp(argv[arg], "proposalSha256", length) ?
            target->proposal_sha256 : NULL;
        if (hash) {
            if (*hash) return reject("重複欄位", argv[arg]);
            const char *value = equals + 1;
            if (strlen(value) != 64 || strspn(value, "0123456789abcdef") != 64)
                return reject("需要小寫 SHA-256", argv[arg]);
            memcpy(hash, value, 65);
            continue;
        }
        size_t i;
        for (i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i)
            if (strlen(fields[i].name) == length && !strncmp(fields[i].name, argv[arg], length))
                break;
        if (i == sizeof(fields) / sizeof(fields[0])) return reject("未知或混用欄位", argv[arg]);
        if (seen[i]) return reject("重複欄位", fields[i].name);
        seen[i] = 1;
        const char *value = equals + 1;
        if (fields[i].text) {
            const char *expected = pf && !strcmp(fields[i].name, "dataflow") ? (pf == 3 ? "ispp" : pf == 2 ? "isap" : "wspp") : fields[i].text;
            if (strcmp(value, expected)) return reject("不支援的固定 F7 配置", fields[i].name);
        } else {
            uint64_t number;
            if (!decimal(value, &number)) return reject("無效或溢位的非負整數", fields[i].name);
            if (number != fields[i].expected) return reject("不支援的固定 F7 配置", fields[i].name);
            *(uint64_t *)((unsigned char *)target + fields[i].offset) = number;
        }
    }
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i)
        if (!seen[i]) return reject("缺少必要欄位", fields[i].name);
    if (pf && (!*target->candidate_sha256 || !*target->proposal_sha256))
        return reject("缺少必要欄位", "candidateSha256/proposalSha256");
    return 0;
}

int target_parse(int argc, char **argv, TargetConfig *target)
{
    return parse(argc, argv, target, 0);
}

int target_parse_pf(int argc, char **argv, TargetConfig *target)
{
    return parse(argc, argv, target, 1);
}

int target_parse_ip_af(int argc, char **argv, TargetConfig *target)
{
    return parse(argc, argv, target, 2);
}

static uint64_t ceil_div(uint64_t n, uint64_t d)
{
    return n / d + (n % d != 0);
}

uint64_t target_cache_vectors(const TargetConfig *t)
{
    const uint64_t count = t->inputCacheBytes / t->k;
    return count < t->m ? count : t->m;
}

/* 僅作已驗配置的索引計算；無效範圍不改寫輸出，也不讀取 tensor。 */
int target_x_index(const TargetConfig *t, uint64_t m, uint64_t k,
                   uint64_t *effective, uint64_t *virtual_index)
{
    if (m >= t->m || k >= t->k) return 0;
    const uint64_t al = t->logicalRows * t->gridCols;
    *effective = m * t->k + k;
    *virtual_index = m * ceil_div(t->k, al) * al + k;
    return 1;
}

int target_w_index(const TargetConfig *t, uint64_t k, uint64_t n,
                   uint64_t *effective, uint64_t *virtual_index)
{
    if (k >= t->k || n >= t->n) return 0;
    const uint64_t al = t->logicalRows * t->gridCols;
    *effective = k * t->n + n;
    *virtual_index = n * ceil_div(t->k, al) * al + k;
    return 1;
}

int target_fragment(const TargetConfig *t, uint64_t p, uint64_t a,
                    uint64_t row, uint64_t col, uint64_t *rows, uint64_t *cols)
{
    const uint64_t al = t->logicalRows * t->gridCols;
    const uint64_t pc = t->logicalCols * t->gridRows;
    if (p >= ceil_div(t->n, pc) || a >= ceil_div(t->k, al) ||
        row >= t->gridRows || col >= t->gridCols) return 0;
    const uint64_t k = a * al + col * t->logicalRows;
    const uint64_t n = p * pc + row * t->logicalCols;
    if (k >= t->k || n >= t->n) return 0;
    *rows = t->k - k < t->logicalRows ? t->k - k : t->logicalRows;
    *cols = t->n - n < t->logicalCols ? t->n - n : t->logicalCols;
    return 1;
}

int target_line_operations(uint64_t bytes, uint64_t width, uint64_t *operations)
{
    if (!width) return 0;
    *operations = ceil_div(bytes, width);
    return 1;
}

static int target_report(const TargetConfig *t, int a, int p, int batches,
                    const int *batch_sizes, int **sets, int **atos, int emit)
{
    const uint64_t al = t->logicalRows * t->gridCols;
    const uint64_t pc = t->logicalCols * t->gridRows;
    if ((uint64_t)a != ceil_div(t->k, al) || (uint64_t)p != ceil_div(t->n, pc) ||
        (uint64_t)batches != ceil_div((uint64_t)a * p, t->macroStorageSets))
        return reject("共用分塊結果不符", "proj_process");
    /* 核對真實共用 AF／SCR 決策，不匯出操作或重新產生策略。 */
    for (int b = 0; b < batches; ++b)
        if (batch_sizes[b] != (int)t->macroStorageSets)
            return reject("SCR 批次不符", "weight_update_ls");
    for (int ip = 0; ip < p; ++ip)
        for (int ia = 0; ia < a; ++ia)
            if (sets[ip][ia] != (t->pf_export ? ip : ia) ||
                atos[ip][ia] != (t->pf_export ? ia + 1 : (ia == a - 1 ? 1 : 0)))
                return reject(t->pf_export ? "PF／SCR 結果不符" : "AF／SCR 結果不符", "ls_matrix/atos_matrix");

    const uint64_t cache = target_cache_vectors(t);
    const uint64_t slot = t->logicalCols * (t->outputBits / 8);
    if (cache != 2 || cache * t->k > t->inputCacheBytes ||
        t->inputCacheBytes > t->globalBufferBytes ||
        t->outputContextBytes > t->globalBufferBytes - t->inputCacheBytes ||
        t->outputContextBytes / slot != 512)
        return reject("有效容量不符", "buffers");

    uint64_t fragments = 0, program_bytes = 0;
    for (int ip = 0; ip < p; ++ip)
        for (int ia = 0; ia < a; ++ia)
            for (uint64_t row = 0; row < t->gridRows; ++row)
                for (uint64_t col = 0; col < t->gridCols; ++col) {
                    uint64_t r, c;
                    if (!target_fragment(t, ip, ia, row, col, &r, &c)) continue;
                    if (r * c > t->peLocalBufferBytes || r * c >
                        t->globalBufferBytes - t->inputCacheBytes - t->outputContextBytes)
                        return reject("有效 fragment 超出暫存配額", "fragment");
                    ++fragments;
                    program_bytes += r * c;
                }
    if (program_bytes != t->k * t->n) return reject("有效範圍不完整", "weight bytes");

    uint64_t x, xv, w, wv, r, c, gb, pe, link;
    if (!target_x_index(t, 1, 8, &x, &xv) || !target_w_index(t, 8, 6, &w, &wv) ||
        !target_fragment(t, 1, 1, 1, 0, &r, &c) ||
        !target_line_operations(1, t->globalBufferReadBytesPerCycle, &gb) ||
        !target_line_operations(1, t->peLocalBufferReadBytesPerCycle, &pe) ||
        !target_line_operations(1, t->tileInterconnectBytesPerCycle, &link))
        return reject("尾端診斷失敗", "indices");

    if (!emit) return 0;
    printf("{\"status\":\"passed\",\"mode\":\"target-f7\",\"config\":{");
#define PRINT_NUMBER(name, expected) printf("\"" #name "\":%" PRIu64 ",", t->name);
    TARGET_F7_NUMBERS(PRINT_NUMBER)
#undef PRINT_NUMBER
    printf("\"placement\":\"NR\",\"binding\":[0,1,2,3,4,5,6,7],"
           "\"weightStationary\":true,\"bitSerial\":true,\"profile\":\"behavioral\","
           "\"operation\":\"proj\",\"dataflow\":\"wsap\",\"ppaEnabled\":false},");
    printf("\"derived\":{\"AL\":%" PRIu64 ",\"PC\":%" PRIu64
           ",\"A\":%d,\"P\":%d,\"residentBatches\":%d,\"virtualK\":%" PRIu64
           ",\"virtualN\":%" PRIu64 ",\"macroCount\":%" PRIu64
           ",\"storedBits\":%" PRIu64 ",\"cacheVectors\":%" PRIu64
           ",\"cachePayloadBytes\":%" PRIu64 ",\"directVectors\":%" PRIu64
           ",\"contextSlotBytes\":%" PRIu64 ",\"outputSlots\":%" PRIu64
           ",\"ingressPartitionBytes\":%" PRIu64 ",\"xBytes\":%" PRIu64
           ",\"wBytes\":%" PRIu64 ",\"yBytes\":%" PRIu64
           ",\"validFragments\":%" PRIu64 "},",
           al, pc, a, p, batches, (uint64_t)a * al, (uint64_t)p * pc,
           t->numTiles * t->pesPerTile * t->numMacros,
           t->numTiles * t->pesPerTile * t->numMacros * t->macroStorageSets *
               t->logicalRows * t->logicalCols * t->weightBits,
           cache, cache * t->k, t->m - cache, slot, t->outputContextBytes / slot,
           t->globalBufferBytes - t->inputCacheBytes - t->outputContextBytes,
           t->m * t->k, program_bytes, t->m * t->n * (t->outputBits / 8), fragments);
    printf("\"indexExamples\":{\"xEffective\":%" PRIu64 ",\"xVirtual\":%" PRIu64
           ",\"wEffective\":%" PRIu64 ",\"wVirtual\":%" PRIu64 "},"
           "\"tail\":{\"rows\":%" PRIu64 ",\"cols\":%" PRIu64
           ",\"weightBytes\":%" PRIu64 ",\"inputBytes\":%" PRIu64
           ",\"outputBytes\":%" PRIu64 "},", x, xv, w, wv, r, c, r * c, r, c * 4);
    printf("\"roundingExamples\":{\"oneByteGbLineAccesses\":%" PRIu64
           ",\"oneBytePeLineAccesses\":%" PRIu64 ",\"oneByteLinkPhaseCycles\":%" PRIu64
           "},\"policy\":{\"capacity\":\"effective_bytes\",\"activationPlanes\":8,"
           "\"validLanesOnly\":true,\"fixedStages\":true,\"roundPerTransaction\":true,"
           "\"addressAlignmentModeled\":false},\"shared_tiling_checked\":true,"
           "\"tensor_data_read\":false,\"operations_exported\":false,"
           "\"counts_supported\":false,\"backend_cycles\":null,\"costs\":null,"
           "\"legacyISDepth\":null,\"legacyOSDepth\":null,\"legacyBusWidth\":null}\n",
           gb, pe, t->tileInterconnectStartupCycles + link);
    return 0;
}

int target_validate(const TargetConfig *t, int a, int p, int batches,
                    const int *sizes, int **sets, int **atos)
{
    return target_report(t, a, p, batches, sizes, sets, atos, 0);
}

int target_diagnose(const TargetConfig *t, int a, int p, int batches,
                    const int *sizes, int **sets, int **atos)
{
    return target_report(t, a, p, batches, sizes, sets, atos, 1);
}

int target_parse_ip_pf(int argc, char **argv, TargetConfig *target)
{
    return parse(argc, argv, target, 3);
}
