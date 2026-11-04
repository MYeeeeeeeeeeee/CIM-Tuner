#ifndef TARGET_CONFIG_H
#define TARGET_CONFIG_H

#include <stdint.h>

/* 固定 F7 配置白名單；欄位是配置契約，不是可搜尋的硬體自由度。 */
#define TARGET_F7_NUMBERS(X) \
    X(macroStorageSets, 2) X(macroBankSelectCycles, 2) \
    X(inputCacheBytes, 18) X(outputContextBytes, 4096) \
    X(globalBufferBytes, 65536) X(peLocalBufferBytes, 4096) X(tileBufferBytes, 4096) \
    X(macroLoadWeightCycles, 1) X(macroLoadInputCycles, 1) X(macroComputeCycles, 1) \
    X(macroAccumulateCycles, 1) X(macroWriteBackCycles, 1) \
    X(tileAccumulateCycles, 3) X(crossTileAccumulateCycles, 4) \
    X(globalBufferReadBytesPerCycle, 64) X(globalBufferWriteBytesPerCycle, 64) \
    X(tileInterconnectBytesPerCycle, 32) X(tileInterconnectStartupCycles, 1) \
    X(peLocalBufferReadBytesPerCycle, 32) X(peLocalBufferWriteBytesPerCycle, 32) \
    X(tileBufferReadBytesPerCycle, 32) X(tileBufferWriteBytesPerCycle, 32) \
    X(tileLocalInterconnectBytesPerCycle, 32) X(peMacroInterconnectBytesPerCycle, 32) \
    X(frequencyMHz, 138) X(m, 5) X(k, 9) X(n, 7) \
    X(logicalRows, 2) X(logicalCols, 2) X(numTiles, 2) X(pesPerTile, 2) X(numMacros, 2) \
    X(inputBits, 8) X(weightBits, 8) X(gridRows, 2) X(gridCols, 4) X(outputBits, 32)

typedef struct {
#define TARGET_MEMBER(name, expected) uint64_t name;
    TARGET_F7_NUMBERS(TARGET_MEMBER)
#undef TARGET_MEMBER
    int pf_export;
    char candidate_sha256[65], proposal_sha256[65];
} TargetConfig;

/* 成功回傳 0；失敗回傳 2，不建立檔案。 */
int target_parse(int argc, char **argv, TargetConfig *target);
int target_parse_pf(int argc, char **argv, TargetConfig *target);
uint64_t target_cache_vectors(const TargetConfig *target);
int target_x_index(const TargetConfig *, uint64_t m, uint64_t k,
                   uint64_t *effective, uint64_t *virtual_index);
int target_w_index(const TargetConfig *, uint64_t k, uint64_t n,
                   uint64_t *effective, uint64_t *virtual_index);
int target_fragment(const TargetConfig *, uint64_t p, uint64_t a,
                    uint64_t row, uint64_t col, uint64_t *rows, uint64_t *cols);
int target_line_operations(uint64_t bytes, uint64_t width, uint64_t *operations);
int target_validate(const TargetConfig *, int a, int p, int batches,
                    const int *batch_sizes, int **sets, int **atos);
int target_diagnose(const TargetConfig *, int a, int p, int batches,
                    const int *batch_sizes, int **sets, int **atos);

#endif
