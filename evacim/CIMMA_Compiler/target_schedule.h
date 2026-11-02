#ifndef TARGET_SCHEDULE_H
#define TARGET_SCHEDULE_H

#include "target_config.h"

/* 只記錄實際決策；logical version 不代表 physical program／commit。 */
void schedule_begin(const TargetConfig *target);
void schedule_tiling(int a, int p, int batches, const int *sizes,
                     int **sets, int **atos, const char *function, int line);
void schedule_cache(int vectors, const char *function, int line);
void schedule_batch(int batch, const char *function, int line);
void schedule_program(int p, int a, int set, const char *function, int line);
void schedule_read(int m, int p, int a, int direct, const char *function, int line);
void schedule_compute(int m, int p, int a, int set, int atos, int owner,
                      const char *function, int line);
int schedule_write(void);

#endif
