#include <stdint.h>
uint32_t off(uint32_t pid,uint32_t lane){return pid*1024u+lane;}
int mask(uint32_t n,uint32_t pid,uint32_t lane){return (int32_t)off(pid,lane)<(int32_t)n;}
uint64_t pointer(uint64_t data,uint32_t bytes,uint32_t pid,uint32_t lane){return data+(uint64_t)(int64_t)(int32_t)off(pid,lane)*bytes;}
