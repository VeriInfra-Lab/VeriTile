#include <stdint.h>
uint32_t off(uint32_t block,uint32_t pid,uint32_t lane){return pid*block+lane;}
int mask(uint32_t n,uint32_t block,uint32_t pid,uint32_t lane){return (int32_t)off(block,pid,lane)<(int32_t)n;}
uint64_t pointer(uint64_t data,uint32_t width,uint32_t block,uint32_t pid,uint32_t lane){return data+(uint64_t)(int64_t)(int32_t)off(block,pid,lane)*width;}
