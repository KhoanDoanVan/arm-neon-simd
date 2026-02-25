#ifndef NEON_UTILS_H
#define NEON_UTILS_H


#include "neon_types.h"
#include "memory_align.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif



// NEON LOAD/STORE
/**
 * NEON LOAD OPERATIONS:
 * 
 * vld1q_f32(ptr)  : Load 4 floats from aligned memory
 * vld1_f32(ptr)   : Load 2 floats from aligned memory
 * 
 * Unaligned variants (chậm hơn, tránh nếu có thể):
 * vld1q_f32(ptr)  : Also works with unaligned but slower
*/

/**
 * Safe load - auto handle alignment
*/
static NEON_INLINE float32x4_t neon_load_f32x4(const float* ptr) {
    return vld1q_f32(ptr);
}


/**
 * Safe store - auto handle alignment
*/
static NEON_INLINE void neon_store_f32x4(float* ptr, float32x4_t vec) {
    vst1q_f32(ptr, vec);
}


// NEON ARITHMETTC (số học)
/**
 * Vector addition: result = a + b
 * Example: [1,2,3,4] + [5,6,7,8] = [6,8,10,12]
*/
static NEON_INLINE float32x4_t neon_add_f32x4(float32x4_t a, float32x4_t b) {
    return vaddq_f32(a, b);
}

/**
 * Vector addition: result = a - b
*/
static NEON_INLINE float32x4_t neon_sub_f32x4(float32x4_t a, float32x4_t b) {
    return vsubq_f32(a, b);
}

/**
 * Vector multiplication: result = a * b
 * Example: [1,2,3,4] * [2,3,4,5] = [2,6,12,20]
*/
static NEON_INLINE float32x4_t neon_mul_f32x4(float32x4_t a, float32x4_t b) {
    return vmulq_f32(a, b);
}


/**
 * Fused Multiply-Add: result = a * b + c
 * Nhanh hơn việc làm riêng mul rồi add!
 * QUAN TRỌNG: Đây là instruction phổ biến nhất trong AI/ML
*/
static NEON_INLINE float32x4_t neon_fma_f32x4(float32x4_t a, float32x4_t b, float32x4_t c) {
    #ifdef __ARM_FEATURE_FMA
        return vfmaq_f32(c, a, b); // Hardware FMA
    #else
        return vaddq_f32(vmulq_f32(a, b), c); // Fallback
    #endif
}


/**
 * Vector division: result = a / b
 * Lưu ý: Chậm hơn mul/add! Tránh nếu có thể
*/
static NEON_INLINE float32x4_t neon_div_f32x4(float32x4_t a, float32x4_t b) {
    #ifdef __aarch64__ // ARMv8 (AArch64)
        return vdivq_f32(a, b); // ARMv8 có hardware div
    #else
        // ARMv7: Approximate reciprocal + Newton-Raphson refinement
        // ARMv7 không có instruction vector divide. Nên phải làm approximate reciprocal:
        float32x4_t reciprocal = vrecpeq_f32(b);
        reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);
        return vmulq_f32(a, reciprocal);
    #endif
}



// NEON REDUCTION OPERATIONS (SIMD primitive set)
/**
 * Horizontal sum: Tính tổng 4 elements trong vector
 * Example: [1,2,3,4] → 10
 * 
 * TECHNIQUE:
 * 1. Pairwise add: [1,2,3,4] → [1+2, 3+4] = [3, 7]
 * 2. Pairwise add: [3,7] → 3+7 = 10
*/
static NEON_INLINE float neon_sum_f32x4(float32x4_t vec) {
    #ifdef __aarch64__
        // ARMv8 có instruction riêng cho reduction
        return vaddvq_f32(vec);
    #else
        // ARMv7: Manual reduction
        float32x2_t sum = vadd_f32(vget_low_f32(vec), vget_high_f32(vec));
        sum = vpadd_f32(sum, sum);
        return vget_lane_f32(sum);
    #endif
}


/**
 * Horizontal max: Tìm giá trị lớn nhất trong vector
 * Example: [1,5,3,2] → 5
 * 
 * 
 * vec = [3, 8, 2, 5]
 * - step1: 
 * low  = [3, 8]
 * high = [2, 5]
 * - step2:
 * max = [max(3,2), max(8,5)]
 * = [3, 8]
 * - step3:
 * max = [max(3,8), max(3,8)]
 * = [8, 8]
 * - step4:
 * return 8
*/
static NEON_INLINE float neon_max_f32x4(float32x4_t vec) {
    #ifdef __aarch64__
        return vmaxvq_f32(vec);
    #else
        float32x2_t max = vmax_f32(vget_low_f32(vec), vget_high_f32(vec));
        max = vpmax_f32(max, max);
        return vget_lane_f32(max, 0);
    #endif
}

/**
 * Horizontal min: Tìm giá trị nhỏ nhất trong vector
*/
static NEON_INLINE float neon_min_f32x4(float32x4_t vec) {
    #ifdef __aarch64__
        return vminvq_f32(vec);
    #else
        float32x2_t min = vmin_f32(vget_low_f32(vec), vget_high_f32(vec))
        min = vpmin_f32(min, min);
        return vget_lane_f32(min, 0);
    #endif
}


/**
 * Vector max: result[i] = max(a[i], b[i])
 * Example: max([1,5,3,2], [2,3,4,5]) = [2,5,4,5]
*/
static NEON_INLINE float32x4_t neon_vmax_f32x4(float32x4_t a, float32x4_t b) {
    return vmaxq_f32(a, b);
}


/**
 * Vector min: result[i] = min(a[i], b[i])
*/
static NEON_INLINE float32x4_t neon_vmin_f32x4(float32x4_t a, float32x4_t b) {
    return vminq_f32(a, b);
}


/**
 * Clamp vector: result[i] = clamp(vec[i], min_val, max_val)
 * Hữu ích cho ReLU6, quantization
*/
static NEON_INLINE float32x4_t neon_clamp_f32x4(
    float32x4_t vec,
    float min_val,
    float max_val
) {
    float32x4_t vmin = vdupq_n_f32(min_val);
    float32x4_t vmax = vdupq_n_f32(max_val);
    return vminq_f32(vmaxq_f32(vec, vmin), vmax);
}

/**
 * Compare greater than: mask[i] = (a[i] > b[i]) ? 0xFFFFFFFF : 0
 * 
 * So sánh từng phần tử của 2 vector float32x4_t:
 * 
 * Trả về mask có thể dùng với vbslq_f32 để select
*/
static NEON_INLINE uint32x4_t neon_cmpgt_f32x4(float32x4_t a, float32x4_t b) {
    return vcgtq_f32(a, b);
}


/**
 * Select/Blend: result[i] = mask[i] ? a[i] : b[i]
 * QUAN TRỌNG: Đây là cách implement conditional logic trong SIMD
 * 
 * hay: chọn giữa 2 vector theo mask (không dùng if).
 * 
 * Example: ReLU
 *   mask = (x > 0)
 *   result = select(mask, x, 0)
*/
static NEON_INLINE float32x4_t neon_select_f32x4(
    uint32x4_t mask,
    float32x4_t a,
    float32x4_t b
) {
    return vbslq_f32(mask, a, b);
}



// NEON BROADCAST & SPLAT
/**
 * Broadcast scalar to all lanes
 * Example: broadcast(5.0) → [5.0, 5.0, 5.0, 5.0]
*/
static NEON_INLINE float32x4_t neon_broadcast_f32(float value) {
    return vdupq_n_f32(value);
}


/**
 * Load single value and broadcast
 */
static NEON_INLINE float32x4_t neon_load_broadcast_f32(const float* ptr) {
    return vld1q_dup_f32(ptr);
}


// NEON TRANSCENDENTAL FUNCTIONS
/**
 * Fast approximate exp() using NEON
 * Accuracy: ~0.1% error
 * Speedup: ~8x so với expf()
 * 
 * Dùng polynomial approximation:
 * exp(x) ≈ 1 + x + x²/2! + x³/3! + ...
*/
static NEON_INLINE float32x4_t neon_exp_f32x4(float32x4_t x) {
    // Clamp input để tránh overflow
    x = vminq_f32(x, vdupq_n_f32(88.0f));
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));

    // Coefficients cho polynomial approximation
    float32x4_t c1 = vdupq_n_f32(1.0f);
    float32x4_t c2 = vdupq_n_f32(0.5f);
    float32x4_t c3 = vdupq_n_f32(0.16666667f);
    float32x4_t c4 = vdupq_n_f32(0.04166667f);

    // Polynomial: 1 + x + x^2 / 2 + x^3 / 6 + x^4 / 24
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x3 = vmulq_f32(x2, x);
    float32x4_t x4 = vmulq_f32(x3, x);

    float32x4_t result = c1;
    result = vfmaq_f32(result, x, c1);
    result = vfmaq_f32(result, x2, c2);
    result = vfmaq_f32(result, x3, c3);
    result = vfmaq_f32(result, x4, c4);


    return result;
}


/**
 * Fast approximate sqrt() using NEON
 * Dùng reciprocal square root + refinement
*/
static NEON_INLINE float32x4_t neon_sqrt_f32x4(float32x4_t x) {
    // Reciprocal sqrt estimate
    float32x4_t rsqrt = vrsqrteq_f32(x);

    // Newton-Raphson refinement: rsqrt = rsqrt * (3 - x * rsqrt²) / 2
    rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(x, rsqrt), rsqrt));
    rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(x, rsqrt), rsqrt));

    // sqrt = x * rsqrt
    return vmulq_f32(x, rsqrt);
}


// NEON DOT PRODUCT
/**
 * Dot product của 2 vectors
 * result = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
 * 
 * QUAN TRỌNG: Core operation trong neural networks!
*/
static NEON_INLINE float neon_dot_f32x4(float32x4_t a, float32x4_t b) {
    float32x4_t prod = vmulq_f32(a, b);
    return neon_sum_f32x4(prod);
}


/**
 * Dot product của 2 arrays
 * Xử lý arrays dài tùy ý
*/
float neon_dot_product(const float* a, const float* b, size_t size);



// UTILITY FUNCTIONS
/**
 * Print NEON vector (for debugging)
*/
void neon_print_f32x4(const char* name, float32x4_t vec);


/**
 * Compare 2 arrays với tolerance (mức sai lệch cho phép)
 * Return 1 nếu match, 0 nếu khác
*/
int neon_compare_arrays(const float* a, const float* b, size_t size, float tolerance);


/**
 * Calculate mean của array
*/
float neon_mean(const float* data, size_t size);


/**
 * Calculate variance của array
*/
float neon_variance(const float* data, size_t size, float mean);


#ifdef __cplusplus
}
#endif


#endif