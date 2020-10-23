#include <stdio.h>
#include <math.h>

void quantizeInt8InFloat(const size_t N, const float *src, float *dst) {
    for (size_t i = 0; i < N; i++) {
        float s = src[i];
        dst[i] = (s < -128.0) ? -128.0 : ((s > 127.0) ? 127.0 : roundf(src[i]));
    }
}

int main(void) {
    const size_t N = 256;
    float src[N], dst[N];

    /* srcを初期化 */
    for (size_t i = 0; i < N; i++) {
        src[i] = (i - 128.0) * 20 / 19;
    }

    quantizeInt8InFloat(N, src, dst);

    printf("%5s  %7s  %7s\n", "Index", "Src", "Dst");
    /* dstを表示 */
    for (size_t i = 0; i < N; i++) {
       printf("%5lu  %7.2f  %7.2f\n", i, src[i], dst[i]);
           /* %lu は unsigned long の意味で、標準的なx64マシンのLinuxでのsize_tの内部実装 */
    }

    return 0;
}