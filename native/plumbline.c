/* Plumbline — the lossless JPEG (SOF3) entropy decoder, in portable C.
 *
 * This is the third implementation of the same decoder. `src/plumbline/reference.py`
 * is the oracle: readable, dependency-free, and slow. `src/plumbline/turbo.py`
 * accelerates it with numba. This file replaces the numba kernels with plain
 * C11 so the application neither JIT-compiles on first use nor ships LLVM,
 * and must agree with the oracle bit for bit — `tests/test_ljpeg_native.py`
 * decodes real discs and random scans through both and diffs the results.
 *
 * The caller (src/plumbline/native.py) parses the JPEG headers, validates
 * them, and hands this file only the numbers: geometry, tables, and the raw
 * (still byte-stuffed) entropy-coded segment. This file owns the three inner
 * loops that cost the time:
 *
 *   - undoing the byte stuffing (memchr between 0xFF bytes, block copies)
 *   - building the fused Huffman lookup (window of sixteen bits -> signed
 *     difference and bits consumed, in one table entry)
 *   - the scan itself (one unaligned 64-bit load per sample, no per-byte
 *     refill loop, no bounds checks in the common path)
 *
 * House rule, inherited from the whole project: decode correctly or return an
 * error — never an image that merely looks decoded. A scan that ends before
 * the image does is refused (PLUMBLINE_TRUNCATED), exactly as turbo
 * refuses it, rather than padded with invented samples the way a permissive
 * decoder would.
 *
 * No global state, no threads, no dependencies beyond libc. Compiles with
 * MSVC, gcc, clang and `zig cc`.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#  include <stdlib.h>
#  define BSWAP64(x) _byteswap_uint64(x)
#  define EXPORT __declspec(dllexport)
#else
#  define BSWAP64(x) __builtin_bswap64(x)
#  if defined(_WIN32)
#    define EXPORT __declspec(dllexport)
#  else
#    define EXPORT __attribute__((visibility("default")))
#  endif
#endif

#define PLUMBLINE_ABI 1

/* Error codes. 0 is success; anything negative is a refusal. */
#define PLUMBLINE_OK              0
#define PLUMBLINE_NO_MEMORY      -1
#define PLUMBLINE_TRUNCATED      -2   /* entropy data ends before the image */
#define PLUMBLINE_BAD_TABLE      -3   /* counts promise more symbols than given */
#define PLUMBLINE_BAD_SSSS       -4   /* a symbol above 16 is not a bit count */
#define PLUMBLINE_BAD_ARGS       -5
#define PLUMBLINE_BAD_CODE       -6   /* a code the scan's table does not define */

#define WINDOW_BITS 16
#define WINDOW_SIZE (1 << WINDOW_BITS)
#define MAX_PRECISION 16
#define SSSS_16_DIFF 32768            /* T.81 H.1.2.2: no mantissa bits follow */

/* ------------------------------------------------------------------------- */
/* Huffman tables                                                            */
/* ------------------------------------------------------------------------- */

/* Two lookups per table, both indexed by the next sixteen bits of the stream.
 *
 * `fused[w]` packs the whole answer for the common case into one int32:
 * bits 6..31 hold the signed difference, bits 0..5 how many bits the code and
 * its mantissa consumed together. Zero in the low six bits means the pair was
 * too long to fuse (or the window matches no code) and the slow path reads
 * `raw[w]` instead: code length in the high byte, SSSS in the low byte, both
 * zero where no code matches — which, matching the oracle, consumes nothing
 * and yields a zero difference. */
/* `first[w >> (16 - FIRST_BITS)]` is a copy of the fused entry for the pairs
 * short enough to be decided by the first FIRST_BITS bits alone. At 16 KiB it
 * lives in L1 where the 256 KiB fused table cannot, and the table lookup sits
 * on the loop-carried critical path — the next sample's window is unknown
 * until this sample's bit count arrives — so its latency is paid per sample.
 * Twelve bits measured best on the real discs: ten starved the 12-bit
 * mammograms, whose differences are wide, back onto the big table. */
#define FIRST_BITS 12
#define FIRST_SIZE (1 << FIRST_BITS)

/* `undefined` is one flag shared by every table of a frame, raised when a
 * window matches none of the table's codes. It is written only from the slow
 * path — the fused path cannot reach a window with no code, because every
 * window a code covers has a nonzero length — so the loop that carries the
 * time never touches it. A table is allowed to leave code space unclaimed;
 * a scan that lands in it is not the scan that table encoded. */
typedef struct {
    int32_t  *fused;
    uint16_t *raw;
    int32_t  *first;
    int32_t  *undefined;
} table_t;

static int build_table(const uint8_t counts[16], const uint8_t *symbols,
                       int32_t nsym, table_t *table)
{
    memset(table->fused, 0, WINDOW_SIZE * sizeof(int32_t));
    memset(table->raw,   0, WINDOW_SIZE * sizeof(uint16_t));

    int64_t code = 0;
    int32_t index = 0;
    for (int bits = 1; bits <= 16; bits++) {
        for (int entry = 0; entry < counts[bits - 1]; entry++) {
            if (index >= nsym)
                return PLUMBLINE_BAD_TABLE;
            int32_t ssss = symbols[index];
            if (ssss > MAX_PRECISION)
                return PLUMBLINE_BAD_SSSS;

            /* Each code owns the window range it prefixes. A table that
             * declares more codes of a length than that length has room for
             * walks the code counter past the last window, and every decoder
             * loses a different symbol off the end. The caller rejects such a
             * table before we are called; refuse it here too, so this library
             * is safe for anyone who calls it without that Python in front. */
            int64_t low  = code << (16 - bits);
            int64_t high = low + ((int64_t)1 << (16 - bits));
            if (high > WINDOW_SIZE)
                return PLUMBLINE_BAD_TABLE;

            int32_t total = (ssss == MAX_PRECISION) ? bits : bits + ssss;
            int32_t span  = (1 << ssss) - 1;
            int32_t half  = ssss ? (1 << (ssss - 1)) : 0;
            uint16_t pair = (uint16_t)((bits << 8) | ssss);

            for (int64_t window = low; window < high; window++) {
                table->raw[window] = pair;
                if (total > WINDOW_BITS)
                    continue;              /* mantissa runs past the window */
                int32_t value;
                if (ssss == MAX_PRECISION) {
                    value = SSSS_16_DIFF;
                } else {
                    int32_t mantissa =
                        (int32_t)(window >> (WINDOW_BITS - total)) & span;
                    value = mantissa < half ? mantissa - span : mantissa;
                }
                table->fused[window] =
                    (int32_t)(((uint32_t)value << 6) | (uint32_t)total);
            }
            index++;
            code++;
        }
        code <<= 1;
    }

    for (int32_t short_window = 0; short_window < FIRST_SIZE; short_window++) {
        int32_t entry = table->fused[short_window << (WINDOW_BITS - FIRST_BITS)];
        table->first[short_window] = (entry & 63) <= FIRST_BITS ? entry : 0;
    }
    return PLUMBLINE_OK;
}

/* ------------------------------------------------------------------------- */
/* entropy-coded segment                                                     */
/* ------------------------------------------------------------------------- */

/* Strip JPEG byte stuffing and record where each restart marker sat.
 *
 * `out` must hold scan_len + 8 bytes; the eight spare bytes are zeroed so the
 * bit reader's unconditional 64-bit load never runs off the buffer, and reads
 * zeros past the end — exactly what the numba kernel's refill does.
 * `restarts` must hold scan_len / 2 + 1 entries. */
static void destuff(const uint8_t *scan, int64_t scan_len,
                    uint8_t *out, int64_t *out_len,
                    int64_t *restarts, int64_t *restart_count)
{
    int64_t read = 0, written = 0, found = 0;

    while (read < scan_len) {
        const uint8_t *mark = memchr(scan + read, 0xFF, (size_t)(scan_len - read));
        if (mark == NULL) {
            memcpy(out + written, scan + read, (size_t)(scan_len - read));
            written += scan_len - read;
            break;
        }
        int64_t position = mark - scan;
        memcpy(out + written, scan + read, (size_t)(position - read));
        written += position - read;

        /* Past the end reads zero, so a scan ending on a lone 0xFF keeps
         * that byte as data — it has no partner to skip. */
        uint8_t following = position + 1 < scan_len ? scan[position + 1] : 0;
        if (following == 0x00) {
            out[written++] = 0xFF;         /* a literal 0xFF in the data */
            read = position + 2;
        } else if (0xD0 <= following && following <= 0xD7) {
            restarts[found++] = written;
            read = position + 2;
        } else {
            break;                         /* EOI, or the next marker */
        }
    }

    memset(out + written, 0, 8);
    *out_len = written;
    *restart_count = found;
}

/* ------------------------------------------------------------------------- */
/* the scan                                                                  */
/* ------------------------------------------------------------------------- */

/* Decode one Huffman symbol and its mantissa into a signed difference.
 *
 * One unaligned big-endian load gives 57+ usable bits — enough for the
 * longest code (16) plus the widest mantissa (16) in a single look, so there
 * is no per-byte refill loop. Beyond the data the stream reads zero, exactly
 * like the oracle's padding and the numba kernel's refill. */
static inline int64_t next_difference(const uint8_t *data, int64_t data_len,
                                      int64_t *bit, const table_t *table)
{
    int64_t position = *bit;
    int64_t byte = position >> 3;
    uint64_t held = 0;
    if (byte < data_len) {                 /* data has 8 zeroed spare bytes */
        uint64_t raw;
        memcpy(&raw, data + byte, 8);
        held = BSWAP64(raw) << (position & 7);
    }

    int32_t fused = table->first[held >> (64 - FIRST_BITS)];
    if (fused) {                           /* short pair, L1-resident table */
        *bit = position + (fused & 63);
        return fused >> 6;                 /* arithmetic: the sign survives */
    }

    uint32_t window = (uint32_t)(held >> (64 - WINDOW_BITS));
    fused = table->fused[window];
    int32_t consumed = fused & 63;
    if (consumed) {                        /* fused, but longer than FIRST_BITS */
        *bit = position + consumed;
        return fused >> 6;
    }

    /* Code and mantissa are too long to share one window, but never too long
     * for the load: the pair takes at most 32 of the 57 bits held. SSSS = 16
     * cannot arrive here — its code alone always fits, so it always fuses.
     * A window matching no code is the one remaining way to land here, and its
     * entry is zero throughout: these are not the bits this table encoded, so
     * raise the flag and let the caller refuse rather than invent the rest of
     * the image out of a stream it cannot read.
     *
     * The flag is OR-ed unconditionally rather than set behind an `if`. The
     * branch measured 3% on the twelve-bit discs, where this path is taken
     * often enough for a mispredict to matter; the unconditional store to a
     * line already hot costs nothing measurable. */
    uint16_t pair = table->raw[window];
    *table->undefined |= (pair == 0);
    int32_t length = pair >> 8;
    int32_t ssss = pair & 0xFF;
    int64_t difference = 0;
    if (ssss) {
        int64_t span = ((int64_t)1 << ssss) - 1;
        difference = (int64_t)((held << length) >> (64 - ssss));
        if (difference <= (span >> 1))     /* T.81 H.1.2.2, the negative half */
            difference -= span;
    }
    *bit = position + length + ssss;
    return difference;
}

/* T.81 H.1.2.1. The selector was validated before the loop was entered. */
static inline int64_t predicted(int32_t selector, int64_t ra, int64_t rb, int64_t rc)
{
    switch (selector) {
    case 1:  return ra;
    case 2:  return rb;
    case 3:  return rc;
    case 4:  return ra + rb - rc;
    case 5:  return ra + ((rb - rc) >> 1);
    case 6:  return rb + ((ra - rc) >> 1);
    default: return (ra + rb) >> 1;        /* selector 7 */
    }
}

/* Greyscale, with restarts only ever landing on a row boundary (or absent):
 * the shape of every real disc seen so far, and the loop where all the time
 * goes. The specialisations that pay are structural, not clever:
 *
 *   - the first row and the first column are peeled out of the interior
 *     loop, so the interior decides nothing per sample but the difference
 *     itself. In the interior, Ra is the sample just written and Rc is the
 *     Rb of a step ago — both stay in registers, so predicting costs one
 *     load (Rb) instead of three.
 *   - the caller expands this inline function once per predictor through the
 *     switch in `scan_mono`, folding the selector to a constant and the
 *     predictor to straight-line code.
 *   - a restart is handled between rows, not checked per sample. T.81 H.1.2.1
 *     puts the whole first line of an interval on Ra, so a restarted row takes
 *     the initial prediction at column 0 and runs the same Ra loop the image's
 *     first row runs — which is why `restarted` selects the loop and not just
 *     the opening prediction.
 *
 * Everything this loop cannot take (colour, an interval that is not a whole
 * number of rows) falls to the general loop, which decodes anything. */
#define DEFINE_MONO(NAME, SAMPLE)                                              \
static inline int64_t NAME(SAMPLE *out, int32_t width, int32_t height,         \
                           const uint8_t *data, int64_t data_len,              \
                           const int64_t *restarts, int64_t restart_count,     \
                           int32_t rows_per_interval, int32_t selector,        \
                           const table_t *table, int64_t initial, int64_t mask)\
{                                                                              \
    int64_t bit = 0;                                                           \
    int64_t used = 0;                                                          \
    SAMPLE *row_out = out;                                                     \
                                                                               \
    for (int32_t row = 0; row < height; row++) {                               \
        int restarted = 0;                                                     \
        if (rows_per_interval && row && row % rows_per_interval == 0) {        \
            if (used < restart_count)                                          \
                bit = restarts[used++] * 8;                                    \
            restarted = 1;                                                     \
        }                                                                      \
                                                                               \
        int64_t ra, rb, rc;                                                    \
        if (row == 0 || restarted)                                             \
            ra = initial + next_difference(data, data_len, &bit, table);       \
        else                                                                   \
            ra = row_out[-width]                                               \
                 + next_difference(data, data_len, &bit, table);               \
        ra &= mask;                                                            \
        row_out[0] = (SAMPLE)ra;                                               \
                                                                               \
        if (row == 0 || restarted) {           /* T.81 H.1.2.1: Ra line */     \
            for (int32_t col = 1; col < width; col++) {                        \
                ra = (ra + next_difference(data, data_len, &bit, table))       \
                     & mask;                                                   \
                row_out[col] = (SAMPLE)ra;                                     \
            }                                                                  \
        } else {                                                               \
            rc = row_out[-width];                                              \
            for (int32_t col = 1; col < width; col++) {                        \
                int64_t diff = next_difference(data, data_len, &bit, table);   \
                rb = row_out[col - width];                                     \
                ra = (predicted(selector, ra, rb, rc) + diff) & mask;          \
                row_out[col] = (SAMPLE)ra;                                     \
                rc = rb;                                                       \
            }                                                                  \
        }                                                                      \
        row_out += width;                                                      \
    }                                                                          \
    return bit;                                                                \
}

DEFINE_MONO(mono_u8,  uint8_t)
DEFINE_MONO(mono_u16, uint16_t)

/* Expand the selector into each mono kernel so the predictor compiles to
 * straight-line code instead of a per-sample switch. */
#define MONO_DISPATCH(NAME, KERNEL, SAMPLE)                                    \
static int64_t NAME(SAMPLE *out, int32_t width, int32_t height,                \
                    const uint8_t *data, int64_t data_len,                     \
                    const int64_t *restarts, int64_t restart_count,            \
                    int32_t rows_per_interval, int32_t selector,               \
                    const table_t *table, int64_t initial, int64_t mask)       \
{                                                                              \
    switch (selector) {                                                        \
    case 1: return KERNEL(out, width, height, data, data_len, restarts,        \
                          restart_count, rows_per_interval, 1, table,          \
                          initial, mask);                                      \
    case 2: return KERNEL(out, width, height, data, data_len, restarts,        \
                          restart_count, rows_per_interval, 2, table,          \
                          initial, mask);                                      \
    case 3: return KERNEL(out, width, height, data, data_len, restarts,       \
                          restart_count, rows_per_interval, 3, table,          \
                          initial, mask);                                      \
    case 4: return KERNEL(out, width, height, data, data_len, restarts,        \
                          restart_count, rows_per_interval, 4, table,          \
                          initial, mask);                                      \
    case 5: return KERNEL(out, width, height, data, data_len, restarts,        \
                          restart_count, rows_per_interval, 5, table,          \
                          initial, mask);                                      \
    case 6: return KERNEL(out, width, height, data, data_len, restarts,        \
                          restart_count, rows_per_interval, 6, table,          \
                          initial, mask);                                      \
    default: return KERNEL(out, width, height, data, data_len, restarts,       \
                           restart_count, rows_per_interval, 7, table,         \
                           initial, mask);                                     \
    }                                                                          \
}

MONO_DISPATCH(scan_mono_u8,  mono_u8,  uint8_t)
MONO_DISPATCH(scan_mono_u16, mono_u16, uint16_t)

/* Greyscale, predictor Ra, restarts on row boundaries with every marker
 * present: four restart intervals decoded at once, interleaved sample by
 * sample on one core.
 *
 * The sequential loops above are bounded by one serial dependency — the next
 * sample's table window is unknown until this sample's bit count arrives, so
 * every load waits on the one before. Restart intervals cut that chain: each
 * begins at a known byte offset with the predictor reset, and under Ra no
 * sample looks above its own interval's rows, so intervals are decodable
 * independently. Turbo proved the same property to run intervals on many
 * cores (`_scan_intervals`); here the independence is spent on instruction-
 * level parallelism instead — four dependency chains in flight on one core,
 * no threads to spawn or synchronise.
 *
 * Sample for sample this decodes exactly what the sequential loop decodes;
 * the refusal for a truncated scan is if anything stricter, because every
 * interval's end is checked, not just the last one reached. */
#define LANES 4

/* One lane's state lives in scalars, not arrays, and the four-lane body is
 * expanded by macro: an indexed array with a variable loop bound would push
 * every lane's cursor to the stack and put a store-to-load forward on the
 * very dependency chain the interleaving exists to hide. */
#define LANE_INIT(SAMPLE, L)                                                   \
    int64_t segment##L = group + L;                                            \
    int64_t bit##L = segment##L ? restarts[segment##L - 1] * 8 : 0;            \
    SAMPLE *dst##L = out + segment##L * interval;                              \
    int64_t ra##L = 0;                                                         \
    int32_t col##L = 0;

#define LANE_STEP(SAMPLE, L)                                                   \
    {                                                                          \
        int64_t diff = next_difference(data, data_len, &bit##L, table);        \
        int64_t prediction = pixel == 0 ? initial                              \
            : col##L == 0 ? (int64_t)dst##L[-width]                            \
            : ra##L;                                                           \
        ra##L = (prediction + diff) & mask;                                    \
        *dst##L++ = (SAMPLE)ra##L;                                             \
        if (++col##L == width)                                                 \
            col##L = 0;                                                        \
    }

#define MAX2(A, B) ((A) > (B) ? (A) : (B))

/* Four whole intervals, all exactly `interval` pixels long. */
#define DEFINE_INTERLEAVED_GROUP(NAME, SAMPLE)                                 \
static int64_t NAME(SAMPLE *out, int32_t width,                                \
                    const uint8_t *data, int64_t data_len,                     \
                    const int64_t *restarts, int32_t interval, int64_t group,  \
                    const table_t *table, int64_t initial, int64_t mask)       \
{                                                                              \
    LANE_INIT(SAMPLE, 0) LANE_INIT(SAMPLE, 1)                                  \
    LANE_INIT(SAMPLE, 2) LANE_INIT(SAMPLE, 3)                                  \
                                                                               \
    for (int64_t pixel = 0; pixel < interval; pixel++) {                       \
        LANE_STEP(SAMPLE, 0) LANE_STEP(SAMPLE, 1)                              \
        LANE_STEP(SAMPLE, 2) LANE_STEP(SAMPLE, 3)                              \
    }                                                                          \
    return MAX2(MAX2(bit0, bit1), MAX2(bit2, bit3));                           \
}

DEFINE_INTERLEAVED_GROUP(group_u8,  uint8_t)
DEFINE_INTERLEAVED_GROUP(group_u16, uint16_t)

#define DEFINE_INTERLEAVED(NAME, GROUP, SAMPLE)                                \
static int64_t NAME(SAMPLE *out, int32_t width, int64_t npix,                  \
                    const uint8_t *data, int64_t data_len,                     \
                    const int64_t *restarts, int32_t interval,                 \
                    int64_t segments, const table_t *table,                    \
                    int64_t initial, int64_t mask)                             \
{                                                                              \
    int64_t worst_end = 0;                                                     \
    int64_t whole = npix / interval;       /* only the last can fall short */  \
    int64_t group = 0;                                                         \
                                                                               \
    for (; group + LANES <= whole; group += LANES)                             \
        worst_end = MAX2(worst_end,                                            \
                         GROUP(out, width, data, data_len, restarts,           \
                               interval, group, table, initial, mask));        \
                                                                               \
    for (; group < segments; group += LANES) {                                 \
        int lanes = (int)(segments - group < LANES ? segments - group : LANES);\
        int64_t bit[LANES], count[LANES], ra[LANES];                           \
        int32_t col[LANES];                                                    \
        SAMPLE *dst[LANES];                                                    \
        int64_t longest = 0;                                                   \
                                                                               \
        for (int lane = 0; lane < lanes; lane++) {                             \
            int64_t segment = group + lane;                                    \
            bit[lane] = segment ? restarts[segment - 1] * 8 : 0;               \
            int64_t offset = segment * interval;                               \
            dst[lane] = out + offset;                                          \
            count[lane] = npix - offset < interval ? npix - offset : interval; \
            ra[lane] = 0;                                                      \
            col[lane] = 0;                                                     \
            if (count[lane] > longest)                                         \
                longest = count[lane];                                         \
        }                                                                      \
                                                                               \
        for (int64_t pixel = 0; pixel < longest; pixel++) {                    \
            for (int lane = 0; lane < lanes; lane++) {                         \
                if (pixel >= count[lane])                                      \
                    continue;                                                  \
                int64_t diff = next_difference(data, data_len, &bit[lane],     \
                                               table);                        \
                int64_t prediction = pixel == 0 ? initial                      \
                    : col[lane] == 0 ? (int64_t)dst[lane][-width]              \
                    : ra[lane];                                                \
                ra[lane] = (prediction + diff) & mask;                         \
                *dst[lane]++ = (SAMPLE)ra[lane];                               \
                if (++col[lane] == width)                                      \
                    col[lane] = 0;                                             \
            }                                                                  \
        }                                                                      \
                                                                               \
        for (int lane = 0; lane < lanes; lane++)                               \
            if (bit[lane] > worst_end)                                         \
                worst_end = bit[lane];                                         \
    }                                                                          \
    return worst_end;                                                          \
}

DEFINE_INTERLEAVED(scan_interleaved_u8,  group_u8,  uint8_t)
DEFINE_INTERLEAVED(scan_interleaved_u16, group_u16, uint16_t)

/* The general loop: any component count, any restart interval. Components
 * decode interleaved — with 1x1 sampling one MCU is one sample of every scan
 * component, so restart intervals count pixels — and each component predicts
 * from its own earlier samples through the interleaved output layout. */
#define DEFINE_SCAN(NAME, SAMPLE)                                              \
static int64_t NAME(SAMPLE *out, int32_t width, int32_t height, int32_t ncomp, \
                    const uint8_t *data, int64_t data_len,                     \
                    const int64_t *restarts, int64_t restart_count,            \
                    int32_t interval, int32_t selector,                        \
                    const table_t *tables, const int32_t *slot_comp,           \
                    int64_t initial, int64_t mask)                             \
{                                                                              \
    int64_t bit = 0;                                                           \
    int64_t used = 0;                                                          \
    int32_t since = 0;                                                         \
    int64_t index = 0;                                                         \
    int64_t rowstride = (int64_t)width * ncomp;                                \
    /* T.81 H.1.2.1: Ra carries the whole first line of the scan and of every  \
     * restart interval. The caller has refused any interval that does not     \
     * begin on a row boundary, so this can only rise at column zero. */       \
    int ra_line = 1;                                                           \
                                                                               \
    for (int32_t row = 0; row < height; row++) {                               \
        if (row)                                                               \
            ra_line = 0;                                                       \
        for (int32_t col = 0; col < width; col++) {                            \
            int restarted = 0;                                                 \
            if (interval && since == interval) {                               \
                if (used < restart_count)                                      \
                    bit = restarts[used++] * 8;                                \
                since = 0;                                                     \
                restarted = 1;                                                 \
                ra_line = 1;                                                   \
            }                                                                  \
            since++;                                                           \
                                                                               \
            for (int32_t slot = 0; slot < ncomp; slot++) {                     \
                int64_t diff = next_difference(data, data_len, &bit,           \
                                               &tables[slot]);                 \
                int64_t at = index + slot_comp[slot];                          \
                int64_t prediction;                                            \
                if (restarted || (row == 0 && col == 0))                       \
                    prediction = initial;                                      \
                else if (col == 0)                                             \
                    prediction = out[at - rowstride];                          \
                else if (ra_line)                                              \
                    prediction = out[at - ncomp];                              \
                else                                                           \
                    prediction = predicted(selector,                           \
                                           out[at - ncomp],                    \
                                           out[at - rowstride],                \
                                           out[at - rowstride - ncomp]);       \
                out[at] = (SAMPLE)((prediction + diff) & mask);                \
            }                                                                  \
            index += ncomp;                                                    \
        }                                                                      \
    }                                                                          \
    return bit;                                                                \
}

DEFINE_SCAN(scan_u8,  uint8_t)
DEFINE_SCAN(scan_u16, uint16_t)

/* ------------------------------------------------------------------------- */
/* entry points                                                              */
/* ------------------------------------------------------------------------- */

EXPORT int32_t plumbline_abi(void)
{
    return PLUMBLINE_ABI;
}

/* Decode one scan. The caller has already parsed and validated the JPEG
 * headers; every argument here is trusted to be internally consistent
 * (1 <= ncomp, slots within range, 1 <= precision <= 16, selector 1..7).
 *
 *   scan, scan_len     the entropy-coded segment, byte stuffing intact
 *   counts             ntables x 16 code-length counts, concatenated
 *   symbols            all tables' symbols, concatenated
 *   symbol_counts      how many symbols each table owns
 *   slot_tables        scan slot -> table index (ncomp entries)
 *   slot_comps         scan slot -> frame component slot (ncomp entries)
 *   out                height * width * ncomp samples, interleaved
 *   wide               nonzero to write uint16 samples, zero for uint8
 *
 * Returns PLUMBLINE_OK or a negative refusal; never a partial success. */
EXPORT int32_t plumbline_decode(const uint8_t *scan, int64_t scan_len,
                                int32_t width, int32_t height, int32_t ncomp,
                                int32_t precision, int32_t selector,
                                int32_t point_transform, int32_t interval,
                                int32_t ntables,
                                const uint8_t *counts, const uint8_t *symbols,
                                const int32_t *symbol_counts,
                                const int32_t *slot_tables,
                                const int32_t *slot_comps,
                                void *out, int32_t wide)
{
    if (scan_len < 0 || width < 0 || height < 0 || ncomp < 1 || ntables < 1
            || precision < 1 || precision > MAX_PRECISION
            || point_transform < 0 || point_transform >= precision
            || selector < 1 || selector > 7
            || !scan || !counts || !symbols || !symbol_counts
            || !slot_tables || !slot_comps || !out)
        return PLUMBLINE_BAD_ARGS;

    int32_t status = PLUMBLINE_OK;

    /* One arena for everything this call allocates. The destuffed data goes
     * last so every other block keeps malloc's alignment. */
    size_t table_bytes = (size_t)WINDOW_SIZE * (sizeof(int32_t) + sizeof(uint16_t))
                         + (size_t)FIRST_SIZE * sizeof(int32_t);
    size_t data_bytes = (size_t)scan_len + 8;
    size_t restart_bytes = ((size_t)scan_len / 2 + 1) * sizeof(int64_t);
    uint8_t *arena = malloc((size_t)ntables * table_bytes
                            + restart_bytes + data_bytes);
    if (arena == NULL)
        return PLUMBLINE_NO_MEMORY;

    table_t *tables = malloc((size_t)ntables * sizeof(table_t)
                             + (size_t)ncomp * sizeof(table_t));
    if (tables == NULL) {
        free(arena);
        return PLUMBLINE_NO_MEMORY;
    }
    table_t *slots = tables + ntables;

    int32_t undefined = 0;
    uint8_t *cursor = arena;
    int64_t symbol_base = 0;
    for (int32_t entry = 0; entry < ntables; entry++) {
        tables[entry].undefined = &undefined;
        tables[entry].fused = (int32_t *)cursor;
        cursor += WINDOW_SIZE * sizeof(int32_t);
        tables[entry].first = (int32_t *)cursor;
        cursor += FIRST_SIZE * sizeof(int32_t);
        tables[entry].raw = (uint16_t *)cursor;
        cursor += WINDOW_SIZE * sizeof(uint16_t);
        status = build_table(counts + (size_t)entry * 16,
                             symbols + symbol_base, symbol_counts[entry],
                             &tables[entry]);
        if (status != PLUMBLINE_OK)
            goto done;
        symbol_base += symbol_counts[entry];
    }
    for (int32_t slot = 0; slot < ncomp; slot++)
        slots[slot] = tables[slot_tables[slot]];

    int64_t *restarts = (int64_t *)cursor;
    uint8_t *data = cursor + restart_bytes;
    int64_t data_len, restart_count;
    destuff(scan, scan_len, data, &data_len, restarts, &restart_count);

    int64_t initial = (int64_t)1 << (precision - 1 - point_transform);
    /* P - Pt, not P. The entropy-coded data is the image after a right
       shift by Pt, so a sample in it is that many bits wide, and `initial`
       one line above already says so. Masking with the wider value let a
       frame with a wrong Al field return samples the shift then pushed
       past the precision the frame declares. */
    int64_t mask = ((int64_t)1 << (precision - point_transform)) - 1;

    int64_t npix = (int64_t)width * height;
    int64_t segments = interval > 0 ? (npix + interval - 1) / interval : 0;

    /* A declared restart interval needs a marker at the end of every interval
     * but the last. Without this check the scan loops still restarted the
     * predictor at each boundary while carrying on from wherever the bitstream
     * happened to be, and returned PLUMBLINE_OK over pixels that are not the
     * image — which is the one outcome this library exists to prevent.
     *
     * The Python caller rejects such a scan before it gets here, so no shipped
     * path reached this. But `build_table` a hundred lines up defends itself
     * against a caller that skips that validation, and the comment there says
     * the library is safe for anyone who calls it directly. That was true of
     * the tables and not of the scan. It is true of both now. */
    if (segments > 1 && restart_count < segments - 1) {
        status = PLUMBLINE_TRUNCATED;
        goto done;
    }

    /* Interleaving pays when each interval is one row: the four lanes then
     * write within a few kilobytes of each other and share the cache. On the
     * one real disc with 25-row intervals it measured *slower* than the
     * plain loop — four write streams 128 KiB apart — so wider intervals
     * keep the sequential kernel. */
    int64_t end;
    if (ncomp == 1 && selector == 1 && width > 0 && interval == width
            && segments > 1 && restart_count >= segments - 1) {
        end = wide
            ? scan_interleaved_u16((uint16_t *)out, width, npix, data,
                                   data_len, restarts, interval, segments,
                                   &slots[0], initial, mask)
            : scan_interleaved_u8((uint8_t *)out, width, npix, data,
                                  data_len, restarts, interval, segments,
                                  &slots[0], initial, mask);
    } else if (ncomp == 1 && width > 0
            && (interval == 0 || interval % width == 0)) {
        int32_t rows_per_interval = interval ? interval / width : 0;
        end = wide
            ? scan_mono_u16((uint16_t *)out, width, height, data, data_len,
                            restarts, restart_count, rows_per_interval,
                            selector, &slots[0], initial, mask)
            : scan_mono_u8((uint8_t *)out, width, height, data, data_len,
                           restarts, restart_count, rows_per_interval,
                           selector, &slots[0], initial, mask);
    } else {
        end = wide
            ? scan_u16((uint16_t *)out, width, height, ncomp, data, data_len,
                       restarts, restart_count, interval, selector,
                       slots, slot_comps, initial, mask)
            : scan_u8((uint8_t *)out, width, height, ncomp, data, data_len,
                      restarts, restart_count, interval, selector,
                      slots, slot_comps, initial, mask);
    }

    /* A well-formed scan never asks for a bit the frame does not contain.
     * When it does, the image is truncated and every sample after the break
     * is invented, so refuse instead of handing back an image that merely
     * looks decoded. */
    if (end > data_len * 8)
        status = PLUMBLINE_TRUNCATED;
    else if (undefined)
        status = PLUMBLINE_BAD_CODE;

done:
    free(tables);
    free(arena);
    return status;
}
