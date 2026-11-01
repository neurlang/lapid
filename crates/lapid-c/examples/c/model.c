/* Load a .lapid file, inspect it, and run an embedding lookup + linear layer.
 *
 *   cc model.c -I include -L lib -llapid -o model && ./model tiny.lapid [device]
 *
 * Expects tensors "embed" [V, D] and "w" [N, D] (f32) and a blob "notes.txt". */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lapid.h"

#define CHECK(call)                                                             \
    do {                                                                        \
        lapid_status s_ = (call);                                               \
        if (s_ != LAPID_OK) {                                                   \
            fprintf(stderr, "%s failed: %s %s\n", #call, lapid_status_name(s_), \
                    lapid_last_error());                                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static unsigned char *read_all(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc((size_t)n);
    if (buf && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.lapid [device]\n", argv[0]);
        return 2;
    }
    CHECK(lapid_check_abi(LAPID_ABI_VERSION));
    lapid_device *dev = NULL;
    CHECK(lapid_device_open(argc > 2 ? argv[2] : "cpu", &dev));

    /* From a path (memory mapped). */
    lapid_file *file = NULL;
    CHECK(lapid_file_open(argv[1], &file));
    char *json = NULL;
    CHECK(lapid_file_metadata_json(file, &json));
    printf("metadata: %s\n", json);
    lapid_free(json);
    CHECK(lapid_file_tensors_json(file, &json));
    printf("tensors: %s\n", json);
    lapid_free(json);

    const uint8_t *blob = NULL;
    size_t blob_len = 0;
    CHECK(lapid_file_blob(file, "notes.txt", &blob, &blob_len));
    printf("blob notes.txt: %.*s\n", (int)blob_len, (const char *)blob);

    lapid_tensor *missing = NULL;
    lapid_status s = lapid_file_tensor(file, "nope", &missing);
    printf("missing tensor: %s %s\n", lapid_status_name(s), lapid_last_error());
    int ok = s == LAPID_ERR_NOT_FOUND;

    /* Same file from memory (copied, so the buffer can be freed right away). */
    size_t len = 0;
    unsigned char *bytes = read_all(argv[1], &len);
    if (!bytes) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    lapid_file *mem = NULL;
    CHECK(lapid_file_open_bytes(bytes, len, &mem));
    free(bytes);

    lapid_tensor *embed = NULL, *w = NULL, *ids = NULL, *x = NULL, *y = NULL;
    CHECK(lapid_file_tensor(file, "embed", &embed));
    CHECK(lapid_file_tensor(mem, "w", &w));
    lapid_file_free(file); /* tensors keep the file image alive */
    lapid_file_free(mem);

    const int32_t id_data[] = {2, 0};
    const size_t id_shape[] = {2};
    CHECK(lapid_tensor_from_data(id_data, id_shape, 1, LAPID_DTYPE_I32, &ids));
    CHECK(lapid_gather(embed, ids, &x)); /* [2, D] */
    CHECK(lapid_matmul_t(x, w, &y));     /* [2, N] */

    size_t n = 0;
    CHECK(lapid_tensor_numel(y, &n));
    float *out = malloc(n * sizeof(float));
    CHECK(lapid_to_host_f32(dev, y, out, n));

    /* Reference on the host. */
    size_t es[2], ws[2];
    CHECK(lapid_tensor_shape(embed, es, 2, NULL));
    CHECK(lapid_tensor_shape(w, ws, 2, NULL));
    float *ev = malloc(es[0] * es[1] * sizeof(float));
    float *wv = malloc(ws[0] * ws[1] * sizeof(float));
    CHECK(lapid_to_host_f32(dev, embed, ev, es[0] * es[1]));
    CHECK(lapid_to_host_f32(dev, w, wv, ws[0] * ws[1]));
    printf("y =");
    for (size_t r = 0; r < 2; r++) {
        for (size_t j = 0; j < ws[0]; j++) {
            float ref = 0;
            for (size_t k = 0; k < es[1]; k++) ref += ev[id_data[r] * es[1] + k] * wv[j * ws[1] + k];
            float got = out[r * ws[0] + j];
            printf(" %g", got);
            ok = ok && (got - ref < 1e-4f && ref - got < 1e-4f);
        }
    }
    printf("\n");

    free(out);
    free(ev);
    free(wv);
    lapid_tensor_free(embed);
    lapid_tensor_free(w);
    lapid_tensor_free(ids);
    lapid_tensor_free(x);
    lapid_tensor_free(y);
    lapid_device_free(dev);
    printf(ok ? "model: OK\n" : "model: WRONG RESULT\n");
    return ok ? 0 : 1;
}
