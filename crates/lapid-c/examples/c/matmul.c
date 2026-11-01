/* Build a small graph (matmul, softmax, topk) and read the results back.
 *
 *   cc matmul.c -I include -L lib -llapid -o matmul && ./matmul [device]
 *
 * device: "cpu" (default), "gpu", "cuda:0", "vulkan", "metal", ... */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

static void on_log(lapid_log_level level, const char *msg, void *user) {
    (void)user;
    fprintf(stderr, "[lapid %d] %s\n", level, msg);
}

int main(int argc, char **argv) {
    CHECK(lapid_check_abi(LAPID_ABI_VERSION));
    lapid_set_log_callback(on_log, NULL);
    printf("lapid %s (abi %u)\n", lapid_version(), lapid_abi_version());

    char *devices = NULL;
    CHECK(lapid_devices_json(&devices));
    printf("devices: %s\n", devices);
    lapid_free(devices);

    lapid_device *dev = NULL;
    CHECK(lapid_device_open(argc > 1 ? argv[1] : "cpu", &dev));

    /* a [2, 3] @ b [3, 2] */
    const float a_data[] = {1, 2, 3, 4, 5, 6};
    const float b_data[] = {1, 0, 0, 1, 1, 1};
    const size_t a_shape[] = {2, 3}, b_shape[] = {3, 2};
    lapid_tensor *a = NULL, *b = NULL, *c = NULL, *p = NULL, *vals = NULL, *idx = NULL;
    CHECK(lapid_tensor_from_data(a_data, a_shape, 2, LAPID_DTYPE_F32, &a));
    CHECK(lapid_tensor_from_data(b_data, b_shape, 2, LAPID_DTYPE_F32, &b));
    CHECK(lapid_matmul(a, b, &c));
    CHECK(lapid_softmax(c, &p));
    CHECK(lapid_topk(p, 1, -1, &vals, &idx));
    /* Inputs can be released right away: the graph keeps what it needs. */
    lapid_tensor_free(a);
    lapid_tensor_free(b);

    size_t shape[4], rank = 0;
    CHECK(lapid_tensor_shape(c, shape, 4, &rank));
    printf("c shape: [%zu, %zu]\n", shape[0], shape[1]);

    /* Evaluate both outputs in one graph, then copy to host. */
    const lapid_tensor *outs[2] = {c, idx};
    lapid_tensor *res[2] = {NULL, NULL};
    CHECK(lapid_eval(dev, outs, 2, res));

    float cv[4];
    int32_t iv[2];
    CHECK(lapid_to_host_f32(dev, res[0], cv, 4));
    CHECK(lapid_to_host_i32(dev, res[1], iv, 2));
    printf("c = [[%g, %g], [%g, %g]]\n", cv[0], cv[1], cv[2], cv[3]);
    printf("argmax = [%d, %d]\n", iv[0], iv[1]);

    const float want[4] = {4, 5, 10, 11};
    int ok = iv[0] == 1 && iv[1] == 1;
    for (int i = 0; i < 4; i++) ok = ok && fabsf(cv[i] - want[i]) < 1e-5f;

    /* Errors: a bad reshape is reported with a status and JSON details. */
    lapid_tensor *bad = NULL;
    const size_t bad_shape[] = {3};
    lapid_status s = lapid_reshape(c, bad_shape, 1, &bad);
    printf("expected error: %s %s\n", lapid_status_name(s), lapid_last_error());
    ok = ok && s == LAPID_ERR_SHAPE_MISMATCH && bad == NULL;

    for (int i = 0; i < 2; i++) lapid_tensor_free(res[i]);
    lapid_tensor_free(c);
    lapid_tensor_free(p);
    lapid_tensor_free(vals);
    lapid_tensor_free(idx);
    lapid_device_free(dev);
    printf(ok ? "matmul: OK\n" : "matmul: WRONG RESULT\n");
    return ok ? 0 : 1;
}
