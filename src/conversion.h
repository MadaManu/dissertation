struct __attribute__((packed)) f8 {
    uint8_t data[1];
};
struct __attribute__((packed)) f16 {
    uint8_t data[2];
};
struct __attribute__((packed)) f24 {
    uint8_t data[3];
};
struct __attribute__((packed)) f40 {
    uint8_t data[5];
};
struct __attribute__((packed)) f48 {
    uint8_t data[6];
};
struct __attribute__((packed)) f56 {
    uint8_t data[7];
};
static inline void round_float_t_f8(float_t * r, struct f8 * m)
{
    uint32_t temp;
    uint32_t stick;
    uint32_t round;
    uint32_t guard;
    uint32_t preguard;
    uint32_t new_mantissa;
    uint32_t new_signexp;
    memcpy((void *) &temp, (void *) r, 4);
    stick = (temp & 16777215) > 0;
    round = (temp & 8388608) >> 23;
    guard = (temp & 16777216) >> 24;
    preguard = (temp & 33554432) >> 25;
    new_mantissa = (temp & 8388607) >> 24;
    new_signexp = temp & 4286578688;
    new_mantissa = new_mantissa + (guard & (round | stick));
    new_mantissa = new_mantissa + (preguard & (guard & ~(round | stick)));
    temp = (new_mantissa << 24) + new_signexp;
    uint8_t * cast_ptr;
    cast_ptr = (uint8_t *) &temp;
    memcpy((void *) &(*m).data[0], (void *) (cast_ptr + 3), 1);
}
static inline void round_float_t_f16(float_t * r, struct f16 * m)
{
    uint32_t temp;
    uint32_t stick;
    uint32_t round;
    uint32_t guard;
    uint32_t preguard;
    uint32_t new_mantissa;
    uint32_t new_signexp;
    memcpy((void *) &temp, (void *) r, 4);
    stick = (temp & 65535) > 0;
    round = (temp & 32768) >> 15;
    guard = (temp & 65536) >> 16;
    preguard = (temp & 131072) >> 17;
    new_mantissa = (temp & 8388607) >> 16;
    new_signexp = temp & 4286578688;
    new_mantissa = new_mantissa + (guard & (round | stick));
    new_mantissa = new_mantissa + (preguard & (guard & ~(round | stick)));
    temp = (new_mantissa << 16) + new_signexp;
    uint8_t * cast_ptr;
    cast_ptr = (uint8_t *) &temp;
    memcpy((void *) &(*m).data[0], (void *) (cast_ptr + 2), 2);
}
static inline void round_float_t_f24(float_t * r, struct f24 * m)
{
    uint32_t temp;
    uint32_t stick;
    uint32_t round;
    uint32_t guard;
    uint32_t preguard;
    uint32_t new_mantissa;
    uint32_t new_signexp;
    memcpy((void *) &temp, (void *) r, 4);
    stick = (temp & 255) > 0;
    round = (temp & 128) >> 7;
    guard = (temp & 256) >> 8;
    preguard = (temp & 512) >> 9;
    new_mantissa = (temp & 8388607) >> 8;
    new_signexp = temp & 4286578688;
    new_mantissa = new_mantissa + (guard & (round | stick));
    new_mantissa = new_mantissa + (preguard & (guard & ~(round | stick)));
    temp = (new_mantissa << 8) + new_signexp;
    uint8_t * cast_ptr;
    cast_ptr = (uint8_t *) &temp;
    memcpy((void *) &(*m).data[0], (void *) (cast_ptr + 1), 3);
}
static inline void round_double_t_f40(double_t * r, struct f40 * m)
{
    uint64_t temp;
    uint64_t stick;
    uint64_t round;
    uint64_t guard;
    uint64_t preguard;
    uint64_t new_mantissa;
    uint64_t new_signexp;
    memcpy((void *) &temp, (void *) r, 8);
    stick = (temp & 16777215) > 0;
    round = (temp & 8388608) >> 23;
    guard = (temp & 16777216) >> 24;
    preguard = (temp & 33554432) >> 25;
    new_mantissa = (temp & 4503599627370495) >> 24;
    new_signexp = temp & 18442240474082181120u;
    new_mantissa = new_mantissa + (guard & (round | stick));
    new_mantissa = new_mantissa + (preguard & (guard & ~(round | stick)));
    temp = (new_mantissa << 24) + new_signexp;
    uint8_t * cast_ptr;
    cast_ptr = (uint8_t *) &temp;
    memcpy((void *) &(*m).data[0], (void *) (cast_ptr + 3), 5);
}
static inline void round_double_t_f48(double_t * r, struct f48 * m)
{
    uint64_t temp;
    uint64_t stick;
    uint64_t round;
    uint64_t guard;
    uint64_t preguard;
    uint64_t new_mantissa;
    uint64_t new_signexp;
    memcpy((void *) &temp, (void *) r, 8);
    stick = (temp & 65535) > 0;
    round = (temp & 32768) >> 15;
    guard = (temp & 65536) >> 16;
    preguard = (temp & 131072) >> 17;
    new_mantissa = (temp & 4503599627370495) >> 16;
    new_signexp = temp & 18442240474082181120u;
    new_mantissa = new_mantissa + (guard & (round | stick));
    new_mantissa = new_mantissa + (preguard & (guard & ~(round | stick)));
    temp = (new_mantissa << 16) + new_signexp;
    uint8_t * cast_ptr;
    cast_ptr = (uint8_t *) &temp;
    memcpy((void *) &(*m).data[0], (void *) (cast_ptr + 2), 6);
}
static inline void round_double_t_f56(double_t * r, struct f56 * m)
{
    uint64_t temp;
    uint64_t stick;
    uint64_t round;
    uint64_t guard;
    uint64_t preguard;
    uint64_t new_mantissa;
    uint64_t new_signexp;
    memcpy((void *) &temp, (void *) r, 8);
    stick = (temp & 255) > 0;
    round = (temp & 128) >> 7;
    guard = (temp & 256) >> 8;
    preguard = (temp & 512) >> 9;
    new_mantissa = (temp & 4503599627370495) >> 8;
    new_signexp = temp & 18442240474082181120u;
    new_mantissa = new_mantissa + (guard & (round | stick));
    new_mantissa = new_mantissa + (preguard & (guard & ~(round | stick)));
    temp = (new_mantissa << 8) + new_signexp;
    uint8_t * cast_ptr;
    cast_ptr = (uint8_t *) &temp;
    memcpy((void *) &(*m).data[0], (void *) (cast_ptr + 1), 7);
}
static inline void round_f8_float_t(struct f8 * r, float_t * m)
{
    uint32_t temp;
    temp = 0;
    memcpy((void *) &temp, (void *) r, 1);
    temp = temp << 24;
    memcpy((void *) m, (void *) &temp, 4);
}
static inline void round_f16_float_t(struct f16 * r, float_t * m)
{
    uint32_t temp;
    temp = 0;
    memcpy((void *) &temp, (void *) r, 2);
    temp = temp << 16;
    memcpy((void *) m, (void *) &temp, 4);
}
static inline void round_f24_float_t(struct f24 * r, float_t * m)
{
    uint32_t temp;
    temp = 0;
    memcpy((void *) &temp, (void *) r, 3);
    temp = temp << 8;
    memcpy((void *) m, (void *) &temp, 4);
}
static inline void round_f40_double_t(struct f40 * r, double_t * m)
{
    uint64_t temp;
    temp = 0;
    memcpy((void *) &temp, (void *) r, 5);
    temp = temp << 24;
    memcpy((void *) m, (void *) &temp, 8);
}
static inline void round_f48_double_t(struct f48 * r, double_t * m)
{
    uint64_t temp;
    temp = 0;
    memcpy((void *) &temp, (void *) r, 6);
    temp = temp << 16;
    memcpy((void *) m, (void *) &temp, 8);
}
static inline void round_f56_double_t(struct f56 * r, double_t * m)
{
    uint64_t temp;
    temp = 0;
    memcpy((void *) &temp, (void *) r, 7);
    temp = temp << 8;
    memcpy((void *) m, (void *) &temp, 8);
}
