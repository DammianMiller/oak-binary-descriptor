# One frame of keypoints with binary descriptors, published on /oak/features.
# keypoints are sorted ascending by zorder (entries without a code go last,
# ordered by x).

std_msgs/Header header

# Which 512->64 bit compression strategy produced the zorder codes.
# Consumers must not compare codes across different compression values.
uint8 COMPRESSION_SUBSAMPLE=0
uint8 COMPRESSION_XORFOLD=1
uint8 COMPRESSION_LSH=2
uint8 COMPRESSION_ITQ=3
uint8 compression

uint16 desc_bits     # width of the full descriptor in bits (e.g. 512),
                     # 0 when full descriptors are not published

Keypoint[] keypoints
