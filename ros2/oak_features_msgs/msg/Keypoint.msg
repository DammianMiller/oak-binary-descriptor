# One keypoint with optional full binary descriptor and optional 64-bit
# compressed descriptor code (zorder). The zorder is a compression of the
# *descriptor* (not the coordinates): it is designed to stay stable as the
# keypoint moves across pixels, enabling XOR+popcount re-identification.

float32 x            # pixel coordinate
float32 y            # pixel coordinate
int32 track_id       # FeatureTracker ID, stable across consecutive frames
int32 age            # tracking age in frames

uint8[] desc         # optional full binary descriptor (e.g. 64 bytes = 512 bits);
                     # empty when absent (patch missed this frame's crop budget,
                     # or full descriptors disabled)
bool has_zorder      # true when zorder carries a valid code
uint64 zorder        # 64-bit compressed descriptor code (strategy in header
                     # of the enclosing KeypointArray)
