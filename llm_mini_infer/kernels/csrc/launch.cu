// The library's launch settings (common.cuh): programmatic dependent launch on or off for every launcher that goes
// through `lmk_launch`. A runner sets it before it captures its CUDA graphs. A graph keeps the launches it captured.
#include "common.cuh"

LMK_API int lmk_set_pdl(int on) {
  lmk_pdl_flag() = on ? 1 : 0;
  return 0;
}

LMK_API int lmk_get_pdl() { return lmk_pdl_flag(); }
