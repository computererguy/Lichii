// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Vulkan {

constexpr bool CanDiscardPendingDraw(bool native, bool depth_enabled, u32 index_count,
                                    u32 vertex_count, bool skip_native = false) {
    // Native draws can produce persistent resources, even with depth enabled. With skip_native
    // (EDEN_NX2_ASYNC_SKIP, as in Selevet Eglantine) they follow the rule below too: waiting for
    // each new pipeline stalled Ocarina of Time's first minutes in new scenery.
    if (native && !skip_native) {
        return false;
    }
    // Preserve the legacy async-shader heuristic.
    return depth_enabled || (index_count > 6 && vertex_count > 6);
}

} // namespace Vulkan
