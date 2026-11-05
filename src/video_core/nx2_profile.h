// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"
#include "video_core/gpu_split.h"
#include "common/logging.h"
#include "common/cpu_features.h"
#include "common/thread_cpu_clock.h"

// EDEN_NX2_GPU_PROFILE: time where the GPU thread spends its time. Each zone counts its own
// (exclusive) time, nested zones excluded; every 10 s the thread logs each zone's share.
namespace VideoCommon::Nx2Profile {

enum class Zone : u32 {
    Other, ///< GPU thread time outside every zone below (register writes, command parsing)
    Idle,
    PendingSleep,
    ResumePending,
    Push,
    Tick,
    Run,
    FlushInvalidate,
    Macro,
    Draw,
    Clear,
    Compute,
    FlushWork,
    PipelineLookup,
    Configure,
    SyncDescriptors,
    ReadHandles,
    FillImageViews,
    TextureBuffers,
    UpdateBuffers,
    DescriptorAcquire,
    BindStageBuffers,
    PushDescriptors,
    RenderTargets,
    ConfigureDraw,
    DynamicState,
    DrawRecord,
    TextureUpload,
    Fence,        ///< Fences made (SignalFence)
    FenceSkipped, ///< Empty fences not made
    Method,       ///< Engine method calls (what a split GPU thread's back half would run)
    DeferredCall, ///< Split GPU thread: work the front deferred to the back (GpuSplit::Defer)
    SplitDrain,   ///< Split GPU thread: the front waits for the back (gpu_split.h)
    SplitWait,    ///< Split GPU thread: the back waits for work
    WriterWait,
    ListSubmitWait,
    ListGpuWait,
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Zone::Count)> zone_names{
    "other",         "idle",          "pending sleep",  "resume pending", "push",
    "tick",          "run",           "flush/inval",    "macro",          "draw",
    "clear",         "compute",       "flush work",     "pipeline lookup", "configure",
    "sync descriptors", "read handles", "fill image views", "texture buffers", "update buffers",
    "descriptor acquire", "bind stage buffers", "push descriptors", "render targets",
    "configure draw", "dynamic state", "draw record", "texture upload", "fence",
    "fence skipped", "method", "deferred call", "split drain", "split wait",
    "writer wait", "list submit wait", "list GPU wait",
};

// Read once at startup: Enabled() runs several times per method write (no guard to check).
// Off while EDEN_NX2_SAMPLE_PROFILE samples (common/nx2_sample_profile.h): the zone timers would
// show up in the samples.
inline const bool g_enabled = std::getenv("EDEN_NX2_GPU_PROFILE") != nullptr &&
                              std::getenv("EDEN_NX2_SAMPLE_PROFILE") == nullptr;

inline bool Enabled() {
    return g_enabled;
}

// Nanoseconds; from the host's own counter where the wall clock reads it directly (the zones
// run ~150,000 times a second, and QueryPerformanceCounter cost ~5% of the GPU thread).
inline u64 Now() {
    return static_cast<u64>(Common::g_wall_clock.GetTimeNS().count());
}

struct ThreadData {
    std::array<u64, static_cast<size_t>(Zone::Count)> exclusive_ns{};
    std::array<u64, static_cast<size_t>(Zone::Count)> count{};
    std::array<u64, 16> counters{}; ///< See Counter
    std::array<u64, static_cast<size_t>(DrainSite::Count)> drain_ns{};
    std::array<u64, static_cast<size_t>(DrainSite::Count)> drain_count{};
    u64 child_ns{};    ///< Time of the zones nested in the current one
    u64 window_start{};
    std::optional<u64> cpu_start_ns;
};

inline thread_local ThreadData t_data;

/// A finer split of a zone's exclusive time (the split's back: per engine method, per deferred
/// call), logged every 10 s with the zones.
struct LabelSlot {
    u64 ns{};
    u64 count{};
};

struct Labels {
    static constexpr size_t MaxEngines = 8;
    /// In the order the back first ran their methods (the engines have no RTTI to name them).
    std::array<const void*, MaxEngines> engines{};
    std::unique_ptr<std::array<LabelSlot, MaxEngines * 0x10000>> methods;
    std::unordered_map<const void*, LabelSlot> calls;
    LabelSlot sink;
};

inline thread_local Labels t_labels;

/// The label of an engine's method while profiling (nullptr otherwise or past MaxEngines).
template <typename Engine>
inline LabelSlot* MethodLabel(const Engine* engine, u32 method) {
    if (!Enabled()) {
        return nullptr;
    }
    auto& labels = t_labels;
    size_t index = 0;
    for (; index < labels.engines.size(); ++index) {
        if (labels.engines[index] == engine) {
            break;
        }
        if (labels.engines[index] == nullptr) {
            labels.engines[index] = engine;
            break;
        }
    }
    if (index == labels.engines.size()) {
        return nullptr;
    }
    if (!labels.methods) {
        labels.methods = std::make_unique<std::array<LabelSlot, Labels::MaxEngines * 0x10000>>();
    }
    return &(*labels.methods)[index * 0x10000 + (method & 0xffff)];
}

inline LabelSlot* CallLabel(const void* thunk) {
    return &t_labels.calls[thunk];
}

inline LabelSlot* SinkLabel() {
    return &t_labels.sink;
}

class Scope {
public:
    explicit Scope(Zone zone_, LabelSlot* label_ = nullptr)
        : zone{zone_}, label{label_}, enabled{Enabled()} {
        if (!enabled) {
            return;
        }
        parent_child_ns = t_data.child_ns;
        t_data.child_ns = 0;
        start = Now();
    }

    ~Scope() {
        if (!enabled) {
            return;
        }
        const u64 elapsed = Now() - start;
        auto& data = t_data;
        const auto index = static_cast<size_t>(zone);
        data.exclusive_ns[index] += elapsed - data.child_ns;
        ++data.count[index];
        if (label != nullptr) {
            label->ns += elapsed - data.child_ns;
            ++label->count;
        }
        data.child_ns = parent_child_ns + elapsed;
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Zone zone;
    LabelSlot* label;
    bool enabled;
    u64 start{};
    u64 parent_child_ns{};
};

/// Amounts counted on the GPU thread, logged per second with the zones.
enum class Counter : u32 {
    DownloadBytes,  ///< Buffer data copied back from the host GPU at fences
    DownloadCopies, ///< Buffer ranges copied back
    DmaReadBytes,   ///< Guest memory the native MME read with DMA
    DownloadBytesOnDmaPages, ///< Of DownloadBytes: on pages the MME has DMA-read
    MethodWrites,   ///< 3D register writes (Maxwell3D::CallMethod below the macro range)
    SameValueWrites, ///< Of MethodWrites: the register already held the value
    AcquireChecks,  ///< Semaphore acquires evaluated
    AcquireWaits,   ///< Of AcquireChecks: not met yet (the channel waits)
    SunkWrites,     ///< Split back: state writes queued in an engine's method sink
};

/// Device pages (4 KiB) the native MME has DMA-read since start (GPU thread only; filled only
/// while profiling): which buffer downloads anything on the GPU thread reads back.
inline std::unordered_set<u64>& DmaReadPages() {
    static std::unordered_set<u64> pages;
    return pages;
}

inline void Add(Counter counter, u64 amount) {
    if (Enabled()) {
        t_data.counters[static_cast<size_t>(counter)] += amount;
    }
}

/// The front waited for the back since `start` (Now) for `site`.
inline void AddDrain(DrainSite site, u64 start) {
    if (Enabled()) {
        const auto index = static_cast<size_t>(site);
        t_data.drain_ns[index] += Now() - start;
        ++t_data.drain_count[index];
    }
}

/// Logs the top labels of the window (exclusive ms and calls per second) and resets them.
inline void ReportLabels(const char* thread_name, u64 window, u64 sunk_writes) {
    auto& labels = t_labels;
    const auto ms = [window](u64 ns) { return static_cast<double>(ns) * 1000.0 / window; };
    const auto per_second = [window](u64 count) {
        return static_cast<double>(count) * 1e9 / window;
    };
    if (labels.methods) {
        struct Entry {
            size_t index;
            LabelSlot slot;
        };
        std::vector<Entry> entries;
        for (size_t i = 0; i < labels.methods->size(); ++i) {
            const LabelSlot& slot = (*labels.methods)[i];
            if (slot.count != 0) {
                entries.push_back({i, slot});
            }
        }
        std::sort(entries.begin(), entries.end(),
                  [](const Entry& a, const Entry& b) { return a.slot.ns > b.slot.ns; });
        u64 total{};
        for (const auto& entry : entries) {
            total += entry.slot.ns;
        }
        std::string line;
        for (size_t i = 0; i < std::min<size_t>(entries.size(), 24); ++i) {
            const auto& entry = entries[i];
            line += fmt::format(" | engine{} {:#x} {:.1f} ms ({:.0f}/s)", entry.index / 0x10000,
                                entry.index % 0x10000, ms(entry.slot.ns),
                                per_second(entry.slot.count));
        }
        LOG_INFO(HW_GPU,
                 "{} method time per second: {:.1f} ms in {} methods, method sink {:.1f} ms "
                 "({:.0f}/s), state writes sunk {:.0f}/s{}",
                 thread_name, ms(total), entries.size(), ms(labels.sink.ns),
                 per_second(labels.sink.count), per_second(sunk_writes), line);
        labels.methods->fill({});
    }
    if (!labels.calls.empty()) {
        std::vector<std::pair<const void*, LabelSlot>> calls(labels.calls.begin(),
                                                             labels.calls.end());
        std::sort(calls.begin(), calls.end(),
                  [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
        std::string line;
        for (size_t i = 0; i < std::min<size_t>(calls.size(), 12); ++i) {
            line += fmt::format(" | {:.1f} ms ({:.0f}/s) {}", ms(calls[i].second.ns),
                                per_second(calls[i].second.count),
                                GpuSplit::ThunkName(calls[i].first));
        }
        LOG_INFO(HW_GPU, "{} deferred calls per second:{}", thread_name, line);
        labels.calls.clear();
    }
    labels.sink = {};
}

/// Called by the GPU thread (and the split's back thread) between commands: logs and resets
/// the totals every 10 s.
inline void Report(const char* thread_name = "GPU thread") {
    if (!Enabled()) {
        return;
    }
    auto& data = t_data;
    const u64 now = Now();
    if (data.window_start == 0) {
        data.window_start = now;
        data.cpu_start_ns = Common::CurrentThreadCpuTimeNs();
        data.child_ns = 0;
        return;
    }
    const u64 window = now - data.window_start;
    if (window < 10'000'000'000ULL) {
        return;
    }
    const auto cpu_now = Common::CurrentThreadCpuTimeNs();
    // Time in no zone at all is the wall time minus the zones' totals.
    u64 zoned{};
    for (size_t i = 1; i < data.exclusive_ns.size(); ++i) {
        zoned += data.exclusive_ns[i];
    }
    data.exclusive_ns[0] = window > zoned ? window - zoned : 0;
    std::string line;
    for (size_t i = 0; i < data.exclusive_ns.size(); ++i) {
        if (data.exclusive_ns[i] == 0 && data.count[i] == 0) {
            continue;
        }
        // Per second of wall time: milliseconds spent and calls made.
        const double ms = static_cast<double>(data.exclusive_ns[i]) * 1000.0 / window;
        const double calls = static_cast<double>(data.count[i]) * 1e9 / window;
        line += fmt::format(" | {} {:.1f} ms ({:.0f}/s)", zone_names[i], ms, calls);
    }
    line += fmt::format(" | buffer downloads {:.0f} KiB in {:.0f} copies ({:.0f} KiB on pages "
                        "MME DMA reads) | MME DMA reads {:.0f} KiB",
                        static_cast<double>(data.counters[0]) * 1e9 / 1024.0 / window,
                        static_cast<double>(data.counters[1]) * 1e9 / window,
                        static_cast<double>(data.counters[3]) * 1e9 / 1024.0 / window,
                        static_cast<double>(data.counters[2]) * 1e9 / 1024.0 / window);
    line += fmt::format(" | 3D register writes {:.0f} ({:.0f} same value)",
                        static_cast<double>(data.counters[4]) * 1e9 / window,
                        static_cast<double>(data.counters[5]) * 1e9 / window);
    line += fmt::format(" | acquires {:.0f} ({:.0f} waited)",
                        static_cast<double>(data.counters[6]) * 1e9 / window,
                        static_cast<double>(data.counters[7]) * 1e9 / window);
    std::string drains;
    for (size_t i = 0; i < data.drain_count.size(); ++i) {
        if (data.drain_count[i] != 0) {
            drains += fmt::format(" {} {:.1f} ms ({:.0f}/s),", drain_site_names[i],
                                  static_cast<double>(data.drain_ns[i]) * 1000.0 / window,
                                  static_cast<double>(data.drain_count[i]) * 1e9 / window);
        }
    }
    if (!drains.empty()) {
        drains.pop_back();
        line += " | waits for the back by site:" + drains;
    }
    if (cpu_now && data.cpu_start_ns && *cpu_now >= *data.cpu_start_ns) {
        line += fmt::format(" | host_cpu_busy_ms_per_s={:.3f} window_wall_ms={:.3f}",
                            static_cast<double>(*cpu_now - *data.cpu_start_ns) * 1000.0 / window,
                            static_cast<double>(window) / 1e6);
    } else {
        line += fmt::format(" | host_cpu_busy_ms_per_s=unavailable window_wall_ms={:.3f}",
                            static_cast<double>(window) / 1e6);
    }
    LOG_INFO(HW_GPU, "{} per second:{}", thread_name, line);
    ReportLabels(thread_name, window, data.counters[static_cast<size_t>(Counter::SunkWrites)]);
    data = {};
    data.window_start = now;
    data.cpu_start_ns = cpu_now;
}

} // namespace VideoCommon::Nx2Profile
