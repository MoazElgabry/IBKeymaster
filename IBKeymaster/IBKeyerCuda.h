#pragma once

#include <string>

#include "IBKeyerBackend.h"

namespace IBKeyerCore {

bool renderCudaInternal(const IBKeyerParams& params, const PackedFrame& frame, std::string& error);
bool renderCudaHost(const IBKeyerParams& params,
                    const DeviceRenderFrame& frame,
                    void* hostCudaStreamOpaque,
                    std::string& error);

} // namespace IBKeyerCore
