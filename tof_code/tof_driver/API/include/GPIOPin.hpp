#pragma once
#include <memory>
namespace sbc { class GPIOPin { public: using SharedPtr = std::shared_ptr<GPIOPin>; }; }
using GPIOPin = sbc::GPIOPin;