#pragma once
#include <memory>
#include <string>
namespace sbc { 
    class I2CBus { 
    public: 
        using SharedPtr = std::shared_ptr<I2CBus>;
        static SharedPtr makeShared(std::string dev) { return nullptr; }
    }; 
}
using I2CBus = sbc::I2CBus;