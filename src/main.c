#include "renderer/window.h"

int main() 
{ 
    createWindow();
    return 0;
}
 
// # 1. From the project root (where CMakeLists.txt is unless builfd folder is specified), create a build folder and change into it
// mkdir -p build && cd build

// # 2. Configure
// cmake .. -DCMAKE_BUILD_TYPE=Release

// Only the first time you build the project with the build folder
// # 3. Build
// cmake --build . --config Release 

// # 4. Run
// ./SaylaAI  