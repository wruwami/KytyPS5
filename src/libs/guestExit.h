#ifndef EMULATOR_INCLUDE_EMULATOR_LIBS_GUEST_EXIT_H_
#define EMULATOR_INCLUDE_EMULATOR_LIBS_GUEST_EXIT_H_

#include <cstdlib>

namespace Libs {

// Guest exit() returns while other guest threads are still running. Normal host exit would run
// KytyClose(), which unloads the guest modules, and the host static destructors under them, so
// leave the same way as closing the window: only the at_quick_exit emergency shutdown runs.
[[noreturn]] inline void GuestExit(int status) {
	std::quick_exit(status);
}

} // namespace Libs

#endif /* EMULATOR_INCLUDE_EMULATOR_LIBS_GUEST_EXIT_H_ */
