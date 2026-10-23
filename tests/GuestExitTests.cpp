#include "common/common.h"
#include "libs/guestExit.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <limits.h>
#include <mach-o/dyld.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

// Exit status of a child whose host atexit teardown ran.
constexpr int HOST_TEARDOWN_STATUS = 97;

const char *g_marker_path = nullptr;
std::atomic<bool> g_worker_running{false};

void Check(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "GuestExitTests: failed: %s\n", message);
    std::exit(1);
  }
}

void WriteMarker(const char *text) {
  if (FILE *file = std::fopen(g_marker_path, "w"); file != nullptr) {
    std::fputs(text, file);
    std::fclose(file);
  }
}

// Stands in for KytyClose(): it must not run while guest threads are live.
void OnHostAtexit() {
  WriteMarker("atexit");
  std::_Exit(HOST_TEARDOWN_STATUS);
}

// Stands in for Subsystems::EmergencyShutdownActive().
void OnQuickExit() { WriteMarker("quick_exit"); }

[[noreturn]] void RunChild(const char *thread, int status, const char *marker) {
  g_marker_path = marker;
  Check(std::atexit(OnHostAtexit) == 0, "atexit failed");
  Check(std::at_quick_exit(OnQuickExit) == 0, "at_quick_exit failed");

  // Another guest thread keeps running while exit() is called.
  std::thread worker([] {
    g_worker_running = true;
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  while (!g_worker_running) {
    std::this_thread::yield();
  }

  if (std::strcmp(thread, "worker") == 0) {
    std::thread exiting([status] { Libs::GuestExit(status); });
    exiting.join();
  } else {
    Libs::GuestExit(status);
  }
  std::_Exit(0x7e);
}

int RunChildProcess(const char *thread, int status, const std::string &marker) {
  const std::string status_text = std::to_string(status);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
  char path[MAX_PATH]{};
  Check(GetModuleFileNameA(nullptr, path, MAX_PATH) != 0,
        "GetModuleFileName failed");
  std::string command = std::string("\"") + path + "\" --child " + thread +
                        " " + status_text + " \"" + marker + "\"";
  std::vector<char> mutable_command(command.begin(), command.end());
  mutable_command.push_back('\0');
  STARTUPINFOA startup{sizeof(startup)};
  PROCESS_INFORMATION process{};
  Check(CreateProcessA(nullptr, mutable_command.data(), nullptr, nullptr, FALSE,
                       CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                       &process) != 0,
        "CreateProcess failed");
  Check(WaitForSingleObject(process.hProcess, 10000) == WAIT_OBJECT_0,
        "child timed out");
  DWORD exit_code = 0;
  Check(GetExitCodeProcess(process.hProcess, &exit_code) != 0,
        "GetExitCodeProcess failed");
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
  return static_cast<int>(exit_code);
#else
#if defined(__APPLE__)
  std::vector<char> path(PATH_MAX);
  uint32_t path_size = static_cast<uint32_t>(path.size());
  if (_NSGetExecutablePath(path.data(), &path_size) != 0) {
    path.resize(path_size);
    Check(_NSGetExecutablePath(path.data(), &path_size) == 0,
          "_NSGetExecutablePath failed");
  }
#endif
  const pid_t pid = ::fork();
  Check(pid >= 0, "fork failed");
  if (pid == 0) {
#if defined(__APPLE__)
    ::execl(path.data(), "GuestExitTests", "--child", thread,
            status_text.c_str(), marker.c_str(), nullptr);
#else
    ::execl("/proc/self/exe", "GuestExitTests", "--child", thread,
            status_text.c_str(), marker.c_str(), nullptr);
#endif
    std::_Exit(0x7e);
  }
  int wait_status = 0;
  Check(::waitpid(pid, &wait_status, 0) == pid, "waitpid failed");
  Check(WIFEXITED(wait_status), "child did not exit normally");
  return WEXITSTATUS(wait_status);
#endif
}

void TestGuestExit(const char *thread, int status) {
  const auto marker = std::filesystem::temp_directory_path() /
                      ("kyty guest exit " + std::string(thread) + " " +
                       std::to_string(status) + ".txt");
  std::filesystem::remove(marker);

  const int exit_code = RunChildProcess(thread, status, marker.string());

  std::string text;
  if (std::ifstream file(marker); file) {
    text.assign(std::istreambuf_iterator<char>(file), {});
  }
  std::filesystem::remove(marker);

  if (exit_code != status || text != "quick_exit") {
    std::fprintf(stderr,
                 "GuestExitTests: exit(%d) from the %s thread: status %d, "
                 "marker \"%s\"\n",
                 status, thread, exit_code, text.c_str());
  }
  Check(text != "atexit", "guest exit ran the host atexit teardown");
  Check(text == "quick_exit", "guest exit skipped the emergency shutdown");
  Check(exit_code == status, "guest exit lost the exit status");
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 5 && std::strcmp(argv[1], "--child") == 0) {
    RunChild(argv[2], std::atoi(argv[3]), argv[4]);
  }
  for (const char *thread : {"main", "worker"}) {
    for (int status : {0, 37, 255}) {
      TestGuestExit(thread, status);
    }
  }
  std::puts("GuestExitTests: all cases passed");
  return 0;
}
