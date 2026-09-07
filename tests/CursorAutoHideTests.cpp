#include "graphics/presentation/window/cursorAutoHide.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::CursorAutoHide;

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "CursorAutoHideTests: failed: %s\n", message);
    std::abort();
  }
}

void TestInitialStateAndExplicitHide() {
  int callback_count = 0;
  bool last_visible = true;

  CursorAutoHide cursor(2500, [&](bool visible) {
    callback_count++;
    last_visible = visible;
  });

  Check(!cursor.IsVisible(), "default cursor state should not be visible");
  Check(cursor.GetRemainingTimeoutMs(1000) == -1,
        "hidden cursor should have infinite (-1) timeout");

  cursor.Hide();
  Check(!cursor.IsVisible(), "cursor should still not be visible after Hide()");
  Check(callback_count == 0, "no callback should fire if already hidden");
  Check(last_visible, "last_visible should be unchanged");
}

void TestMouseActivityShowsCursor() {
  int show_count = 0;
  int hide_count = 0;

  CursorAutoHide cursor(2500, [&](bool visible) {
    if (visible) {
      show_count++;
    } else {
      hide_count++;
    }
  });

  cursor.OnButtonOrWheel(1000);
  Check(cursor.IsVisible(),
        "cursor should be visible after button/wheel activity");
  Check(cursor.GetLastActivityMs() == 1000, "last activity timestamp mismatch");
  Check(show_count == 1, "show callback should fire once");
  Check(hide_count == 0, "hide callback should not have fired");

  // Consecutive activity while visible should refresh timestamp without
  // re-triggering show callback
  cursor.OnButtonOrWheel(1500);
  Check(cursor.IsVisible(), "cursor should remain visible");
  Check(cursor.GetLastActivityMs() == 1500,
        "last activity timestamp should be updated");
  Check(show_count == 1,
        "show callback should not fire again when already visible");
}

void TestMotionFiltering() {
  CursorAutoHide cursor(2500);

  // Zero-displacement noise should not trigger cursor visibility
  cursor.OnMotion(0, 0, 1000);
  Check(!cursor.IsVisible(), "zero-displacement motion must not show cursor");

  // Non-zero displacement must show cursor
  cursor.OnMotion(5, 0, 1200);
  Check(cursor.IsVisible(), "positive horizontal motion should show cursor");
  Check(cursor.GetLastActivityMs() == 1200,
        "activity timestamp updated on motion");

  cursor.Hide();
  Check(!cursor.IsVisible(), "cursor hidden explicitly");

  cursor.OnMotion(0, -3, 1400);
  Check(cursor.IsVisible(), "vertical motion should show cursor");
  Check(cursor.GetLastActivityMs() == 1400,
        "activity timestamp updated on vertical motion");
}

void TestIdleAutoHiding() {
  int hide_count = 0;
  CursorAutoHide cursor(2500, [&](bool visible) {
    if (!visible) {
      hide_count++;
    }
  });

  cursor.Show(1000);
  Check(cursor.IsVisible(), "cursor should be visible");

  // Before idle timeout
  cursor.CheckIdle(1000 + 2499);
  Check(cursor.IsVisible(),
        "cursor should stay visible before timeout expires");
  Check(hide_count == 0, "hide callback should not have fired yet");

  // Exactly at or after idle timeout
  cursor.CheckIdle(1000 + 2500);
  Check(!cursor.IsVisible(),
        "cursor should be auto-hidden when idle timeout reached");
  Check(hide_count == 1, "hide callback should have fired");

  // Subsequent idle checks while hidden do nothing
  cursor.CheckIdle(1000 + 5000);
  Check(!cursor.IsVisible(), "cursor stays hidden");
  Check(hide_count == 1, "hide callback should not fire again");
}

void TestRemainingTimeoutCalculation() {
  CursorAutoHide cursor(2500);

  Check(cursor.GetRemainingTimeoutMs(0) == -1,
        "timeout must be -1 when cursor is hidden");

  cursor.Show(1000);
  Check(cursor.GetRemainingTimeoutMs(1000) == 2500,
        "remaining timeout should be full delay immediately");
  Check(cursor.GetRemainingTimeoutMs(2000) == 1500,
        "remaining timeout should decrement accurately");
  Check(cursor.GetRemainingTimeoutMs(3500) == 0,
        "remaining timeout should be 0 when expired");
  Check(cursor.GetRemainingTimeoutMs(4000) == 0,
        "remaining timeout should clamp to 0 past expiry");

  cursor.Hide();
  Check(cursor.GetRemainingTimeoutMs(2000) == -1,
        "timeout must return -1 once hidden");
}

void TestWindowLeaveRestoresCursor() {
  bool visible_flag = false;
  CursorAutoHide cursor(2500, [&](bool visible) { visible_flag = visible; });

  Check(!cursor.IsVisible(), "initially hidden");

  cursor.OnWindowLeave();
  Check(cursor.IsVisible(),
        "cursor should be restored visible on window leave");
  Check(visible_flag, "callback should reflect visible state on window leave");
}

void TestCustomIdleDelay() {
  CursorAutoHide cursor(500);
  Check(cursor.GetIdleDelayMs() == 500, "custom idle delay mismatch");

  cursor.Show(100);
  cursor.CheckIdle(599);
  Check(cursor.IsVisible(), "should be visible before 500ms");

  cursor.CheckIdle(600);
  Check(!cursor.IsVisible(), "should hide at exactly 500ms");
}

} // namespace

int main() {
  TestInitialStateAndExplicitHide();
  TestMouseActivityShowsCursor();
  TestMotionFiltering();
  TestIdleAutoHiding();
  TestRemainingTimeoutCalculation();
  TestWindowLeaveRestoresCursor();
  TestCustomIdleDelay();

  std::printf("CursorAutoHideTests: all cases passed\n");
  return 0;
}
