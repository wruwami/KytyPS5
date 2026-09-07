#ifndef KYTY_GRAPHICS_PRESENTATION_WINDOW_CURSOR_AUTO_HIDE_H_
#define KYTY_GRAPHICS_PRESENTATION_WINDOW_CURSOR_AUTO_HIDE_H_

#include <cstdint>
#include <functional>
#include <utility>

namespace Libs::Graphics {

constexpr uint64_t DEFAULT_CURSOR_AUTO_HIDE_DELAY_MS = 2500;

class CursorAutoHide {
public:
	using ShowCursorFn = std::function<void(bool visible)>;

	explicit CursorAutoHide(uint64_t     idle_delay_ms  = DEFAULT_CURSOR_AUTO_HIDE_DELAY_MS,
	                        ShowCursorFn show_cursor_fn = nullptr)
	    : m_idle_delay_ms(idle_delay_ms), m_show_cursor_fn(std::move(show_cursor_fn)) {}

	void SetShowCursorCallback(ShowCursorFn fn) { m_show_cursor_fn = std::move(fn); }

	[[nodiscard]] bool IsVisible() const noexcept { return m_visible; }

	[[nodiscard]] uint64_t GetLastActivityMs() const noexcept { return m_last_activity_ms; }

	[[nodiscard]] uint64_t GetIdleDelayMs() const noexcept { return m_idle_delay_ms; }

	void Hide() {
		if (m_visible) {
			m_visible = false;
			if (m_show_cursor_fn) {
				m_show_cursor_fn(false);
			}
		}
	}

	void Show(uint64_t now_ms) {
		m_last_activity_ms = now_ms;
		if (!m_visible) {
			m_visible = true;
			if (m_show_cursor_fn) {
				m_show_cursor_fn(true);
			}
		}
	}

	void OnMotion(int xrel, int yrel, uint64_t now_ms) {
		if (xrel != 0 || yrel != 0) {
			Show(now_ms);
		}
	}

	void OnButtonOrWheel(uint64_t now_ms) { Show(now_ms); }

	void OnWindowLeave() {
		if (!m_visible) {
			m_visible = true;
			if (m_show_cursor_fn) {
				m_show_cursor_fn(true);
			}
		}
	}

	void CheckIdle(uint64_t now_ms) {
		if (m_visible && (now_ms - m_last_activity_ms >= m_idle_delay_ms)) {
			Hide();
		}
	}

	[[nodiscard]] int GetRemainingTimeoutMs(uint64_t now_ms) const noexcept {
		if (!m_visible) {
			return -1;
		}
		const uint64_t elapsed = now_ms - m_last_activity_ms;
		if (elapsed >= m_idle_delay_ms) {
			return 0;
		}
		return static_cast<int>(m_idle_delay_ms - elapsed);
	}

private:
	uint64_t     m_idle_delay_ms    = DEFAULT_CURSOR_AUTO_HIDE_DELAY_MS;
	bool         m_visible          = false;
	uint64_t     m_last_activity_ms = 0;
	ShowCursorFn m_show_cursor_fn;
};

} // namespace Libs::Graphics

#endif // KYTY_GRAPHICS_PRESENTATION_WINDOW_CURSOR_AUTO_HIDE_H_
