(function(global) {
  const ALERT_WINDOW_MS = 5000;

  function summarizeWindowFireCount(state, now, fireCount) {
    if (!state) state = {};
    const value = Number(fireCount || 0);
    if (value <= 0) return Number(state.alertWindowTotal || 0);

    if (!state.alertWindowStartAt || now - state.alertWindowStartAt >= ALERT_WINDOW_MS) {
      return value;
    }

    return Number(state.alertWindowTotal || 0) + value;
  }

  function shouldReportAlert(state, now, fireCount) {
    if (!state) state = {};
    const value = Number(fireCount || 0);
    if (value <= 0) return false;

    if (!state.alertWindowStartAt || now - state.alertWindowStartAt >= ALERT_WINDOW_MS) {
      state.alertWindowStartAt = now;
      state.alertWindowTotal = 0;
    }

    state.alertWindowTotal = Number(state.alertWindowTotal || 0) + value;

    const lastAlertAt = Number(state.lastAlertAt || 0);
    if (!lastAlertAt || now - lastAlertAt >= ALERT_WINDOW_MS) {
      state.lastAlertAt = now;
      return true;
    }

    return false;
  }

  const api = {
    ALERT_WINDOW_MS,
    summarizeWindowFireCount,
    shouldReportAlert,
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }

  global.AlertPolicy = api;
})(typeof window !== 'undefined' ? window : globalThis);
