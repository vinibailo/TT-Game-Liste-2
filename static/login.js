document.addEventListener('DOMContentLoaded', () => {
  const passwordInput = document.getElementById('password');
  const toggleButton = document.querySelector('[data-toggle-password]');
  const submitButton = document.querySelector('[data-login-submit]');

  if (!passwordInput || !submitButton) {
    return;
  }

  const updateSubmitState = () => {
    const hasValue = passwordInput.value.trim().length > 0;
    submitButton.disabled = !hasValue;
  };

  updateSubmitState();
  passwordInput.addEventListener('input', updateSubmitState);

  if (toggleButton) {
    toggleButton.addEventListener('click', (event) => {
      event.preventDefault();
      const isPassword = passwordInput.getAttribute('type') === 'password';
      passwordInput.setAttribute('type', isPassword ? 'text' : 'password');
      toggleButton.setAttribute('aria-pressed', String(isPassword));
      toggleButton.setAttribute('aria-label', isPassword ? 'Hide password' : 'Show password');
      const toggleText = toggleButton.querySelector('.toggle-text');
      if (toggleText) {
        toggleText.textContent = isPassword ? 'Hide' : 'Show';
      }
      passwordInput.focus();
      if (typeof passwordInput.setSelectionRange === 'function') {
        const length = passwordInput.value.length;
        passwordInput.setSelectionRange(length, length);
      }
    });
  }
});
