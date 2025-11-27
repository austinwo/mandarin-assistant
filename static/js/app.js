document.addEventListener("DOMContentLoaded", () => {
  const mainForm   = document.getElementById("main-form");
  const randomForm = document.getElementById("random-form");
  const loading    = document.getElementById("loading-indicator");
  const buttons    = document.querySelectorAll(".submit-btn");
  const results    = document.getElementById("results-area");

  function handleSubmit() {
    // disable buttons
    buttons.forEach(btn => (btn.disabled = true));

    // spinner
    if (loading) {
      loading.classList.remove("d-none");
      loading.setAttribute("aria-busy", "true");
    }

    // hide and clear last result
    if (results) {
      results.classList.add("hidden");
      results.innerHTML = "";
    }
  }

  if (mainForm) mainForm.addEventListener("submit", handleSubmit);
  if (randomForm) randomForm.addEventListener("submit", handleSubmit);
});
