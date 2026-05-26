document.addEventListener('DOMContentLoaded', () => {
  const filters = document.querySelectorAll('[data-company-filter]');

  filters.forEach((input) => {
    const targetId = input.getAttribute('data-target');
    const target = document.getElementById(targetId);
    if (!target) {
      return;
    }

    input.addEventListener('input', () => {
      const query = String(input.value || '').trim().toLowerCase();
      const rows = target.querySelectorAll('[data-search-text]');
      rows.forEach((row) => {
        const haystack = String(row.getAttribute('data-search-text') || '').toLowerCase();
        row.style.display = !query || haystack.includes(query) ? '' : 'none';
      });
    });
  });
});
