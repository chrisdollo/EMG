/* ─── Data ───────────────────────────────────────────────────────────────── */
const losoData = [
  { subj: 'Subj 03', acc: 98.21 },
  { subj: 'Subj 04', acc: 71.07 },
  { subj: 'Subj 05', acc: 67.14 },
  { subj: 'Subj 06', acc: 53.93 },
  { subj: 'Subj 07', acc: 63.57 },
  { subj: 'Subj 08', acc: 73.21 },
  { subj: 'Subj 09', acc: 82.50 },
  { subj: 'Subj 10', acc: 88.93 },
  { subj: 'Subj 11', acc: 73.57 },
  { subj: 'Subj 12', acc: 92.50 },
  { subj: 'Subj 13', acc: 79.29 },
  { subj: 'Subj 14', acc: 99.29 },
  { subj: 'Subj 15', acc: 85.00 },
  { subj: 'Subj 16', acc: 62.50 },
  { subj: 'Subj 17', acc: 65.36 },
  { subj: 'Subj 18', acc: 71.07 },
  { subj: 'Subj 19', acc: 70.36 },
  { subj: 'Subj 20', acc: 76.79 },
  { subj: 'Subj 22', acc: 78.21 },
  { subj: 'Subj 23', acc: 86.07 },
  { subj: 'Subj 24', acc: 95.00 },
  { subj: 'Subj 25', acc: 66.07 },
  { subj: 'Subj 26', acc: 56.07 },
  { subj: 'Subj 27', acc: 97.50 },
  { subj: 'Subj 29', acc: 93.57 },
  { subj: 'Subj 30', acc: 76.79 },
  { subj: 'Subj 31', acc: 68.93 },
  { subj: 'Subj 33', acc: 93.21 },
  { subj: 'Subj 34', acc: 72.86 },
  { subj: 'Subj 35', acc: 96.79 },
  { subj: 'Subj 36', acc: 88.21 },
  { subj: 'Subj 38', acc: 86.43 },
  { subj: 'Subj 39', acc: 75.71 },
  { subj: 'Subj 42', acc: 81.43 },
  { subj: 'Subj 43', acc: 84.64 },
  { subj: 'Subj 45', acc: 84.29 },
  { subj: 'Subj 46', acc: 89.29 },
  { subj: 'Subj 47', acc: 61.07 },
  { subj: 'Subj 48', acc: 99.29 },
  { subj: 'Subj 49', acc: 93.93 },
  { subj: 'Subj 50', acc: 94.29 },
  { subj: 'Subj 51', acc: 77.14 },
  { subj: 'Subj 53', acc: 91.79 },
  { subj: 'Subj 54', acc: 90.71 },
];

/* ─── Bar chart ──────────────────────────────────────────────────────────── */
function buildBarChart() {
  const chart = document.getElementById('barChart');
  if (!chart) return;
  losoData.forEach(d => {
    const c = d.acc >= 86 ? 'var(--green)' : d.acc >= 71 ? 'var(--accent)' : 'var(--red)';
    chart.innerHTML += `
      <div class="bar-row">
        <span class="subj">${d.subj}</span>
        <div class="bar-track">
          <div class="bar-fill" style="width:${d.acc}%;background:${c}"></div>
        </div>
        <span class="pct" style="color:${c}">${d.acc.toFixed(1)}%</span>
      </div>`;
  });
}

/* ─── Navigation & transitions ───────────────────────────────────────────── */
const DURATION = 300; // ms
let slides, total, cur, isTransitioning;

function updateUI() {
  document.getElementById('counter').textContent = `${cur + 1} / ${total}`;
  document.getElementById('prev').disabled = cur === 0;
  document.getElementById('next').disabled = cur === total - 1;
  document.getElementById('progress').style.width = `${((cur + 1) / total) * 100}%`;
}

function go(dir) {
  if (isTransitioning) return;
  const next = Math.max(0, Math.min(total - 1, cur + dir));
  if (next === cur) return;

  isTransitioning = true;
  const leaving  = slides[cur];
  const entering = slides[next];

  // Reveal entering slide off-screen (no transition yet)
  entering.style.cssText = [
    'display:flex',
    `opacity:0`,
    `transform:translateX(${dir > 0 ? 56 : -56}px)`,
    'transition:none',
  ].join(';');

  // Force reflow so the browser registers the initial state
  entering.offsetWidth; // eslint-disable-line no-unused-expressions

  // Apply transitions to both slides
  const t = `opacity ${DURATION}ms ease, transform ${DURATION}ms ease`;
  leaving.style.transition  = t;
  entering.style.transition = t;

  // Animate
  leaving.style.opacity   = '0';
  leaving.style.transform = `translateX(${dir > 0 ? -56 : 56}px)`;
  entering.style.opacity  = '1';
  entering.style.transform = 'translateX(0)';

  cur = next;
  updateUI();

  setTimeout(() => {
    // Hide the old slide
    leaving.removeAttribute('style');
    leaving.classList.remove('active');

    // Clean up entering slide — let CSS take over
    entering.removeAttribute('style');
    entering.classList.add('active');

    isTransitioning = false;
  }, DURATION + 20);
}

/* ─── Keyboard navigation ────────────────────────────────────────────────── */
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === ' ') {
    e.preventDefault();
    go(1);
  }
  if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
    e.preventDefault();
    go(-1);
  }
});

/* ─── Init ───────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  slides         = Array.from(document.querySelectorAll('.slide'));
  total          = slides.length;
  cur            = 0;
  isTransitioning = false;

  buildBarChart();
  updateUI();
});
