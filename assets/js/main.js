document.addEventListener('DOMContentLoaded', () => {
  const year = document.querySelector('#year');
  if (year) year.textContent = new Date().getFullYear();

  document.querySelectorAll('.nav-toggle').forEach((button) => {
    const navbar = button.closest('.navbar');
    if (!navbar) return;
    button.addEventListener('click', () => {
      const isOpen = navbar.classList.toggle('nav-open');
      button.setAttribute('aria-expanded', String(isOpen));
    });
  });

  const escapeHtml = (value) => String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');


  const translationNode = document.querySelector('#site-translations');
  let siteTranslations = {};
  try { siteTranslations = translationNode ? JSON.parse(translationNode.textContent || '{}') : {}; }
  catch (err) { console.warn('[i18n] Could not parse translations:', err); }

  function getCurrentLanguage() {
    return localStorage.getItem('site-language') || document.documentElement.lang || 'en';
  }

  function localizedValue(item, field) {
    const lang = getCurrentLanguage();
    return lang === 'es' ? (item[`${field}_es`] || item[field] || '') : (item[field] || '');
  }

  function applyLanguage(lang) {
    const normalized = lang === 'es' ? 'es' : 'en';
    localStorage.setItem('site-language', normalized);
    document.documentElement.lang = normalized;
    document.querySelectorAll('[data-i18n]').forEach((el) => {
      const value = siteTranslations[el.dataset.i18n]?.[normalized];
      if (typeof value === 'string') el.textContent = value;
    });
    document.querySelectorAll('.lang-button').forEach((button) => {
      const active = button.dataset.lang === normalized;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    document.title = normalized === 'es' ? 'Luis González Ramírez | Astrofísica' : 'Luis González Ramírez | Astrophysics';
  }

  document.querySelectorAll('.lang-button').forEach((button) => {
    button.addEventListener('click', () => {
      applyLanguage(button.dataset.lang);
      renderAstroGallery();
    });
  });

  function ensureImageModal() {
    let modal = document.querySelector('#image-modal');
    if (modal) return modal;
    modal = document.createElement('div');
    modal.id = 'image-modal';
    modal.className = 'image-modal';
    modal.innerHTML = `
      <div class="image-modal-card" role="dialog" aria-modal="true" aria-label="Astrophotography image preview">
        <div class="image-modal-header">
          <h3 class="image-modal-title" id="image-modal-title"></h3>
          <button class="image-modal-close" type="button" aria-label="Close image preview">×</button>
        </div>
        <img id="image-modal-img" alt="" />
        <div class="image-modal-footer">
          <span class="capture" id="image-modal-caption"></span>
          <a class="button secondary compact" id="image-modal-open" href="#" target="_blank" rel="noopener">Open full image →</a>
        </div>
      </div>`;
    document.body.appendChild(modal);
    const close = () => modal.classList.remove('open');
    modal.querySelector('.image-modal-close').addEventListener('click', close);
    modal.addEventListener('click', (event) => {
      if (event.target === modal) close();
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') close();
    });
    return modal;
  }

  const gallery = document.querySelector('#astro-gallery');
  function renderAstroGallery() {
    if (!gallery || !Array.isArray(window.ASTRO_GALLERY)) return;
    gallery.innerHTML = window.ASTRO_GALLERY.map((item, index) => `
      <article class="gallery-card">
        <a class="gallery-image-link" href="${escapeHtml(item.image)}" data-gallery-index="${index}" aria-label="Open ${escapeHtml(localizedValue(item, 'title'))} preview">
          <img
            src="${escapeHtml(item.thumb || item.image)}"
            alt="${escapeHtml(localizedValue(item, 'title'))}"
            loading="lazy"
            decoding="async"
            onerror="this.onerror=null; this.src=this.closest('.gallery-image-link').href;"
          />
        </a>
        <div class="gallery-card-body">
          <p class="badge">${escapeHtml(localizedValue(item, 'type'))}</p>
          <h3>${escapeHtml(localizedValue(item, 'title'))}</h3>
          <p class="capture">${escapeHtml(localizedValue(item, 'capture'))}</p>
          <p>${escapeHtml(localizedValue(item, 'description'))}</p>
          <a class="text-link" href="${escapeHtml(item.image)}" target="_blank" rel="noopener">${getCurrentLanguage() === 'es' ? 'Abrir imagen completa →' : 'Open full image →'}</a>
        </div>
      </article>
    `).join('');

    gallery.querySelectorAll('.gallery-image-link').forEach((link) => {
      link.addEventListener('click', (event) => {
        event.preventDefault();
        const item = window.ASTRO_GALLERY[Number(link.dataset.galleryIndex)];
        if (!item) return;
        const modal = ensureImageModal();
        modal.querySelector('#image-modal-title').textContent = localizedValue(item, 'title');
        modal.querySelector('#image-modal-caption').textContent = localizedValue(item, 'capture') || '';
        const img = modal.querySelector('#image-modal-img');
        img.src = item.image;
        img.alt = localizedValue(item, 'title');
        const openLink = modal.querySelector('#image-modal-open');
        openLink.href = item.image;
        openLink.textContent = getCurrentLanguage() === 'es' ? 'Abrir imagen completa →' : 'Open full image →';
        modal.classList.add('open');
      });
    });
  }
  applyLanguage(getCurrentLanguage());
  renderAstroGallery();

  const cvCanvas = document.querySelector('#cv-canvas');
  const cvOpen = document.querySelector('#cv-open-link');
  const cvLabel = document.querySelector('#cv-current-label');
  const cvFallback = document.querySelector('#cv-fallback');
  const cvPageLabel = document.querySelector('#cv-page-label');
  const cvPrev = document.querySelector('#cv-prev');
  const cvNext = document.querySelector('#cv-next');
  let cvPdf = null;
  let cvPage = 1;
  let cvRendering = false;
  let cvPendingPage = null;
  let cvUrl = document.querySelector('.cv-tab.active')?.dataset.cv || 'assets/pdf/CV_Luis_Gonzalez_Ramirez_EN.pdf';

  function updateCvControls() {
    const total = cvPdf ? cvPdf.numPages : '…';
    if (cvPageLabel) cvPageLabel.textContent = `Page ${cvPage} / ${total}`;
    if (cvPrev) cvPrev.disabled = cvPage <= 1 || !cvPdf;
    if (cvNext) cvNext.disabled = !cvPdf || cvPage >= cvPdf.numPages;
  }

  async function renderCvPage(pageNumber) {
    if (!cvPdf || !cvCanvas) return;
    if (cvRendering) {
      cvPendingPage = pageNumber;
      return;
    }
    cvRendering = true;
    try {
      const page = await cvPdf.getPage(pageNumber);
      const shell = cvCanvas.closest('.cv-canvas-shell');
      const available = shell ? Math.max(420, shell.clientWidth - 36) : 900;
      const baseViewport = page.getViewport({ scale: 1 });
      const scale = Math.min(1.55, available / baseViewport.width);
      const viewport = page.getViewport({ scale });
      const context = cvCanvas.getContext('2d');
      cvCanvas.width = Math.floor(viewport.width);
      cvCanvas.height = Math.floor(viewport.height);
      await page.render({ canvasContext: context, viewport }).promise;
      cvPage = pageNumber;
      updateCvControls();
    } catch (err) {
      console.error('[cv] PDF render failed:', err);
      if (cvFallback) cvFallback.hidden = false;
    } finally {
      cvRendering = false;
      if (cvPendingPage !== null) {
        const next = cvPendingPage;
        cvPendingPage = null;
        renderCvPage(next);
      }
    }
  }

  async function loadCv(url) {
    cvUrl = url;
    cvPage = 1;
    updateCvControls();
    if (!cvCanvas) return;
    if (!window.pdfjsLib) {
      if (cvFallback) cvFallback.hidden = false;
      return;
    }
    try {
      pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
      cvPdf = await pdfjsLib.getDocument(url).promise;
      if (cvFallback) cvFallback.hidden = true;
      await renderCvPage(1);
    } catch (err) {
      console.error('[cv] PDF load failed:', err);
      if (cvFallback) cvFallback.hidden = false;
    }
  }

  document.querySelectorAll('.cv-tab').forEach((button) => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.cv-tab').forEach((b) => b.classList.remove('active'));
      button.classList.add('active');
      const url = button.dataset.cv;
      const label = button.dataset.label || button.textContent.trim();
      if (cvOpen) cvOpen.setAttribute('href', url);
      if (cvLabel) cvLabel.textContent = label;
      loadCv(url);
    });
  });
  if (cvPrev) cvPrev.addEventListener('click', () => { if (cvPdf && cvPage > 1) renderCvPage(cvPage - 1); });
  if (cvNext) cvNext.addEventListener('click', () => { if (cvPdf && cvPage < cvPdf.numPages) renderCvPage(cvPage + 1); });

  const cvSection = document.querySelector('#cv');
  let cvHasLoaded = false;
  function ensureCvLoaded() {
    if (!cvCanvas || cvHasLoaded) return;
    cvHasLoaded = true;
    loadCv(cvUrl);
  }
  if (cvCanvas) {
    if (window.location.hash === '#cv') {
      window.setTimeout(ensureCvLoaded, 120);
    } else if ('IntersectionObserver' in window && cvSection) {
      const cvObserver = new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          ensureCvLoaded();
          cvObserver.disconnect();
        }
      }, { rootMargin: '450px 0px' });
      cvObserver.observe(cvSection);
    } else {
      ensureCvLoaded();
    }
  }
  document.querySelectorAll('a[href="#cv"]').forEach((link) => {
    link.addEventListener('click', () => window.setTimeout(ensureCvLoaded, 80));
  });
  document.addEventListener('keydown', (event) => {
    if (!cvPdf || !cvSection || !document.body.classList.contains('home-page')) return;
    const active = document.activeElement;
    if (active && ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'A'].includes(active.tagName)) return;
    const rect = cvSection.getBoundingClientRect();
    const vh = window.innerHeight || document.documentElement.clientHeight || 800;
    const visible = Math.min(rect.bottom, vh) - Math.max(rect.top, 0);
    if (visible < Math.min(220, rect.height * 0.20)) return;
    if (event.key === 'ArrowLeft' && cvPage > 1) { event.preventDefault(); renderCvPage(cvPage - 1); }
    if (event.key === 'ArrowRight' && cvPage < cvPdf.numPages) { event.preventDefault(); renderCvPage(cvPage + 1); }
  });

  if (document.body.classList.contains('home-page')) {
    const revealTargets = Array.from(document.querySelectorAll('main > section, .publication-card, .link-card, .card, .gallery-card'));
    const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    revealTargets.forEach((el) => el.classList.add('reveal-on-scroll'));

    if (reduceMotion || !('IntersectionObserver' in window)) {
      revealTargets.forEach((el) => el.classList.add('is-visible'));
    } else {
      const revealObserver = new IntersectionObserver((entries) => {
        window.requestAnimationFrame(() => {
          entries.forEach((entry) => {
            entry.target.classList.toggle('is-visible', entry.isIntersecting);
          });
        });
      }, {
        rootMargin: '12% 0px 12% 0px',
        threshold: 0.06
      });
      revealTargets.forEach((el) => revealObserver.observe(el));
    }
  }
});
