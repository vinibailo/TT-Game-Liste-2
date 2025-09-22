(function () {
    const config = window.updatesConfig || {};
    const state = {
        updates: [],
        filtered: [],
        sortKey: 'refreshed_at',
        sortDir: 'desc',
        searchTerm: '',
        detailCache: new Map(),
        updateMap: new Map(),
    };

    const elements = {
        tableBody: document.querySelector('[data-updates-body]'),
        emptyState: document.querySelector('[data-empty-state]'),
        loadingState: document.querySelector('[data-loading-state]'),
        searchInput: document.querySelector('[data-search]'),
        refreshButton: document.querySelector('[data-refresh]'),
        countLabel: document.querySelector('[data-count]'),
        statusLabel: document.querySelector('[data-refresh-status]'),
        sortButtons: Array.from(document.querySelectorAll('.sort-button[data-sort]')),
    };

    const modal = {
        backdrop: document.querySelector('[data-modal]'),
        closeButton: document.querySelector('[data-close-modal]'),
        subtitle: document.querySelector('[data-modal-subtitle]'),
        gameId: document.querySelector('[data-modal-game-id]'),
        igdbId: document.querySelector('[data-modal-igdb-id]'),
        igdbUpdated: document.querySelector('[data-modal-igdb-updated]'),
        localEdited: document.querySelector('[data-modal-local-edited]'),
        empty: document.querySelector('[data-modal-empty]'),
        diffList: document.querySelector('[data-diff-list]'),
    };

    const modalEmptyDefaultText = modal.empty ? modal.empty.textContent : '';
    const toast = document.getElementById('toast');
    let toastTimer = null;
    let lastFocusedElement = null;

    function showToast(message, type = 'success') {
        if (!toast) {
            return;
        }
        const normalized = type === 'error' ? 'warning' : type;
        toast.textContent = message;
        toast.className = '';
        void toast.offsetWidth;
        toast.classList.add(normalized, 'show');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => {
            toast.classList.remove('show');
        }, 3200);
    }

    function parseDate(value) {
        if (!value) {
            return null;
        }
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) {
            return null;
        }
        return date;
    }

    function formatDate(value) {
        if (!value) {
            return '—';
        }
        const date = parseDate(value);
        if (!date) {
            return value;
        }
        return new Intl.DateTimeFormat(undefined, {
            dateStyle: 'medium',
            timeStyle: 'short',
        }).format(date);
    }

    function buildEditUrl(gameId) {
        const base = config.editBaseUrl || '/';
        try {
            const url = new URL(base, window.location.origin);
            url.searchParams.set('game_id', gameId);
            return url.pathname + url.search + url.hash;
        } catch (err) {
            const separator = base.includes('?') ? '&' : '?';
            return `${base}${separator}game_id=${encodeURIComponent(gameId)}`;
        }
    }

    function setLoading(loading) {
        if (!elements.loadingState || !elements.tableBody) {
            return;
        }
        if (loading) {
            elements.loadingState.hidden = false;
            elements.tableBody.innerHTML = '';
            if (elements.emptyState) {
                elements.emptyState.hidden = true;
            }
        } else {
            elements.loadingState.hidden = true;
        }
    }

    function updateCount() {
        if (!elements.countLabel) {
            return;
        }
        elements.countLabel.textContent = String(state.filtered.length);
    }

    function updateStatusMessage() {
        if (!elements.statusLabel) {
            return;
        }
        if (state.searchTerm) {
            const count = state.filtered.length;
            const plural = count === 1 ? '' : 's';
            elements.statusLabel.textContent = `Showing ${count} result${plural} for “${state.searchTerm}”.`;
            return;
        }
        const latest = state.updates.reduce((acc, item) => {
            const date = parseDate(item.refreshed_at || item.igdb_updated_at || item.local_last_edited_at);
            if (!date) {
                return acc;
            }
            if (!acc) {
                return date;
            }
            return date > acc ? date : acc;
        }, null);
        if (latest) {
            elements.statusLabel.textContent = `Last refreshed ${formatDate(latest.toISOString())}`;
        } else {
            elements.statusLabel.textContent = '';
        }
    }

    function compareValues(a, b, key) {
        switch (key) {
            case 'name': {
                const first = (a.name || '').toLocaleLowerCase();
                const second = (b.name || '').toLocaleLowerCase();
                if (first < second) return -1;
                if (first > second) return 1;
                return 0;
            }
            case 'processed_game_id': {
                const first = Number.parseInt(a.processed_game_id, 10) || 0;
                const second = Number.parseInt(b.processed_game_id, 10) || 0;
                return first - second;
            }
            case 'refreshed_at':
            default: {
                const first = parseDate(a.refreshed_at) || parseDate(a.igdb_updated_at) || parseDate(a.local_last_edited_at);
                const second = parseDate(b.refreshed_at) || parseDate(b.igdb_updated_at) || parseDate(b.local_last_edited_at);
                if (!first && !second) return 0;
                if (!first) return -1;
                if (!second) return 1;
                return first.getTime() - second.getTime();
            }
        }
    }

    function applyFilters() {
        const term = state.searchTerm.trim().toLocaleLowerCase();
        const filtered = term
            ? state.updates.filter((item) => {
                  const name = (item.name || '').toLocaleLowerCase();
                  const id = String(item.processed_game_id || '');
                  const igdbId = String(item.igdb_id || '');
                  return name.includes(term) || id.includes(term) || igdbId.includes(term);
              })
            : state.updates.slice();

        filtered.sort((a, b) => {
            const comparison = compareValues(a, b, state.sortKey);
            return state.sortDir === 'asc' ? comparison : -comparison;
        });

        state.filtered = filtered;
        updateCount();
        renderTable();
        updateStatusMessage();
    }

    function createIcon(content, className) {
        const span = document.createElement('span');
        span.className = className;
        span.setAttribute('aria-hidden', 'true');
        span.textContent = content;
        return span;
    }

    function renderTable() {
        if (!elements.tableBody || !elements.emptyState) {
            return;
        }
        elements.tableBody.innerHTML = '';
        if (!state.filtered.length) {
            elements.emptyState.hidden = false;
            return;
        }
        elements.emptyState.hidden = true;
        const fragment = document.createDocumentFragment();
        state.filtered.forEach((item) => {
            const row = document.createElement('tr');
            row.classList.toggle('has-diff', Boolean(item.has_diff));

            const refreshedCell = document.createElement('td');
            refreshedCell.textContent = formatDate(item.refreshed_at || item.igdb_updated_at);
            row.appendChild(refreshedCell);

            const nameCell = document.createElement('td');
            nameCell.className = 'cell-primary';
            nameCell.textContent = item.name || 'Unnamed game';
            row.appendChild(nameCell);

            const idCell = document.createElement('td');
            idCell.textContent = item.processed_game_id ? String(item.processed_game_id) : '—';
            row.appendChild(idCell);

            const igdbCell = document.createElement('td');
            igdbCell.textContent = item.igdb_id ? String(item.igdb_id) : '—';
            row.appendChild(igdbCell);

            const igdbUpdatedCell = document.createElement('td');
            igdbUpdatedCell.textContent = formatDate(item.igdb_updated_at);
            row.appendChild(igdbUpdatedCell);

            const localEditedCell = document.createElement('td');
            localEditedCell.textContent = formatDate(item.local_last_edited_at);
            row.appendChild(localEditedCell);

            const actionsCell = document.createElement('td');
            actionsCell.className = 'actions-cell';
            const editLink = document.createElement('a');
            editLink.href = buildEditUrl(item.processed_game_id);
            editLink.className = 'icon-button';
            editLink.title = 'Edit game';
            editLink.appendChild(createIcon('edit', 'material-symbols-rounded'));
            const editLabel = document.createElement('span');
            editLabel.className = 'sr-only';
            editLabel.textContent = `Edit ${item.name || 'game'}`;
            editLink.appendChild(editLabel);
            actionsCell.appendChild(editLink);

            const viewButton = document.createElement('button');
            viewButton.type = 'button';
            viewButton.className = 'icon-button';
            viewButton.title = 'View changes';
            viewButton.dataset.updateId = String(item.processed_game_id);
            viewButton.appendChild(createIcon('difference', 'material-symbols-rounded'));
            const viewLabel = document.createElement('span');
            viewLabel.className = 'sr-only';
            viewLabel.textContent = `View changes for ${item.name || 'game'}`;
            viewButton.appendChild(viewLabel);
            viewButton.addEventListener('click', () => openDiffModal(item.processed_game_id));
            actionsCell.appendChild(viewButton);

            row.appendChild(actionsCell);
            fragment.appendChild(row);
        });
        elements.tableBody.appendChild(fragment);
    }

    function updateSortIndicators() {
        elements.sortButtons.forEach((button) => {
            const key = button.dataset.sort;
            const indicator = button.querySelector('.sort-indicator');
            const header = button.closest('th');
            if (state.sortKey === key) {
                button.classList.add('is-active');
                if (indicator) {
                    indicator.textContent = state.sortDir === 'asc' ? 'arrow_upward' : 'arrow_downward';
                }
                if (header) {
                    header.setAttribute('aria-sort', state.sortDir === 'asc' ? 'ascending' : 'descending');
                }
            } else {
                button.classList.remove('is-active');
                if (indicator) {
                    indicator.textContent = 'unfold_more';
                }
                if (header) {
                    header.setAttribute('aria-sort', 'none');
                }
            }
        });
    }

    function handleSort(event) {
        const button = event.currentTarget;
        if (!button) {
            return;
        }
        const key = button.dataset.sort;
        if (!key) {
            return;
        }
        if (state.sortKey === key) {
            state.sortDir = state.sortDir === 'asc' ? 'desc' : 'asc';
        } else {
            state.sortKey = key;
            state.sortDir = key === 'name' ? 'asc' : 'desc';
        }
        updateSortIndicators();
        applyFilters();
    }

    function attachSortHandlers() {
        elements.sortButtons.forEach((button) => {
            button.addEventListener('click', handleSort);
        });
    }

    function updateSearchTerm(event) {
        state.searchTerm = event.target.value || '';
        applyFilters();
    }

    function setRefreshButtonLoading(loading) {
        const button = elements.refreshButton;
        if (!button) {
            return;
        }
        const label = button.querySelector('.btn-label');
        if (loading) {
            button.disabled = true;
            if (label) {
                button.dataset.originalLabel = label.textContent;
                label.textContent = 'Updating…';
            }
        } else {
            button.disabled = false;
            if (label && button.dataset.originalLabel) {
                label.textContent = button.dataset.originalLabel;
            }
        }
    }

    function buildDetailUrl(id) {
        if (config.detailUrlTemplate) {
            return config.detailUrlTemplate.replace('{id}', encodeURIComponent(id));
        }
        return `/api/updates/${encodeURIComponent(id)}`;
    }

    async function fetchDiffDetail(id) {
        if (state.detailCache.has(id)) {
            return state.detailCache.get(id);
        }
        const response = await fetch(buildDetailUrl(id));
        let payload = null;
        try {
            payload = await response.json();
        } catch (err) {
            payload = null;
        }
        if (!response.ok || !payload || payload.error) {
            const errorMessage = payload && payload.error ? payload.error : 'Failed to load update details.';
            throw new Error(errorMessage);
        }
        state.detailCache.set(id, payload);
        return payload;
    }

    function showModalShell() {
        if (!modal.backdrop) {
            return;
        }
        lastFocusedElement = document.activeElement instanceof HTMLElement ? document.activeElement : null;
        modal.backdrop.hidden = false;
        modal.backdrop.classList.remove('hidden');
        document.body.classList.add('modal-open');
        if (modal.closeButton) {
            modal.closeButton.focus();
        }
    }

    function closeModal() {
        if (!modal.backdrop) {
            return;
        }
        modal.backdrop.classList.add('hidden');
        modal.backdrop.hidden = true;
        document.body.classList.remove('modal-open');
        if (modal.diffList) {
            modal.diffList.innerHTML = '';
        }
        if (modal.empty) {
            modal.empty.textContent = modalEmptyDefaultText;
            modal.empty.hidden = true;
        }
        if (modal.subtitle) {
            modal.subtitle.textContent = '';
        }
        if (lastFocusedElement && document.body.contains(lastFocusedElement)) {
            lastFocusedElement.focus();
        }
        lastFocusedElement = null;
    }

    function setMetaValue(element, value) {
        if (!element) {
            return;
        }
        element.textContent = value ? String(value) : '—';
    }

    function renderDiff(diff) {
        if (!modal.diffList || !modal.empty) {
            return;
        }
        modal.diffList.innerHTML = '';
        modal.empty.textContent = modalEmptyDefaultText;
        const entries = Object.entries(diff || {});
        if (!entries.length) {
            modal.empty.hidden = false;
            return;
        }
        modal.empty.hidden = true;
        entries.sort((a, b) => a[0].localeCompare(b[0]));
        const fragment = document.createDocumentFragment();
        entries.forEach(([field, payload]) => {
            const section = document.createElement('section');
            section.className = 'diff-field';
            const heading = document.createElement('h3');
            heading.textContent = field;
            section.appendChild(heading);
            const columns = document.createElement('div');
            columns.className = 'diff-columns';
            columns.appendChild(buildDiffColumn('IGDB', payload.added, 'diff-added'));
            columns.appendChild(buildDiffColumn('Local', payload.removed, 'diff-removed'));
            section.appendChild(columns);
            fragment.appendChild(section);
        });
        modal.diffList.appendChild(fragment);
    }

    function buildDiffColumn(label, value, modifier) {
        const column = document.createElement('div');
        column.className = 'diff-column';
        const title = document.createElement('h4');
        title.textContent = label;
        column.appendChild(title);
        if (Array.isArray(value) && value.length) {
            const list = document.createElement('ul');
            list.className = 'diff-pills';
            value.forEach((item) => {
                const pill = document.createElement('li');
                pill.className = `diff-pill ${modifier}`;
                pill.textContent = item;
                list.appendChild(pill);
            });
            column.appendChild(list);
        } else if (value) {
            const paragraph = document.createElement('p');
            paragraph.className = `diff-value ${modifier}`;
            paragraph.textContent = value;
            column.appendChild(paragraph);
        } else {
            const empty = document.createElement('p');
            empty.className = 'diff-value diff-empty';
            empty.textContent = '—';
            column.appendChild(empty);
        }
        return column;
    }

    async function openDiffModal(id) {
        const update = state.updateMap.get(id);
        if (!update) {
            showToast('Unable to find update details for that game.', 'warning');
            return;
        }
        showModalShell();
        if (modal.empty) {
            modal.empty.textContent = 'Loading changes…';
            modal.empty.hidden = false;
        }
        if (modal.diffList) {
            modal.diffList.innerHTML = '';
        }
        setMetaValue(modal.gameId, update.processed_game_id);
        setMetaValue(modal.igdbId, update.igdb_id);
        setMetaValue(modal.igdbUpdated, formatDate(update.igdb_updated_at));
        setMetaValue(modal.localEdited, formatDate(update.local_last_edited_at));
        if (modal.subtitle) {
            modal.subtitle.textContent = update.name ? `IGDB diff for ${update.name}` : '';
        }
        try {
            const detail = await fetchDiffDetail(id);
            setMetaValue(modal.gameId, detail.processed_game_id);
            setMetaValue(modal.igdbId, detail.igdb_id);
            setMetaValue(modal.igdbUpdated, formatDate(detail.igdb_updated_at));
            setMetaValue(modal.localEdited, formatDate(detail.local_last_edited_at));
            if (modal.subtitle) {
                const refreshed = formatDate(detail.refreshed_at);
                modal.subtitle.textContent = detail.name
                    ? `${detail.name} • Refreshed ${refreshed}`
                    : `Refreshed ${refreshed}`;
            }
            renderDiff(detail.diff);
        } catch (error) {
            console.error(error);
            if (modal.empty) {
                modal.empty.textContent = 'Failed to load changes for this update.';
                modal.empty.hidden = false;
            }
            showToast(error.message, 'warning');
        }
    }

    async function handleRefresh() {
        if (!config.refreshUrl) {
            await loadUpdates();
            return;
        }
        setRefreshButtonLoading(true);
        try {
            const response = await fetch(config.refreshUrl, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
            });
            let payload = null;
            try {
                payload = await response.json();
            } catch (err) {
                payload = null;
            }
            if (!response.ok || !payload || payload.error) {
                const errorMessage = payload && payload.error ? payload.error : 'Failed to refresh updates.';
                throw new Error(errorMessage);
            }
            state.detailCache.clear();
            const updated = Number(payload.updated) || 0;
            const missingCount = Array.isArray(payload.missing) ? payload.missing.length : 0;
            let statusMessage = `Fetched ${updated} update${updated === 1 ? '' : 's'}.`;
            if (missingCount) {
                statusMessage += ` ${missingCount} IGDB record${missingCount === 1 ? '' : 's'} missing.`;
            }
            if (elements.statusLabel) {
                elements.statusLabel.textContent = statusMessage;
            }
            showToast(statusMessage, 'success');
            await loadUpdates();
        } catch (error) {
            console.error(error);
            showToast(error.message, 'warning');
        } finally {
            setRefreshButtonLoading(false);
        }
    }

    async function loadUpdates() {
        setLoading(true);
        try {
            const response = await fetch(config.updatesUrl || '/api/updates');
            let payload = null;
            try {
                payload = await response.json();
            } catch (err) {
                payload = null;
            }
            if (!response.ok || !payload || !Array.isArray(payload.updates)) {
                throw new Error(payload && payload.error ? payload.error : 'Failed to load updates.');
            }
            state.updates = payload.updates;
            state.updateMap = new Map(state.updates.map((item) => [item.processed_game_id, item]));
            applyFilters();
            updateSortIndicators();
        } catch (error) {
            console.error(error);
            showToast(error.message, 'warning');
            state.updates = [];
            state.filtered = [];
            if (elements.tableBody) {
                elements.tableBody.innerHTML = '';
            }
            if (elements.emptyState) {
                elements.emptyState.hidden = false;
            }
            updateCount();
            updateStatusMessage();
        } finally {
            setLoading(false);
        }
    }

    function bindEvents() {
        if (elements.searchInput) {
            elements.searchInput.addEventListener('input', updateSearchTerm);
        }
        if (elements.refreshButton) {
            elements.refreshButton.addEventListener('click', handleRefresh);
        }
        if (modal.closeButton) {
            modal.closeButton.addEventListener('click', closeModal);
        }
        if (modal.backdrop) {
            modal.backdrop.addEventListener('click', (event) => {
                if (event.target === modal.backdrop) {
                    closeModal();
                }
            });
        }
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && modal.backdrop && !modal.backdrop.hidden) {
                event.preventDefault();
                closeModal();
            }
        });
        attachSortHandlers();
    }

    bindEvents();
    loadUpdates();
})();
