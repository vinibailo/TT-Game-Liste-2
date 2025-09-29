(function () {
    const config = window.updatesConfig || {};
    const resolvedPageSize = toPositiveInt(config.pageSize, 200, { max: 500 });
    const state = {
        updates: [],
        filtered: [],
        filteredAll: [],
        sortKey: 'refreshed_at',
        sortDir: 'desc',
        searchTerm: '',
        detailCache: new Map(),
        updateMap: new Map(),
        coverCache: new Map(),
        fixingNames: false,
        deduping: false,
        refreshing: false,
        comparing: false,
        jobs: new Map(),
        jobPollers: new Map(),
        activeJobIds: new Map(),
        completedJobs: new Set(),
        refreshStatusTimer: null,
        refreshPhase: 'idle',
        refreshObservedActive: false,
        refreshPendingPolls: 0,
        waitingForRefreshStart: false,
        taskEntries: new Map(),
        taskVisibility: new Map(),
        cacheStatus: null,
        page: 1,
        pageSize: resolvedPageSize,
        pageCount: 1,
        totalAvailable: 0,
    };

    const JOB_TYPES = Object.freeze({
        refresh: 'refresh_updates',
        compare: 'compare_updates',
        fix: 'fix_names',
        dedupe: 'remove_duplicates',
    });
    const JOB_ACTIVE_STATUSES = new Set(['pending', 'running']);
    const JOB_POLL_INTERVAL = 2000;
    const REFRESH_STATUS_INTERVAL = 1000;
    const REFRESH_PENDING_MAX_POLLS = 5;

    const TASK_CONFIG = {
        [JOB_TYPES.refresh]: {
            name: 'Updating IGDB cache',
            icon: 'refresh',
            defaultMessage: 'Updating IGDB cache…',
        },
        [JOB_TYPES.compare]: {
            name: 'Comparing catalog entries',
            icon: 'compare_arrows',
            defaultMessage: 'Comparing processed games…',
        },
        [JOB_TYPES.fix]: {
            name: 'Fixing IGDB names',
            icon: 'spellcheck',
            defaultMessage: 'Fixing IGDB names…',
        },
        [JOB_TYPES.dedupe]: {
            name: 'Removing duplicates',
            icon: 'layers_clear',
            defaultMessage: 'Removing duplicates…',
        },
    };

    const TASK_ORDER = [
        JOB_TYPES.refresh,
        JOB_TYPES.compare,
        JOB_TYPES.fix,
        JOB_TYPES.dedupe,
    ];

    Object.values(JOB_TYPES).forEach((jobType) => {
        state.taskVisibility.set(jobType, false);
    });

    const placeholderImage = '/no-image.jpg';
    const COVER_FETCH_CONCURRENCY = 4;

    const coverFetchState = {
        active: 0,
        queue: [],
        enqueued: new Set(),
        targets: new Map(),
    };

    const DEFAULT_OFFSET = 0;
    const DEFAULT_REFRESH_LIMIT = 200;
    const DEFAULT_UPDATES_LIMIT = 100;

    const elements = {
        tableBody: document.querySelector('[data-updates-body]'),
        emptyState: document.querySelector('[data-empty-state]'),
        loadingState: document.querySelector('[data-loading-state]'),
        searchInput: document.querySelector('[data-search]'),
        refreshButton: document.querySelector('[data-refresh]'),
        compareButton: document.querySelector('[data-compare]'),
        countLabel: document.querySelector('[data-count]'),
        statusLabel: document.querySelector('[data-refresh-status]'),
        sortButtons: Array.from(document.querySelectorAll('.sort-button[data-sort]')),
        fixButton: document.querySelector('[data-fix-names]'),
        dedupeButton: document.querySelector('[data-remove-duplicates]'),
        taskBanner: document.querySelector('[data-task-banner]'),
        taskList: document.querySelector('[data-task-list]'),
        cacheStatusMessage: document.querySelector('[data-cache-status-message]'),
        cacheCount: document.querySelector('[data-cache-count]'),
        cacheSynced: document.querySelector('[data-cache-synced]'),
        cacheRemote: document.querySelector('[data-cache-remote]'),
        cacheSummary: document.querySelector('[data-cache-summary]'),
        cacheInserted: document.querySelector('[data-cache-inserted]'),
        cacheUpdated: document.querySelector('[data-cache-updated]'),
        cacheUnchanged: document.querySelector('[data-cache-unchanged]'),
        cacheRefreshed: document.querySelector('[data-cache-refreshed]'),
        cacheRefreshValue: document.querySelector('[data-cache-refresh]'),
        pagination: document.querySelector('[data-pagination]'),
        paginationInfo: document.querySelector('[data-page-info]'),
        paginationPrev: document.querySelector('[data-page-prev]'),
        paginationNext: document.querySelector('[data-page-next]'),
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
        cover: document.querySelector('[data-modal-cover]'),
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

    function clampPercent(value) {
        const number = Number(value);
        if (!Number.isFinite(number)) {
            return null;
        }
        if (number <= 0) {
            return 0;
        }
        if (number >= 100) {
            return 100;
        }
        return number;
    }

    function formatPercentLabel(value) {
        if (value === null || value === undefined || Number.isNaN(value)) {
            return '…';
        }
        const normalized = clampPercent(value);
        if (normalized === null) {
            return '…';
        }
        return normalized % 1 === 0
            ? `${normalized.toFixed(0)}%`
            : `${normalized.toFixed(1)}%`;
    }

    function showTask(jobType, overrides = {}) {
        if (!jobType || !TASK_CONFIG[jobType]) {
            return;
        }
        state.taskVisibility.set(jobType, true);
        const config = TASK_CONFIG[jobType];
        const percentValue = overrides.percent !== undefined && overrides.percent !== null
            ? clampPercent(overrides.percent)
            : null;
        const messageValue = typeof overrides.message === 'string' && overrides.message.trim()
            ? overrides.message.trim()
            : config.defaultMessage;
        state.taskEntries.set(jobType, {
            name: config.name,
            icon: config.icon,
            message: messageValue,
            percent: percentValue,
        });
        renderTaskBanner();
    }

    function updateTask(jobType, overrides = {}) {
        if (!jobType || !TASK_CONFIG[jobType]) {
            return;
        }
        if (!state.taskVisibility.get(jobType)) {
            return;
        }
        const existing = state.taskEntries.get(jobType) || {};
        const config = TASK_CONFIG[jobType];
        const percentValue = overrides.percent !== undefined && overrides.percent !== null
            ? clampPercent(overrides.percent)
            : existing.percent ?? null;
        const messageValue = typeof overrides.message === 'string' && overrides.message.trim()
            ? overrides.message.trim()
            : existing.message || config.defaultMessage;
        state.taskEntries.set(jobType, {
            name: config.name,
            icon: config.icon,
            message: messageValue,
            percent: percentValue,
        });
        renderTaskBanner();
    }

    function hideTask(jobType) {
        if (!jobType) {
            return;
        }
        state.taskVisibility.set(jobType, false);
        state.taskEntries.delete(jobType);
        renderTaskBanner();
    }

    function renderTaskBanner() {
        const container = elements.taskBanner;
        const list = elements.taskList;
        if (!container || !list) {
            return;
        }
        list.innerHTML = '';
        const entries = [];
        TASK_ORDER.forEach((jobType) => {
            if (!state.taskEntries.has(jobType)) {
                return;
            }
            const entry = state.taskEntries.get(jobType);
            if (!entry) {
                return;
            }
            entries.push({ type: jobType, ...entry });
        });
        if (!entries.length) {
            container.hidden = true;
            container.setAttribute('aria-hidden', 'true');
            return;
        }
        const fragment = document.createDocumentFragment();
        entries.forEach((entry) => {
            const item = document.createElement('li');
            item.className = 'task-banner-item';
            const main = document.createElement('div');
            main.className = 'task-banner-item-main';
            const icon = document.createElement('span');
            icon.className = 'task-banner-item-icon material-symbols-rounded';
            icon.textContent = entry.icon;
            icon.setAttribute('aria-hidden', 'true');
            const text = document.createElement('div');
            text.className = 'task-banner-item-text';
            const name = document.createElement('span');
            name.className = 'task-banner-item-name';
            name.textContent = entry.name;
            const message = document.createElement('span');
            message.className = 'task-banner-item-message';
            message.textContent = entry.message || TASK_CONFIG[entry.type].defaultMessage;
            text.appendChild(name);
            text.appendChild(message);
            main.appendChild(icon);
            main.appendChild(text);
            item.appendChild(main);
            const progress = document.createElement('span');
            progress.className = 'task-banner-item-progress';
            progress.textContent = formatPercentLabel(entry.percent);
            item.appendChild(progress);
            fragment.appendChild(item);
        });
        list.appendChild(fragment);
        container.hidden = false;
        container.removeAttribute('aria-hidden');
    }

    function formatCount(value, fallback = '—') {
        if (value === null || value === undefined) {
            return fallback;
        }
        const number = Number(value);
        if (!Number.isFinite(number)) {
            return fallback;
        }
        return Math.max(0, Math.round(number)).toLocaleString();
    }

    function buildProgressMessage(prefix, processed, total, noun, fallback) {
        const processedValue = toNonNegativeInt(processed, 0);
        const totalNumber = Number(total);
        const totalValue = Number.isFinite(totalNumber)
            ? Math.max(Math.round(totalNumber), 0)
            : 0;
        const nounSuffix = noun ? ` ${noun}` : '';
        if (totalValue > 0) {
            return `${prefix} ${formatCount(processedValue)} of ${formatCount(totalValue)}${nounSuffix}`;
        }
        if (processedValue > 0) {
            return `${prefix} ${formatCount(processedValue)}${nounSuffix}`;
        }
        return fallback;
    }

    function toNonNegativeInt(value, fallback = 0) {
        const number = Number(value);
        if (!Number.isFinite(number)) {
            return fallback;
        }
        return Math.max(Math.trunc(number), 0);
    }

    function toPositiveInt(value, fallback = 1, options = {}) {
        const number = Number(value);
        if (!Number.isFinite(number) || number <= 0) {
            return options.max ? Math.min(fallback, options.max) : fallback;
        }
        const truncated = Math.trunc(number);
        if (options.max && truncated > options.max) {
            return options.max;
        }
        return truncated;
    }

    function resolveOffset(value) {
        if (value === undefined || value === null || value === '') {
            return DEFAULT_OFFSET;
        }
        return toNonNegativeInt(value, DEFAULT_OFFSET);
    }

    function resolveLimit(value, fallback = DEFAULT_REFRESH_LIMIT) {
        if (value === undefined || value === null || value === '') {
            return fallback;
        }
        return toPositiveInt(value, fallback, { max: 500 });
    }

    function logFetch(url) {
        try {
            const resolved = url instanceof URL ? url.toString() : String(url);
            console.log('[updates] Fetching URL:', resolved);
        } catch (err) {
            console.log('[updates] Fetching URL:', url);
        }
    }

    function normalizeId(value) {
        const parsed = Number.parseInt(value, 10);
        if (Number.isNaN(parsed)) {
            return null;
        }
        return parsed;
    }

    function getUpdateById(value) {
        const normalized = normalizeId(value);
        if (normalized !== null && state.updateMap.has(normalized)) {
            return state.updateMap.get(normalized) || null;
        }
        if (state.updateMap.has(value)) {
            return state.updateMap.get(value) || null;
        }
        return null;
    }

    function clearRefreshStatusTimer() {
        if (state.refreshStatusTimer) {
            window.clearInterval(state.refreshStatusTimer);
            state.refreshStatusTimer = null;
            state.refreshPendingPolls = 0;
            state.waitingForRefreshStart = false;
        }
    }

    function startRefreshStatusPolling(immediate = false) {
        if (state.refreshStatusTimer) {
            return;
        }
        if (immediate) {
            pollRefreshStatus();
        }
        state.refreshStatusTimer = window.setInterval(pollRefreshStatus, REFRESH_STATUS_INTERVAL);
    }

    async function pollRefreshStatus() {
        const url = config.refreshStatusUrl || '/api/updates/status';
        try {
            logFetch(url);
            const response = await fetch(url, { headers: { Accept: 'application/json' } });
            if (response.status === 504) {
                showToast('Server is busy; retrying...', 'warning');
                return;
            }
            let payload = null;
            try {
                payload = await response.json();
            } catch (err) {
                payload = null;
            }
            if (!response.ok || !payload || payload.error) {
                throw new Error(payload && payload.error ? payload.error : 'Failed to fetch refresh status.');
            }
            applyRefreshStatus(payload);
        } catch (error) {
            if (error instanceof TypeError) {
                showToast('Server is busy; retrying...', 'warning');
                return;
            }
            console.error('Failed to fetch refresh status', error);
            showToast(error.message || 'Failed to fetch refresh status.', 'warning');
            clearRefreshStatusTimer();
            state.refreshObservedActive = false;
            state.refreshPhase = 'idle';
            state.refreshing = false;
            state.refreshPendingPolls = 0;
            state.waitingForRefreshStart = false;
            setRefreshButtonLoading(false);
            setRefreshProgressVisible(false);
            updateRefreshProgress(0, 0, 'idle');
        }
    }

    function applyRefreshStatus(status) {
        if (!status || typeof status !== 'object') {
            return;
        }
        const phase = typeof status.phase === 'string' ? status.phase : 'idle';
        const processedNumber = Number(status.processed);
        const queuedNumber = Number(status.queued);
        const processed = Number.isFinite(processedNumber) ? Math.max(Math.round(processedNumber), 0) : 0;
        const queued = Number.isFinite(queuedNumber) ? Math.max(Math.round(queuedNumber), 0) : 0;
        const total = queued > 0 ? processed + queued : processed;
        const waitingForStart = state.waitingForRefreshStart;

        state.refreshPhase = phase;
        const displayPhase = phase === 'idle' && waitingForStart ? 'pending' : phase;
        updateRefreshProgress(processed, total, displayPhase);

        if (phase !== 'idle') {
            state.refreshObservedActive = true;
            state.refreshing = true;
            state.refreshPendingPolls = 0;
            state.waitingForRefreshStart = false;
            setRefreshButtonLoading(true);
            setRefreshProgressVisible(true);
            return;
        }

        if (state.refreshObservedActive) {
            state.refreshObservedActive = false;
            clearRefreshStatusTimer();
            state.refreshing = false;
            state.refreshPendingPolls = 0;
            state.waitingForRefreshStart = false;
            setRefreshButtonLoading(false);
            setRefreshProgressVisible(false);
            state.detailCache.clear();
            loadUpdates();
            loadCacheStatus();
            return;
        }

        if (waitingForStart) {
            state.refreshPendingPolls += 1;
            if (state.refreshPendingPolls <= REFRESH_PENDING_MAX_POLLS) {
                state.refreshing = true;
                setRefreshButtonLoading(true);
                setRefreshProgressVisible(true);
                return;
            }
            state.waitingForRefreshStart = false;
            state.refreshPendingPolls = 0;
        }

        clearRefreshStatusTimer();
        state.refreshing = false;
        state.refreshPendingPolls = 0;
        state.waitingForRefreshStart = false;
        setRefreshButtonLoading(false);
        setRefreshProgressVisible(false);
        updateRefreshProgress(0, 0, 'idle');
    }

    async function fetchJson(url, options = {}) {
        const { logRequest = true, headers: providedHeaders, ...rest } = options || {};
        if (logRequest) {
            logFetch(url);
        }
        const headers = new Headers(providedHeaders || {});
        headers.set('Accept', 'application/json');
        const requestOptions = {
            credentials: 'same-origin',
            ...rest,
            headers,
        };
        const response = await fetch(url, requestOptions);
        if (response.status === 504) {
            throw new Error('Server timed out (504). Please try again later.');
        }
        const contentType = (response.headers && response.headers.get('content-type')) || '';
        const isJson = contentType.toLowerCase().includes('application/json');
        if (response.ok && !isJson && (response.status === 204 || response.status === 205 || response.status === 304)) {
            return null;
        }
        if (!isJson) {
            let preview = '';
            try {
                const text = await response.text();
                preview = text.slice(0, 160);
            } catch (err) {
                preview = '';
            }
            const message = preview ? `Non-JSON response: ${preview}` : 'Non-JSON response: (empty)';
            throw new Error(message);
        }
        let payload;
        try {
            payload = await response.json();
        } catch (err) {
            throw new Error('Invalid JSON response.');
        }
        if (response.ok) {
            return payload;
        }
        if (payload && typeof payload === 'object' && payload.error) {
            throw new Error(String(payload.error));
        }
        if (payload && typeof payload === 'object' && typeof payload.message === 'string') {
            throw new Error(payload.message);
        }
        throw new Error(`Request failed with status ${response.status}`);
    }

    function getJobDetailUrl(jobId) {
        if (!jobId) {
            return null;
        }
        if (config.jobDetailUrlTemplate) {
            return config.jobDetailUrlTemplate.replace('{id}', encodeURIComponent(jobId));
        }
        return `/api/updates/jobs/${encodeURIComponent(jobId)}`;
    }

    function isJobActive(status) {
        return status && JOB_ACTIVE_STATUSES.has(status);
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
        elements.countLabel.textContent = String(state.filteredAll.length);
    }

    function updateStatusMessage() {
        if (!elements.statusLabel) {
            return;
        }
        if (state.searchTerm) {
            const count = state.filteredAll.length;
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

    function applyCacheStatus(status, options = {}) {
        const data = status && typeof status === 'object' ? status : null;
        state.cacheStatus = data;
        const errorMessage = options && typeof options.error === 'string' && options.error.trim()
            ? options.error.trim()
            : '';

        if (!data) {
            if (elements.cacheStatusMessage) {
                elements.cacheStatusMessage.textContent = errorMessage || 'Cache status unavailable.';
            }
            if (elements.cacheCount) {
                elements.cacheCount.textContent = '—';
            }
            if (elements.cacheSynced) {
                elements.cacheSynced.textContent = '—';
            }
            if (elements.cacheRemote) {
                elements.cacheRemote.textContent = '—';
            }
            if (elements.cacheSummary) {
                elements.cacheSummary.hidden = true;
            }
            if (elements.cacheRefreshed) {
                elements.cacheRefreshed.hidden = true;
            }
            if (elements.cacheRefreshValue) {
                elements.cacheRefreshValue.textContent = '—';
            }
            if (elements.cacheInserted) {
                elements.cacheInserted.textContent = '0';
            }
            if (elements.cacheUpdated) {
                elements.cacheUpdated.textContent = '0';
            }
            if (elements.cacheUnchanged) {
                elements.cacheUnchanged.textContent = '0';
            }
            return;
        }

        const cachedEntries = data.cached_entries;
        if (elements.cacheCount) {
            elements.cacheCount.textContent = formatCount(cachedEntries, '0');
        }

        const remoteNumber = Number(data.remote_total);
        const remoteTotal = Number.isFinite(remoteNumber) ? Math.max(Math.round(remoteNumber), 0) : null;
        if (elements.cacheRemote) {
            elements.cacheRemote.textContent = remoteTotal !== null ? formatCount(remoteTotal, '—') : '—';
        }

        const syncedText = data.last_synced_at ? formatDate(data.last_synced_at) : '—';
        if (elements.cacheSynced) {
            elements.cacheSynced.textContent = syncedText;
        }

        const lastRefresh = data.last_refresh && typeof data.last_refresh === 'object'
            ? data.last_refresh
            : null;
        if (elements.cacheSummary) {
            const hasSummary = Boolean(lastRefresh);
            elements.cacheSummary.hidden = !hasSummary;
            if (hasSummary) {
                if (elements.cacheInserted) {
                    elements.cacheInserted.textContent = formatCount(lastRefresh.inserted, '0');
                }
                if (elements.cacheUpdated) {
                    elements.cacheUpdated.textContent = formatCount(lastRefresh.updated, '0');
                }
                if (elements.cacheUnchanged) {
                    elements.cacheUnchanged.textContent = formatCount(lastRefresh.unchanged, '0');
                }
            }
        }

        if (elements.cacheRefreshed) {
            const finishedAt = lastRefresh && (lastRefresh.finished_at || lastRefresh.started_at);
            if (finishedAt) {
                elements.cacheRefreshed.hidden = false;
                if (elements.cacheRefreshValue) {
                    elements.cacheRefreshValue.textContent = formatDate(finishedAt);
                }
            } else {
                elements.cacheRefreshed.hidden = true;
                if (elements.cacheRefreshValue) {
                    elements.cacheRefreshValue.textContent = '—';
                }
            }
        }

        let summaryMessage = '';
        if (lastRefresh && typeof lastRefresh.message === 'string' && lastRefresh.message.trim()) {
            summaryMessage = lastRefresh.message.trim();
        } else if (remoteTotal !== null && Number.isFinite(remoteTotal)) {
            const cachedValue = Number.isFinite(Number(cachedEntries))
                ? toNonNegativeInt(cachedEntries, 0)
                : null;
            if (cachedValue !== null) {
                summaryMessage = cachedValue >= remoteTotal
                    ? 'Cache is up to date with IGDB.'
                    : `Cached ${formatCount(cachedValue)} of ${formatCount(remoteTotal)} IGDB records.`;
            } else {
                summaryMessage = `IGDB reports ${formatCount(remoteTotal)} records.`;
            }
        } else if (Number.isFinite(Number(cachedEntries))) {
            const cachedValue = toNonNegativeInt(cachedEntries, 0);
            summaryMessage = `Cached ${formatCount(cachedValue)} IGDB record${cachedValue === 1 ? '' : 's'}.`;
        } else {
            summaryMessage = 'Cache status unavailable.';
        }

        if (elements.cacheStatusMessage) {
            elements.cacheStatusMessage.textContent = summaryMessage;
        }
    }

    async function loadCacheStatus() {
        if (!config.cacheStatusUrl) {
            return;
        }
        try {
            const payload = await fetchJson(config.cacheStatusUrl, { logRequest: false });
            applyCacheStatus(payload || {});
        } catch (error) {
            console.error('Failed to load cache status', error);
            applyCacheStatus(null, { error: error.message || 'Unable to load cache status.' });
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

    function applyFilters(options = {}) {
        const { resetPage = false } = options;
        const term = state.searchTerm.trim().toLocaleLowerCase();
        const filteredAll = term
            ? state.updates.filter((item) => {
                  const name = (item.name || '').toLocaleLowerCase();
                  const id = String(item.processed_game_id || '');
                  const igdbId = String(item.igdb_id || '');
                  return name.includes(term) || id.includes(term) || igdbId.includes(term);
              })
            : state.updates.slice();

        filteredAll.sort((a, b) => {
            const comparison = compareValues(a, b, state.sortKey);
            return state.sortDir === 'asc' ? comparison : -comparison;
        });

        state.filteredAll = filteredAll;
        if (resetPage) {
            state.page = 1;
        }
        updateVisiblePage();
    }

    function updateVisiblePage() {
        const totalItems = state.filteredAll.length;
        const totalPages = Math.max(1, Math.ceil(totalItems / state.pageSize));
        state.pageCount = totalPages;
        if (totalItems === 0) {
            state.page = 1;
        } else if (state.page > totalPages) {
            state.page = totalPages;
        } else if (state.page < 1) {
            state.page = 1;
        }
        const start = (state.page - 1) * state.pageSize;
        const end = start + state.pageSize;
        state.filtered = state.filteredAll.slice(start, end);
        updateCount();
        renderTable();
        renderPagination();
        updateStatusMessage();
    }

    function createIcon(content, className) {
        const span = document.createElement('span');
        span.className = className;
        span.setAttribute('aria-hidden', 'true');
        span.textContent = content;
        return span;
    }

    function createTag(label, modifier) {
        const span = document.createElement('span');
        span.className = 'update-tag';
        if (modifier) {
            span.classList.add(modifier);
        }
        span.textContent = label;
        return span;
    }

    function buildDeleteDuplicateUrl(id) {
        if (config.deleteDuplicateUrlTemplate) {
            return config.deleteDuplicateUrlTemplate.replace('{id}', encodeURIComponent(id));
        }
        return `/api/updates/remove-duplicate/${encodeURIComponent(id)}`;
    }

    function resolveCoverSource(source) {
        return source ? source : placeholderImage;
    }

    function setModalCover(source, name) {
        if (!modal.cover) {
            return;
        }
        modal.cover.src = resolveCoverSource(source);
        modal.cover.alt = name ? `${name} cover` : 'Game cover';
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
            const gameCell = document.createElement('div');
            gameCell.className = 'game-cell';
            const coverWrapper = document.createElement('div');
            coverWrapper.className = 'game-cover';
            const coverImage = document.createElement('img');
            coverImage.loading = 'lazy';
            coverImage.decoding = 'async';
            coverImage.alt = '';
            ensureCoverImageById(item.processed_game_id, coverImage, item);
            coverWrapper.appendChild(coverImage);
            const infoWrapper = document.createElement('div');
            infoWrapper.className = 'game-info';
            const nameText = document.createElement('span');
            nameText.className = 'game-name';
            nameText.textContent = item.name || 'Unnamed game';
            infoWrapper.appendChild(nameText);
            const tagList = document.createElement('div');
            tagList.className = 'game-tags';
            if (item.update_type === 'mismatch') {
                tagList.appendChild(createTag('Mismatch', 'is-mismatch'));
            }
            if (item.update_type === 'duplicate') {
                tagList.appendChild(createTag('Duplicate', 'is-duplicate'));
            }
            if (tagList.childElementCount > 0) {
                infoWrapper.appendChild(tagList);
            }
            gameCell.appendChild(coverWrapper);
            gameCell.appendChild(infoWrapper);
            nameCell.appendChild(gameCell);
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

            if (item.detail_available !== false) {
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
            }

            if (item.update_type === 'duplicate') {
                const deleteButton = document.createElement('button');
                deleteButton.type = 'button';
                deleteButton.className = 'icon-button icon-button-danger';
                deleteButton.title = 'Delete duplicate';
                deleteButton.dataset.duplicateId = String(item.processed_game_id);
                deleteButton.appendChild(createIcon('delete', 'material-symbols-rounded'));
                const deleteLabel = document.createElement('span');
                deleteLabel.className = 'sr-only';
                deleteLabel.textContent = `Delete duplicate ${item.name || 'game'}`;
                deleteButton.appendChild(deleteLabel);
                deleteButton.addEventListener('click', () => {
                    handleDeleteDuplicate(item, deleteButton);
                });
                actionsCell.appendChild(deleteButton);
            }

            row.appendChild(actionsCell);
            fragment.appendChild(row);
        });
        elements.tableBody.appendChild(fragment);
    }

    function renderPagination() {
        const container = elements.pagination;
        if (!container) {
            return;
        }
        const totalItems = state.filteredAll.length;
        const shouldShow = totalItems > state.pageSize;
        container.hidden = !shouldShow;
        if (!shouldShow) {
            return;
        }
        if (elements.paginationInfo) {
            elements.paginationInfo.textContent = `Page ${state.page} of ${state.pageCount}`;
        }
        if (elements.paginationPrev) {
            const disablePrev = state.page <= 1;
            elements.paginationPrev.disabled = disablePrev;
            elements.paginationPrev.setAttribute('aria-disabled', disablePrev ? 'true' : 'false');
        }
        if (elements.paginationNext) {
            const disableNext = state.page >= state.pageCount;
            elements.paginationNext.disabled = disableNext;
            elements.paginationNext.setAttribute('aria-disabled', disableNext ? 'true' : 'false');
        }
    }

    function setPage(page) {
        const totalPages = Math.max(1, state.pageCount);
        const clamped = Math.min(Math.max(page, 1), totalPages);
        if (clamped === state.page) {
            return;
        }
        state.page = clamped;
        updateVisiblePage();
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
        applyFilters({ resetPage: true });
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

    function setCompareButtonLoading(loading) {
        const button = elements.compareButton;
        if (!button) {
            return;
        }
        const label = button.querySelector('.btn-label');
        if (loading) {
            button.disabled = true;
            button.setAttribute('aria-busy', 'true');
            if (label) {
                if (!button.dataset.originalLabel) {
                    button.dataset.originalLabel = label.textContent || '';
                }
                label.textContent = 'Comparing…';
            }
        } else {
            button.disabled = false;
            button.removeAttribute('aria-busy');
            if (label && button.dataset.originalLabel) {
                label.textContent = button.dataset.originalLabel;
            }
        }
    }

    function setFixButtonLoading(loading) {
        const button = elements.fixButton;
        if (!button) {
            return;
        }
        const label = button.querySelector('.btn-label');
        if (loading) {
            button.disabled = true;
            button.setAttribute('aria-busy', 'true');
            if (label) {
                if (!button.dataset.originalLabel) {
                    button.dataset.originalLabel = label.textContent || '';
                }
                label.textContent = 'Fixing…';
            }
        } else {
            button.disabled = false;
            button.removeAttribute('aria-busy');
            if (label) {
                const original = button.dataset.originalLabel;
                if (original) {
                    label.textContent = original;
                }
            }
        }
    }

    function setDedupeButtonLoading(loading) {
        const button = elements.dedupeButton;
        if (!button) {
            return;
        }
        const label = button.querySelector('.btn-label');
        if (loading) {
            button.disabled = true;
            button.setAttribute('aria-busy', 'true');
            if (label) {
                if (!button.dataset.originalLabel) {
                    button.dataset.originalLabel = label.textContent || '';
                }
                label.textContent = 'Removing…';
            }
        } else {
            button.disabled = false;
            button.removeAttribute('aria-busy');
            if (label && button.dataset.originalLabel) {
                label.textContent = button.dataset.originalLabel;
            }
        }
    }

    function setDedupeProgressVisible(visible) {
        if (visible) {
            showTask(JOB_TYPES.dedupe, {
                message: 'Preparing duplicate cleanup…',
                percent: null,
            });
        } else {
            hideTask(JOB_TYPES.dedupe);
        }
    }

    function updateDedupeProgress(processed, total) {
        if (!state.taskVisibility.get(JOB_TYPES.dedupe)) {
            return;
        }
        const processedValue = toNonNegativeInt(processed, 0);
        const totalNumber = Number(total);
        const totalValue = Number.isFinite(totalNumber)
            ? Math.max(Math.round(totalNumber), 0)
            : 0;
        const percent = totalValue > 0 ? (processedValue / totalValue) * 100 : null;
        const message = buildProgressMessage(
            'Processing',
            processed,
            total,
            'records',
            'Removing duplicates…',
        );
        updateTask(JOB_TYPES.dedupe, { percent, message });
    }

    function setFixProgressVisible(visible) {
        if (visible) {
            showTask(JOB_TYPES.fix, {
                message: 'Preparing name fixes…',
                percent: null,
            });
        } else {
            hideTask(JOB_TYPES.fix);
        }
    }

    function updateFixProgress(processed, total) {
        if (!state.taskVisibility.get(JOB_TYPES.fix)) {
            return;
        }
        const processedValue = toNonNegativeInt(processed, 0);
        const totalNumber = Number(total);
        const totalValue = Number.isFinite(totalNumber)
            ? Math.max(Math.round(totalNumber), 0)
            : 0;
        const percent = totalValue > 0 ? (processedValue / totalValue) * 100 : null;
        const message = buildProgressMessage('Fixing', processed, total, 'names', 'Fixing IGDB names…');
        updateTask(JOB_TYPES.fix, { percent, message });
    }

    function setRefreshProgressVisible(visible) {
        if (visible) {
            showTask(JOB_TYPES.refresh, {
                message: 'Preparing IGDB cache…',
                percent: null,
            });
        } else {
            hideTask(JOB_TYPES.refresh);
        }
    }

    function updateRefreshProgress(processed, total, phase = state.refreshPhase || 'idle', options = {}) {
        if (!state.taskVisibility.get(JOB_TYPES.refresh)) {
            return;
        }
        const processedValue = toNonNegativeInt(processed, 0);
        const totalNumber = Number(total);
        const totalValue = Number.isFinite(totalNumber)
            ? Math.max(Math.round(totalNumber), 0)
            : 0;
        let percent = null;
        if (totalValue > 0) {
            percent = (processedValue / totalValue) * 100;
        } else if (phase === 'idle') {
            percent = 100;
        }
        let message = options && typeof options.message === 'string' && options.message.trim()
            ? options.message.trim()
            : '';
        if (!message) {
            switch (phase) {
                case 'pending':
                    message = 'Preparing IGDB cache…';
                    break;
                case 'running':
                case 'cache':
                    message = buildProgressMessage(
                        'Processing',
                        processed,
                        total,
                        'cache rows',
                        'Updating IGDB cache…',
                    );
                    break;
                case 'idle':
                    message = 'IGDB cache update complete.';
                    break;
                default:
                    message = 'Updating IGDB cache…';
                    break;
            }
        }
        updateTask(JOB_TYPES.refresh, { percent, message });
    }

    function setCompareProgressVisible(visible) {
        if (visible) {
            showTask(JOB_TYPES.compare, {
                message: 'Preparing comparison…',
                percent: null,
            });
        } else {
            hideTask(JOB_TYPES.compare);
        }
    }

    function updateCompareProgress(processed, total, options = {}) {
        if (!state.taskVisibility.get(JOB_TYPES.compare)) {
            return;
        }
        const processedValue = toNonNegativeInt(processed, 0);
        const totalNumber = Number(total);
        const totalValue = Number.isFinite(totalNumber)
            ? Math.max(Math.round(totalNumber), 0)
            : 0;
        const percent = totalValue > 0 ? (processedValue / totalValue) * 100 : null;
        let message = options && typeof options.message === 'string' && options.message.trim()
            ? options.message.trim()
            : '';
        if (!message) {
            const data = options && typeof options === 'object' ? options.data : null;
            if (data && typeof data === 'object' && typeof data.phase === 'string') {
                if (data.phase === 'diffs' && Number.isFinite(Number(data.missing_count))) {
                    const missingCount = Number(data.missing_count);
                    if (missingCount > 0) {
                        message = `Comparing entries… ${formatCount(missingCount)} missing IGDB record${missingCount === 1 ? '' : 's'}.`;
                    }
                } else if (data.phase === 'idle' || data.phase === 'done') {
                    message = 'Comparison complete.';
                }
            }
        }
        if (!message) {
            message = buildProgressMessage('Comparing', processed, total, 'entries', 'Comparing entries…');
        }
        updateTask(JOB_TYPES.compare, { percent, message });
    }

    function updateProgressBar(processed, total) {
        const processedValue = toNonNegativeInt(processed, 0);
        const totalValueRaw = toNonNegativeInt(total, processedValue);
        const normalizedTotal = Math.max(totalValueRaw, processedValue);
        const phase = normalizedTotal > 0 && processedValue >= normalizedTotal ? 'idle' : 'running';
        updateRefreshProgress(processedValue, normalizedTotal, phase);
    }

    function clearJobPoller(jobId) {
        const timer = state.jobPollers.get(jobId);
        if (timer) {
            window.clearTimeout(timer);
            state.jobPollers.delete(jobId);
        }
    }

    async function fetchJobDetail(jobId) {
        const url = getJobDetailUrl(jobId);
        if (!url) {
            throw new Error('Job tracking endpoint is not configured.');
        }
        const payload = await fetchJson(url);
        if (!payload || !payload.job) {
            throw new Error('Failed to retrieve job details.');
        }
        return payload.job;
    }

    async function startJob(url) {
        const payload = await fetchJson(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });
        if (!payload || !payload.job) {
            throw new Error('Job response was empty.');
        }
        return payload.job;
    }

    function updateJobUi(job) {
        if (!job || !job.id) {
            return;
        }
        state.jobs.set(job.id, job);
        const type = job.job_type;
        const active = isJobActive(job.status);
        if (type) {
            if (active) {
                state.activeJobIds.set(type, job.id);
            } else if (state.activeJobIds.get(type) === job.id) {
                state.activeJobIds.delete(type);
            }
        }

        switch (type) {
            case JOB_TYPES.compare: {
                state.comparing = active;
                setCompareButtonLoading(active);
                if (active) {
                    setCompareProgressVisible(true);
                    updateCompareProgress(job.progress_current || 0, job.progress_total || 0, {
                        data: job.data,
                        message: job.message,
                    });
                } else {
                    updateCompareProgress(0, 0);
                    setCompareProgressVisible(false);
                }
                break;
            }
            case JOB_TYPES.fix: {
                state.fixingNames = active;
                setFixButtonLoading(active);
                if (active) {
                    setFixProgressVisible(true);
                    updateFixProgress(job.progress_current || 0, job.progress_total || 0);
                } else {
                    updateFixProgress(0, 0);
                    setFixProgressVisible(false);
                }
                break;
            }
            case JOB_TYPES.refresh: {
                state.refreshing = active;
                if (active) {
                    const phase = job.data && typeof job.data.phase === 'string' ? job.data.phase : 'running';
                    state.refreshPhase = phase;
                    if (phase !== 'idle') {
                        state.refreshObservedActive = true;
                    }
                    state.refreshPendingPolls = 0;
                    state.waitingForRefreshStart = false;
                    setRefreshButtonLoading(true);
                    setRefreshProgressVisible(true);
                    startRefreshStatusPolling(true);
                } else {
                    if (!state.refreshStatusTimer) {
                        state.refreshPhase = 'idle';
                        state.refreshObservedActive = false;
                        state.refreshPendingPolls = 0;
                        state.waitingForRefreshStart = false;
                        setRefreshButtonLoading(false);
                        setRefreshProgressVisible(false);
                        updateRefreshProgress(0, 0, 'idle');
                    }
                }
                if (elements.statusLabel && job.message) {
                    elements.statusLabel.textContent = job.message;
                }
                break;
            }
            case JOB_TYPES.dedupe: {
                state.deduping = active;
                setDedupeButtonLoading(active);
                if (active) {
                    setDedupeProgressVisible(true);
                    updateDedupeProgress(job.progress_current || 0, job.progress_total || 0);
                } else {
                    updateDedupeProgress(0, 0);
                    setDedupeProgressVisible(false);
                }
                break;
            }
            default:
                break;
        }
    }

    function handleJobCompletion(job) {
        if (!job || !job.id || state.completedJobs.has(job.id)) {
            return;
        }
        state.completedJobs.add(job.id);
        clearJobPoller(job.id);
        const type = job.job_type;
        const status = job.status;
        const result = job.result || {};
        if (type === JOB_TYPES.compare) {
            state.comparing = false;
            setCompareButtonLoading(false);
            updateCompareProgress(0, 0);
            setCompareProgressVisible(false);
            if (status === 'success') {
                state.detailCache.clear();
                loadUpdates();
            }
        } else if (type === JOB_TYPES.fix) {
            state.fixingNames = false;
            setFixButtonLoading(false);
            updateFixProgress(0, 0);
            setFixProgressVisible(false);
            if (status === 'success') {
                state.detailCache.clear();
                loadUpdates();
            }
        } else if (type === JOB_TYPES.refresh) {
            state.refreshing = false;
            state.refreshPhase = 'idle';
            state.refreshObservedActive = false;
            state.refreshPendingPolls = 0;
            state.waitingForRefreshStart = false;
            clearRefreshStatusTimer();
            setRefreshButtonLoading(false);
            updateRefreshProgress(0, 0, 'idle');
            setRefreshProgressVisible(false);
            if (status === 'success') {
                state.detailCache.clear();
                loadUpdates();
                loadCacheStatus();
            }
        } else if (type === JOB_TYPES.dedupe) {
            state.deduping = false;
            setDedupeButtonLoading(false);
            updateDedupeProgress(0, 0);
            setDedupeProgressVisible(false);
            if (status === 'success' && Number(result.removed) > 0) {
                state.detailCache.clear();
                loadUpdates();
            }
        }

        if (status === 'error') {
            showToast(job.error || 'Background task failed.', 'warning');
        } else if (result && result.message) {
            showToast(result.message, result.toast_type || 'success');
        }
    }

    async function monitorJob(job) {
        if (!job || !job.id) {
            return;
        }
        updateJobUi(job);
        if (!isJobActive(job.status)) {
            handleJobCompletion(job);
            return;
        }
        clearJobPoller(job.id);
        const poll = async () => {
            try {
                const latest = await fetchJobDetail(job.id);
                updateJobUi(latest);
                if (isJobActive(latest.status)) {
                    const timer = window.setTimeout(poll, JOB_POLL_INTERVAL);
                    state.jobPollers.set(job.id, timer);
                } else {
                    state.jobPollers.delete(job.id);
                    handleJobCompletion(latest);
                }
            } catch (error) {
                console.error('Failed to poll job status', error);
                state.jobPollers.delete(job.id);
                showToast(error.message, 'warning');
            }
        };
        const timer = window.setTimeout(poll, JOB_POLL_INTERVAL);
        state.jobPollers.set(job.id, timer);
    }

    async function loadExistingJobs() {
        if (!config.jobsUrl) {
            return;
        }
        try {
            const payload = await fetchJson(config.jobsUrl);
            const jobs = Array.isArray(payload.jobs) ? payload.jobs : [];
            jobs.forEach((job) => {
                if (!job || !job.id) {
                    return;
                }
                if (job.job_type === JOB_TYPES.refresh) {
                    if (isJobActive(job.status)) {
                        state.completedJobs.delete(job.id);
                    } else {
                        state.completedJobs.add(job.id);
                    }
                    updateJobUi(job);
                    return;
                }
                if (isJobActive(job.status)) {
                    state.completedJobs.delete(job.id);
                    monitorJob(job);
                } else {
                    state.completedJobs.add(job.id);
                    updateJobUi(job);
                }
            });
        } catch (error) {
            console.error('Failed to load background jobs', error);
        }
    }

    function buildDetailUrl(id) {
        if (config.detailUrlTemplate) {
            return config.detailUrlTemplate.replace('{id}', encodeURIComponent(id));
        }
        return `/api/updates/${encodeURIComponent(id)}`;
    }

    function buildCoverUrl(id) {
        if (config.coverUrlTemplate) {
            return config.coverUrlTemplate.replace('{id}', encodeURIComponent(id));
        }
        return `/api/updates/${encodeURIComponent(id)}/cover`;
    }

    async function fetchCoverData(id) {
        const url = buildCoverUrl(id);
        logFetch(url);
        const response = await fetch(url, {
            headers: { Accept: 'application/json' },
        });
        if (response.status === 404) {
            const error = new Error('Cover not found.');
            error.status = 404;
            throw error;
        }
        let payload = null;
        try {
            payload = await response.json();
        } catch (err) {
            payload = null;
        }
        if (!response.ok || !payload || !payload.cover) {
            const message = payload && payload.error ? payload.error : 'Failed to load cover image.';
            const error = new Error(message);
            error.status = response.status;
            throw error;
        }
        return payload.cover;
    }

    function registerCoverTarget(id, target) {
        if (!(target instanceof HTMLImageElement)) {
            return;
        }
        let targets = coverFetchState.targets.get(id);
        if (!targets) {
            targets = new Set();
            coverFetchState.targets.set(id, targets);
        }
        targets.add(target);
    }

    function updateCoverTargets(id, cover) {
        const targets = coverFetchState.targets.get(id);
        if (!targets) {
            return;
        }
        targets.forEach((target) => {
            if (!(target instanceof HTMLImageElement)) {
                return;
            }
            if (target.dataset.coverId !== String(id)) {
                return;
            }
            target.src = cover ? resolveCoverSource(cover) : placeholderImage;
        });
    }

    function processCoverQueue() {
        while (
            coverFetchState.active < COVER_FETCH_CONCURRENCY &&
            coverFetchState.queue.length > 0
        ) {
            const nextId = coverFetchState.queue.shift();
            if (typeof nextId === 'undefined') {
                break;
            }
            coverFetchState.active += 1;
            runCoverFetch(nextId);
        }
    }

    async function runCoverFetch(id) {
        try {
            const cover = await fetchCoverData(id);
            state.coverCache.set(id, cover);
            const update = getUpdateById(id);
            if (update) {
                update.cover = cover;
                update.cover_available = true;
            }
            updateCoverTargets(id, cover);
        } catch (error) {
            const status = error && typeof error === 'object' ? error.status : undefined;
            if (status === 404) {
                state.coverCache.set(id, null);
                const update = getUpdateById(id);
                if (update) {
                    update.cover = null;
                    update.cover_available = false;
                }
            } else {
                console.error('Failed to fetch cover image', error);
            }
            updateCoverTargets(id, null);
        } finally {
            coverFetchState.enqueued.delete(id);
            coverFetchState.targets.delete(id);
            coverFetchState.active = Math.max(coverFetchState.active - 1, 0);
            if (coverFetchState.queue.length > 0) {
                processCoverQueue();
            }
        }
    }

    function queueCoverFetch(id, target) {
        if (!Number.isFinite(id)) {
            return;
        }
        if (target instanceof HTMLImageElement) {
            registerCoverTarget(id, target);
        }
        if (coverFetchState.enqueued.has(id)) {
            return;
        }
        coverFetchState.queue.push(id);
        coverFetchState.enqueued.add(id);
        processCoverQueue();
    }

    function ensureCoverImageById(id, imgElement, fallbackItem) {
        if (!imgElement) {
            return;
        }
        const normalizedId = normalizeId(id);
        if (normalizedId === null) {
            imgElement.src = placeholderImage;
            return;
        }
        imgElement.dataset.coverId = String(normalizedId);
        const update = getUpdateById(normalizedId) || fallbackItem || null;
        let cached = update && update.cover ? update.cover : null;
        if (!cached && state.coverCache.has(normalizedId)) {
            cached = state.coverCache.get(normalizedId);
        }
        if (cached) {
            state.coverCache.set(normalizedId, cached);
            if (update) {
                update.cover = cached;
                update.cover_available = true;
            }
            imgElement.src = resolveCoverSource(cached);
            return;
        }
        if (state.coverCache.has(normalizedId) && !state.coverCache.get(normalizedId)) {
            imgElement.src = placeholderImage;
            if (update) {
                update.cover = null;
                update.cover_available = false;
            }
            return;
        }
        imgElement.src = placeholderImage;
        let available = true;
        if (update && Object.prototype.hasOwnProperty.call(update, 'cover_available')) {
            available = update.cover_available !== false;
        } else if (
            fallbackItem &&
            Object.prototype.hasOwnProperty.call(fallbackItem, 'cover_available')
        ) {
            available = fallbackItem.cover_available !== false;
        }
        if (!available) {
            state.coverCache.set(normalizedId, null);
            if (update) {
                update.cover_available = false;
            }
            return;
        }
        if (!state.updateMap.has(normalizedId) && update) {
            state.updateMap.set(normalizedId, update);
        }
        queueCoverFetch(normalizedId, imgElement);
    }

    async function fetchDiffDetail(id) {
        if (state.detailCache.has(id)) {
            return state.detailCache.get(id);
        }
        const url = buildDetailUrl(id);
        logFetch(url);
        const response = await fetch(url);
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

    async function handleFixNames() {
        if (state.fixingNames) {
            return;
        }
        if (!config.fixNamesUrl) {
            showToast('Name fixing endpoint is not configured.', 'warning');
            return;
        }
        state.fixingNames = true;
        setFixButtonLoading(true);
        setFixProgressVisible(true);
        updateFixProgress(0, 0);
        try {
            const job = await startJob(config.fixNamesUrl);
            monitorJob(job);
        } catch (error) {
            console.error(error);
            showToast(error.message, 'warning');
            state.fixingNames = false;
            setFixButtonLoading(false);
            updateFixProgress(0, 0);
            setFixProgressVisible(false);
        } finally {
            // Loading state will be reset when the job completes or fails.
        }
    }

    async function handleRemoveDuplicates() {
        if (state.deduping) {
            return;
        }
        if (!config.removeDuplicatesUrl) {
            showToast('Duplicate removal endpoint is not configured.', 'warning');
            return;
        }
        state.deduping = true;
        setDedupeButtonLoading(true);
        setDedupeProgressVisible(true);
        updateDedupeProgress(0, 0);
        try {
            const job = await startJob(config.removeDuplicatesUrl);
            monitorJob(job);
        } catch (error) {
            console.error(error);
            showToast(error.message, 'warning');
            state.deduping = false;
            setDedupeButtonLoading(false);
            updateDedupeProgress(0, 0);
            setDedupeProgressVisible(false);
        }
    }

    async function handleDeleteDuplicate(item, button) {
        if (!item || !item.processed_game_id) {
            return;
        }
        if (!config.deleteDuplicateUrlTemplate) {
            showToast('Duplicate removal endpoint is not configured.', 'warning');
            return;
        }
        const id = item.processed_game_id;
        if (button) {
            button.disabled = true;
            button.setAttribute('aria-busy', 'true');
        }
        try {
            const url = buildDeleteDuplicateUrl(id);
            logFetch(url);
            const response = await fetch(url, { method: 'POST' });
            let payload = null;
            try {
                payload = await response.json();
            } catch (err) {
                payload = null;
            }
            if (!response.ok || !payload || payload.error) {
                throw new Error(payload && payload.error ? payload.error : 'Failed to remove duplicate.');
            }
            if (payload.message) {
                showToast(payload.message, payload.toast_type || 'success');
            } else {
                showToast('Removed duplicate entry.', 'success');
            }
            state.detailCache.delete(id);
            await loadUpdates();
        } catch (error) {
            console.error(error);
            showToast(error.message, 'warning');
        } finally {
            if (button) {
                button.disabled = false;
                button.removeAttribute('aria-busy');
            }
        }
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
        setModalCover(null, null);
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
        const update = getUpdateById(id);
        if (!update) {
            showToast('Unable to find update details for that game.', 'warning');
            return;
        }
        showModalShell();
        setModalCover(update.cover, update.name);
        ensureCoverImageById(id, modal.cover, update);
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
            const normalizedId = normalizeId(id);
            if (detail && typeof detail === 'object') {
                if (update) {
                    if (detail.cover) {
                        update.cover = detail.cover;
                    }
                    if (Object.prototype.hasOwnProperty.call(detail, 'cover_available')) {
                        update.cover_available = detail.cover_available !== false;
                    }
                }
                if (normalizedId !== null) {
                    if (detail.cover) {
                        state.coverCache.set(normalizedId, detail.cover);
                    } else if (detail.cover_available === false) {
                        state.coverCache.set(normalizedId, null);
                    }
                }
            }
            setMetaValue(modal.gameId, detail.processed_game_id);
            setMetaValue(modal.igdbId, detail.igdb_id);
            setMetaValue(modal.igdbUpdated, formatDate(detail.igdb_updated_at));
            setMetaValue(modal.localEdited, formatDate(detail.local_last_edited_at));
            setModalCover(detail.cover || update.cover, detail.name || update.name);
            ensureCoverImageById(id, modal.cover, update || detail);
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
        if (state.refreshing) {
            return;
        }
        state.refreshing = true;
        state.refreshPhase = 'pending';
        state.refreshObservedActive = false;
        state.refreshPendingPolls = 0;
        state.waitingForRefreshStart = false;
        clearRefreshStatusTimer();
        setRefreshProgressVisible(true);
        updateRefreshProgress(0, 0, 'pending');
        setRefreshButtonLoading(true);
        const limit = DEFAULT_REFRESH_LIMIT;
        const visitedOffsets = new Set();
        let offset = DEFAULT_OFFSET;
        try {
            let done = false;
            while (!done) {
                const resolvedOffset = resolveOffset(offset);
                if (visitedOffsets.has(resolvedOffset)) {
                    throw new Error('Refresh appears to be stuck.');
                }
                visitedOffsets.add(resolvedOffset);
                const url = `/api/updates/refresh?offset=${encodeURIComponent(resolvedOffset)}&limit=${encodeURIComponent(limit)}`;
                const payload = await fetchJson(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({}),
                });
                if (!payload || typeof payload !== 'object') {
                    throw new Error('Invalid refresh response.');
                }
                const processed = toNonNegativeInt(
                    payload.processed ?? payload.progress_current ?? payload.current ?? payload.completed,
                    0,
                );
                const total = toNonNegativeInt(
                    payload.total ?? payload.progress_total ?? payload.queued ?? payload.total_items ?? payload.count,
                    processed,
                );
                state.refreshPhase = typeof payload.phase === 'string' ? payload.phase : 'running';
                updateProgressBar(processed, total);
                done = Boolean(payload.done);
                const nextOffsetRaw = payload.next_offset ?? (resolvedOffset + limit);
                const nextOffset = resolveOffset(nextOffsetRaw);
                if (done) {
                    break;
                }
                offset = nextOffset > resolvedOffset ? nextOffset : resolvedOffset + limit;
                await new Promise((resolve) => {
                    window.setTimeout(resolve, 300);
                });
            }
            state.refreshPhase = 'idle';
            state.detailCache.clear();
            await loadUpdates();
            await loadCacheStatus();
            showToast('IGDB refresh completed', 'success');
        } catch (error) {
            console.error(error);
            showToast(error.message, 'error');
        } finally {
            state.refreshing = false;
            state.refreshPhase = 'idle';
            state.refreshObservedActive = false;
            state.refreshPendingPolls = 0;
            state.waitingForRefreshStart = false;
            clearRefreshStatusTimer();
            setRefreshButtonLoading(false);
            setRefreshProgressVisible(false);
            updateRefreshProgress(0, 0, 'idle');
        }
    }

    async function handleCompare() {
        if (state.comparing) {
            return;
        }
        if (!config.compareUrl) {
            showToast('Comparison endpoint is not configured.', 'warning');
            return;
        }
        state.comparing = true;
        setCompareButtonLoading(true);
        setCompareProgressVisible(true);
        updateCompareProgress(0, 0);
        try {
            const job = await startJob(config.compareUrl);
            monitorJob(job);
        } catch (error) {
            console.error(error);
            showToast(error.message, 'warning');
            state.comparing = false;
            setCompareButtonLoading(false);
            updateCompareProgress(0, 0);
            setCompareProgressVisible(false);
        }
    }

    async function fetchUpdatesBatch(offset = DEFAULT_OFFSET, limit = DEFAULT_UPDATES_LIMIT) {
        const resolvedOffset = resolveOffset(offset);
        const resolvedLimit = resolveLimit(limit, DEFAULT_UPDATES_LIMIT);
        const url = `/api/updates?offset=${encodeURIComponent(resolvedOffset)}&limit=${encodeURIComponent(resolvedLimit)}`;
        logFetch(url);
        const payload = await fetchJson(url, { logRequest: false });
        if (Array.isArray(payload)) {
            return {
                items: payload,
                total: payload.length,
                nextOffset: resolvedOffset + payload.length,
            };
        }
        if (payload && typeof payload === 'object' && Array.isArray(payload.items)) {
            const total = toNonNegativeInt(
                payload.total ?? payload.total_available ?? payload.count ?? payload.total_items,
                payload.items.length,
            );
            const nextOffset = resolveOffset(payload.next_offset ?? (resolvedOffset + payload.items.length));
            return {
                items: payload.items,
                total,
                nextOffset,
            };
        }
        throw new Error(payload && payload.error ? payload.error : 'Failed to load updates.');
    }

    async function loadUpdates() {
        setLoading(true);
        try {
            const aggregated = [];
            const seenIds = new Set();
            let total = 0;
            let offset = DEFAULT_OFFSET;
            let batchLimit = state.pageSize || DEFAULT_UPDATES_LIMIT;
            const visitedOffsets = new Set();

            while (true) {
                const resolvedOffset = resolveOffset(offset);
                if (visitedOffsets.has(resolvedOffset)) {
                    break;
                }
                visitedOffsets.add(resolvedOffset);
                const normalizedLimit = resolveLimit(batchLimit, DEFAULT_UPDATES_LIMIT);
                const payload = await fetchUpdatesBatch(resolvedOffset, normalizedLimit);
                const items = Array.isArray(payload.items) ? payload.items : [];
                const normalizedTotal = toNonNegativeInt(payload.total, items.length);
                if (normalizedTotal > total) {
                    total = normalizedTotal;
                }
                items.forEach((item) => {
                    if (!item) {
                        return;
                    }
                    const key = item.processed_game_id;
                    if (key === undefined || key === null) {
                        aggregated.push(item);
                        return;
                    }
                    if (seenIds.has(key)) {
                        return;
                    }
                    seenIds.add(key);
                    aggregated.push(item);
                });

                const received = items.length;
                if (received === 0) {
                    break;
                }
                if (total > 0 && aggregated.length >= total) {
                    break;
                }
                const nextOffset = payload.nextOffset !== undefined
                    ? resolveOffset(payload.nextOffset)
                    : resolvedOffset + received;
                if (nextOffset <= resolvedOffset) {
                    offset = resolvedOffset + normalizedLimit;
                    break;
                }
                offset = nextOffset;
                batchLimit = normalizedLimit;
            }

            state.updates = aggregated;
            state.totalAvailable = total || aggregated.length;
            const previousCoverCache = state.coverCache instanceof Map ? state.coverCache : new Map();
            const newCoverCache = new Map();
            const mapEntries = [];
            state.updates.forEach((item) => {
                const normalizedId = normalizeId(item.processed_game_id);
                const key = normalizedId === null ? item.processed_game_id : normalizedId;
                mapEntries.push([key, item]);
                if (normalizedId === null) {
                    return;
                }
                if (item.cover) {
                    newCoverCache.set(normalizedId, item.cover);
                    return;
                }
                if (previousCoverCache.has(normalizedId)) {
                    const cached = previousCoverCache.get(normalizedId);
                    if (cached) {
                        item.cover = cached;
                    }
                    newCoverCache.set(normalizedId, cached);
                    return;
                }
                if (item.cover_available === false) {
                    newCoverCache.set(normalizedId, null);
                }
            });
            state.coverCache = newCoverCache;
            state.updateMap = new Map(mapEntries);
            state.page = 1;
            applyFilters({ resetPage: true });
            updateSortIndicators();
        } catch (error) {
            console.error(error);
            setLoading(false);
            showToast(error.message, 'warning');
            state.updates = [];
            state.filtered = [];
            state.filteredAll = [];
            state.pageCount = 1;
            state.page = 1;
            state.totalAvailable = 0;
            state.updateMap = new Map();
            state.coverCache = new Map();
            if (elements.tableBody) {
                elements.tableBody.innerHTML = '';
            }
            if (elements.emptyState) {
                elements.emptyState.hidden = false;
            }
            updateCount();
            renderPagination();
            updateStatusMessage();
            return;
        }
        setLoading(false);
    }

    function bindEvents() {
        if (elements.searchInput) {
            elements.searchInput.addEventListener('input', updateSearchTerm);
        }
        if (elements.refreshButton) {
            elements.refreshButton.addEventListener('click', handleRefresh);
        }
        if (elements.compareButton) {
            elements.compareButton.addEventListener('click', handleCompare);
        }
        if (elements.fixButton) {
            elements.fixButton.addEventListener('click', handleFixNames);
        }
        if (elements.dedupeButton) {
            elements.dedupeButton.addEventListener('click', handleRemoveDuplicates);
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
        if (elements.paginationPrev) {
            elements.paginationPrev.addEventListener('click', () => {
                setPage(state.page - 1);
            });
        }
        if (elements.paginationNext) {
            elements.paginationNext.addEventListener('click', () => {
                setPage(state.page + 1);
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
    loadExistingJobs();
    loadCacheStatus();
    loadUpdates();
})();
