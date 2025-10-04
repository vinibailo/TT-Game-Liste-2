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
        fixingNames: false,
        deduping: false,
        refreshing: false,
        comparing: false,
        jobs: new Map(),
        jobPollers: new Map(),
        jobPollBackoffs: new Map(),
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
        nextAfter: null,
        updatesEtag: null,
        activeModalId: null,
        modalSelections: new Map(),
        modalFieldNames: [],
        applyingUpdate: false,
    };

    const MAPPED_FIELD_NAMES = new Set(['Genres', 'Game Modes']);

    const JOB_TYPES = Object.freeze({
        refresh: 'refresh_updates',
        compare: 'compare_updates',
        fix: 'fix_names',
        dedupe: 'remove_duplicates',
    });
    const JOB_ACTIVE_STATUSES = new Set(['pending', 'running']);
    const JOB_POLL_INTERVAL = 2000;
    const JOB_POLL_BACKOFF_FACTOR = 2;
    const JOB_POLL_MAX_INTERVAL = 20000;

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

    const placeholderImage = '/static/no-image.jpg';

    const DEFAULT_OFFSET = 0;
    const DEFAULT_REFRESH_LIMIT = 200;
    const DEFAULT_UPDATES_LIMIT = 100;
    const LOCAL_STORAGE_KEYS = Object.freeze({
        updates: 'updates.cachedRows',
        lastSeen: 'updates.lastSeenUpdatedAt',
    });

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
        body: document.querySelector('[data-modal-body]'),
        gameId: document.querySelector('[data-modal-game-id]'),
        igdbId: document.querySelector('[data-modal-igdb-id]'),
        igdbUpdated: document.querySelector('[data-modal-igdb-updated]'),
        localEdited: document.querySelector('[data-modal-local-edited]'),
        empty: document.querySelector('[data-modal-empty]'),
        diffList: document.querySelector('[data-diff-list]'),
        cover: document.querySelector('[data-modal-cover]'),
        actions: null,
        applyButton: null,
        applyDefaultLabel: 'Apply',
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

    function getLocalStorage() {
        if (typeof window === 'undefined') {
            return null;
        }
        try {
            return window.localStorage || null;
        } catch (error) {
            console.debug('localStorage is unavailable', error);
            return null;
        }
    }

    function loadCachedUpdates() {
        const storage = getLocalStorage();
        if (!storage) {
            return null;
        }
        try {
            const raw = storage.getItem(LOCAL_STORAGE_KEYS.updates);
            if (!raw) {
                return null;
            }
            const parsed = JSON.parse(raw);
            if (!parsed || typeof parsed !== 'object') {
                return null;
            }
            const items = Array.isArray(parsed.items) ? parsed.items : [];
            const nextAfter = parsed.nextAfter ?? parsed.next_after ?? null;
            const etag = typeof parsed.etag === 'string' && parsed.etag.trim()
                ? parsed.etag.trim()
                : null;
            return {
                items,
                nextAfter,
                etag,
            };
        } catch (error) {
            console.debug('Failed to load cached updates', error);
            return null;
        }
    }

    function saveUpdatesCache(payload) {
        const storage = getLocalStorage();
        if (!storage) {
            return;
        }
        if (!payload || typeof payload !== 'object') {
            storage.removeItem(LOCAL_STORAGE_KEYS.updates);
            return;
        }
        const items = Array.isArray(payload.items) ? payload.items : [];
        const nextAfter = payload.nextAfter ?? null;
        const etag = typeof payload.etag === 'string' && payload.etag.trim() ? payload.etag.trim() : null;
        try {
            storage.setItem(
                LOCAL_STORAGE_KEYS.updates,
                JSON.stringify({
                    items,
                    nextAfter,
                    etag,
                }),
            );
        } catch (error) {
            console.debug('Failed to persist updates cache', error);
        }
    }

    function loadLastSeenUpdatedAt() {
        const storage = getLocalStorage();
        if (!storage) {
            return null;
        }
        try {
            const value = storage.getItem(LOCAL_STORAGE_KEYS.lastSeen);
            if (!value) {
                return null;
            }
            const trimmed = value.trim();
            return trimmed ? trimmed : null;
        } catch (error) {
            console.debug('Failed to load last seen timestamp', error);
            return null;
        }
    }

    function saveLastSeenUpdatedAt(value) {
        const storage = getLocalStorage();
        if (!storage) {
            return;
        }
        if (typeof value !== 'string' || !value.trim()) {
            storage.removeItem(LOCAL_STORAGE_KEYS.lastSeen);
            return;
        }
        try {
            storage.setItem(LOCAL_STORAGE_KEYS.lastSeen, value.trim());
        } catch (error) {
            console.debug('Failed to persist last seen timestamp', error);
        }
    }

    function clearUpdatesCache() {
        state.updatesEtag = null;
        const storage = getLocalStorage();
        if (!storage) {
            return;
        }
        try {
            storage.removeItem(LOCAL_STORAGE_KEYS.updates);
            storage.removeItem(LOCAL_STORAGE_KEYS.lastSeen);
        } catch (error) {
            console.debug('Failed to clear cached updates', error);
        }
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

    function normalizeNextAfter(value) {
        if (typeof value === 'number' && Number.isFinite(value)) {
            return Math.max(Math.trunc(value), 0);
        }
        if (typeof value === 'string' && value.trim()) {
            const parsed = Number.parseInt(value, 10);
            if (!Number.isNaN(parsed) && parsed >= 0) {
                return parsed;
            }
        }
        return null;
    }

    function resolveItemUpdatedAt(item) {
        if (!item || typeof item !== 'object') {
            return null;
        }
        const candidates = [
            item.updated_at,
            item.cursor_value,
            item.refreshed_at,
            item.local_last_edited_at,
            item.igdb_updated_at,
        ];
        for (const candidate of candidates) {
            if (typeof candidate === 'string' && candidate.trim()) {
                return candidate.trim();
            }
        }
        return null;
    }

    function getLatestUpdatedAt(items) {
        if (!Array.isArray(items) || !items.length) {
            return null;
        }
        let latest = null;
        items.forEach((item) => {
            const candidate = resolveItemUpdatedAt(item);
            if (!candidate) {
                return;
            }
            if (!latest || candidate > latest) {
                latest = candidate;
            }
        });
        return latest;
    }

    function normalizeJobId(value) {
        if (value === undefined || value === null) {
            return null;
        }
        let text = String(value).trim().toLowerCase();
        if (!text) {
            return null;
        }
        if (/^[0-9a-f]{32}s$/.test(text)) {
            text = text.slice(0, -1);
        }
        if (/^[0-9a-f]{32}$/.test(text)) {
            return text;
        }
        const match = text.match(/[0-9a-f]{32}/);
        return match ? match[0] : null;
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
        }
        state.refreshPendingPolls = 0;
        state.waitingForRefreshStart = false;
    }

    class HttpError extends Error {
        constructor(message, options = {}) {
            super(message);
            this.name = 'HttpError';
            this.status = options.status ?? null;
            this.payload = options.payload;
            this.response = options.response ?? null;
            this.isClientError = typeof this.status === 'number' && this.status >= 400 && this.status < 500;
            this.isServerError = typeof this.status === 'number' && this.status >= 500;
        }
    }

    async function fetchJson(url, options = {}) {
        const {
            logRequest = true,
            headers: providedHeaders,
            onNotModified,
            onResponse,
            ...rest
        } = options || {};
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
        const status = response.status;
        if (status === 504) {
            throw new HttpError('Server timed out (504). Please try again later.', { status, response });
        }
        if (status === 304) {
            if (typeof onResponse === 'function') {
                onResponse(response, null);
            }
            if (typeof onNotModified === 'function') {
                const fallback = await onNotModified({ response });
                if (fallback !== undefined) {
                    return fallback;
                }
            }
            return null;
        }
        const contentType = (response.headers && response.headers.get('content-type')) || '';
        const isJson = contentType.toLowerCase().includes('application/json');
        if (response.ok && !isJson && (response.status === 204 || response.status === 205)) {
            if (typeof onResponse === 'function') {
                onResponse(response, null);
            }
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
            throw new HttpError(message, { status, response });
        }
        let payload;
        try {
            payload = await response.json();
        } catch (err) {
            throw new HttpError('Invalid JSON response.', { status, response });
        }
        if (typeof onResponse === 'function') {
            onResponse(response, payload);
        }
        if (response.ok) {
            return payload;
        }
        if (payload && typeof payload === 'object' && payload.error) {
            throw new HttpError(String(payload.error), { status, response, payload });
        }
        if (payload && typeof payload === 'object' && typeof payload.message === 'string') {
            throw new HttpError(payload.message, { status, response, payload });
        }
        throw new HttpError(`Request failed with status ${status}`, { status, response, payload });
    }

    class ProgressStream {
        constructor(url, options = {}) {
            this.url = url;
            this.eventSource = null;
            this.onMessage = options.onMessage || null;
            this.onError = options.onError || null;
            this.onOpen = options.onOpen || null;
        }

        start() {
            if (typeof window === 'undefined' || !('EventSource' in window)) {
                return;
            }
            if (this.eventSource) {
                return;
            }
            try {
                this.eventSource = new EventSource(this.url);
            } catch (error) {
                console.error('Failed to open progress stream', error);
                if (typeof this.onError === 'function') {
                    this.onError(error);
                }
                return;
            }
            if (typeof this.onMessage === 'function') {
                this.eventSource.onmessage = (event) => {
                    this.onMessage(event);
                };
            }
            if (typeof this.onError === 'function') {
                this.eventSource.onerror = (event) => {
                    this.onError(event);
                };
            }
            if (typeof this.onOpen === 'function') {
                this.eventSource.onopen = (event) => {
                    this.onOpen(event);
                };
            }
        }

        stop() {
            if (this.eventSource) {
                this.eventSource.close();
                this.eventSource = null;
            }
        }
    }

    function getJobDetailUrl(jobId) {
        if (!jobId) {
            return null;
        }
        return `/api/updates/${encodeURIComponent(jobId)}`;
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
        if (!id) {
            return null;
        }
        if (config.deleteDuplicateBaseUrl) {
            return `${config.deleteDuplicateBaseUrl}/${encodeURIComponent(id)}`;
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
        state.jobPollBackoffs.delete(jobId);
    }

    function resolveJobType(jobId) {
        if (!jobId) {
            return null;
        }
        const existing = state.jobs.get(jobId);
        if (existing && existing.job_type) {
            return existing.job_type;
        }
        for (const [type, activeId] of state.activeJobIds.entries()) {
            if (activeId === jobId) {
                return type;
            }
        }
        return null;
    }

    function createFailedJob(jobId, error, fallbackMessage = 'Background task failed.') {
        const existing = state.jobs.get(jobId);
        const base = existing ? { ...existing } : {};
        let message = fallbackMessage;
        if (error instanceof HttpError && error.status === 404) {
            message = 'Background job could not be found.';
        } else if (error && typeof error.message === 'string' && error.message.trim()) {
            message = error.message.trim();
        }
        const resolvedType = base.job_type || resolveJobType(jobId);
        return {
            ...base,
            id: jobId,
            job_type: resolvedType || null,
            status: 'error',
            error: message,
            message,
        };
    }

    function hasJobProgressChanged(previousJob, nextJob) {
        if (!previousJob || !nextJob) {
            return true;
        }
        if (previousJob.status !== nextJob.status) {
            return true;
        }
        const prevCurrent = Number(previousJob.progress_current) || 0;
        const nextCurrent = Number(nextJob.progress_current) || 0;
        if (prevCurrent !== nextCurrent) {
            return true;
        }
        const prevTotal = Number(previousJob.progress_total) || 0;
        const nextTotal = Number(nextJob.progress_total) || 0;
        if (prevTotal !== nextTotal) {
            return true;
        }
        const prevMessage = (previousJob.message || '').trim();
        const nextMessage = (nextJob.message || '').trim();
        return prevMessage !== nextMessage;
    }

    async function fetchJobDetail(jobId) {
        const url = getJobDetailUrl(jobId);
        if (!url) {
            throw new Error('Job tracking endpoint is not configured.');
        }
        try {
            const payload = await fetchJson(url);
            if (!payload || !payload.job) {
                throw new Error('Failed to retrieve job details.');
            }
            return payload.job;
        } catch (error) {
            console.error('Failed to fetch job detail', error);
            return createFailedJob(jobId, error, 'Failed to retrieve job details.');
        }
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
        const normalizedId = normalizeJobId(job.id);
        if (!normalizedId) {
            return;
        }
        const resolvedJob = normalizedId !== job.id ? { ...job, id: normalizedId } : job;
        state.jobs.set(normalizedId, resolvedJob);
        const type = job.job_type;
        const active = isJobActive(job.status);
        if (type) {
            if (active) {
                state.activeJobIds.set(type, normalizedId);
            } else if (state.activeJobIds.get(type) === normalizedId) {
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
                    updateRefreshProgress(
                        job.progress_current || 0,
                        job.progress_total || 0,
                        phase,
                        { message: job.message },
                    );
                } else {
                    state.refreshPhase = 'idle';
                    state.refreshObservedActive = false;
                    state.refreshPendingPolls = 0;
                    state.waitingForRefreshStart = false;
                    setRefreshButtonLoading(false);
                    setRefreshProgressVisible(false);
                    updateRefreshProgress(0, 0, 'idle');
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
        if (!job || !job.id) {
            return;
        }
        const normalizedId = normalizeJobId(job.id);
        if (!normalizedId || state.completedJobs.has(normalizedId)) {
            return;
        }
        const resolvedJob = normalizedId !== job.id ? { ...job, id: normalizedId } : job;
        state.completedJobs.add(normalizedId);
        clearJobPoller(normalizedId);
        const type = resolvedJob.job_type;
        const status = resolvedJob.status;
        const result = resolvedJob.result || {};
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
            showToast(resolvedJob.error || 'Background task failed.', 'warning');
        } else if (result && result.message) {
            showToast(result.message, result.toast_type || 'success');
        }
    }

    async function monitorJob(job) {
        if (!job || !job.id) {
            return;
        }
        const normalizedId = normalizeJobId(job.id);
        if (!normalizedId) {
            return;
        }
        const resolvedJob = normalizedId !== job.id ? { ...job, id: normalizedId } : job;
        updateJobUi(resolvedJob);
        if (!isJobActive(resolvedJob.status)) {
            handleJobCompletion(resolvedJob);
            return;
        }
        clearJobPoller(normalizedId);
        const poll = async () => {
            try {
                const latest = await fetchJobDetail(normalizedId);
                const previous = state.jobs.get(normalizedId) || null;
                updateJobUi(latest);
                if (isJobActive(latest.status)) {
                    let delay = state.jobPollBackoffs.get(normalizedId);
                    if (!Number.isFinite(delay) || delay < JOB_POLL_INTERVAL) {
                        delay = JOB_POLL_INTERVAL;
                    } else {
                        delay = Math.min(delay, JOB_POLL_MAX_INTERVAL);
                    }
                    if (hasJobProgressChanged(previous, latest)) {
                        delay = JOB_POLL_INTERVAL;
                    }
                    const nextDelay = Math.min(Math.max(delay * JOB_POLL_BACKOFF_FACTOR, JOB_POLL_INTERVAL), JOB_POLL_MAX_INTERVAL);
                    const timer = window.setTimeout(poll, delay);
                    state.jobPollers.set(normalizedId, timer);
                    state.jobPollBackoffs.set(normalizedId, nextDelay);
                } else {
                    state.jobPollers.delete(normalizedId);
                    handleJobCompletion(latest);
                }
            } catch (error) {
                console.error('Failed to poll job status', error);
                state.jobPollers.delete(normalizedId);
                const failedJob = createFailedJob(normalizedId, error, 'Failed to poll job status.');
                updateJobUi(failedJob);
                handleJobCompletion(failedJob);
            }
        };
        const timer = window.setTimeout(poll, JOB_POLL_INTERVAL);
        state.jobPollers.set(normalizedId, timer);
        state.jobPollBackoffs.set(normalizedId, JOB_POLL_INTERVAL);
    }

    function handleProgressStreamMessage(event) {
        if (!event || typeof event.data !== 'string' || !event.data) {
            return;
        }
        let payload;
        try {
            payload = JSON.parse(event.data);
        } catch (error) {
            console.error('Failed to parse progress update', error);
            return;
        }
        if (!payload || typeof payload !== 'object') {
            return;
        }
        const jobsPayload = [];
        if (Array.isArray(payload.jobs)) {
            payload.jobs.forEach((job) => {
                if (job && typeof job === 'object') {
                    jobsPayload.push(job);
                }
            });
        } else if (payload.jobs && typeof payload.jobs === 'object') {
            Object.values(payload.jobs).forEach((job) => {
                if (job && typeof job === 'object') {
                    jobsPayload.push(job);
                }
            });
        }
        jobsPayload.forEach((job) => {
            if (!job || !job.id) {
                return;
            }
            const normalizedId = normalizeJobId(job.id);
            if (!normalizedId) {
                return;
            }
            const resolvedJob = normalizedId !== job.id ? { ...job, id: normalizedId } : job;
            updateJobUi(resolvedJob);
        });
    }

    function handleProgressStreamError(event) {
        console.debug('Progress stream error', event);
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
                const normalizedId = normalizeJobId(job.id);
                if (!normalizedId) {
                    return;
                }
                const resolvedJob = normalizedId !== job.id ? { ...job, id: normalizedId } : job;
                if (job.job_type === JOB_TYPES.refresh) {
                    if (isJobActive(job.status)) {
                        state.completedJobs.delete(normalizedId);
                    } else {
                        state.completedJobs.add(normalizedId);
                    }
                    updateJobUi(resolvedJob);
                    return;
                }
                if (isJobActive(job.status)) {
                    state.completedJobs.delete(normalizedId);
                    monitorJob(resolvedJob);
                } else {
                    state.completedJobs.add(normalizedId);
                    updateJobUi(resolvedJob);
                }
            });
        } catch (error) {
            console.error('Failed to load background jobs', error);
        }
    }

    function buildDetailUrl(id) {
        if (config.detailBaseUrl) {
            return `${config.detailBaseUrl}/${encodeURIComponent(id)}`;
        }
        return `/api/updates/${encodeURIComponent(id)}`;
    }

    function buildApplyUrl(id) {
        if (id === undefined || id === null) {
            return null;
        }
        const normalized = normalizeId(id);
        const resolvedId = normalized !== null ? normalized : id;
        if (config.detailBaseUrl) {
            return `${config.detailBaseUrl}/${encodeURIComponent(resolvedId)}/apply`;
        }
        return `/api/updates/${encodeURIComponent(resolvedId)}/apply`;
    }

    function ensureCoverImageById(id, imgElement, fallbackItem) {
        if (!(imgElement instanceof HTMLImageElement)) {
            return;
        }
        const normalizedId = normalizeId(id);
        if (normalizedId !== null) {
            imgElement.dataset.coverId = String(normalizedId);
        }
        const update =
            (normalizedId !== null ? getUpdateById(normalizedId) : null) ||
            fallbackItem ||
            null;
        const source = update && update.cover_url ? update.cover_url : null;
        imgElement.src = resolveCoverSource(source);
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
        const id = item.processed_game_id;
        if (button) {
            button.disabled = true;
            button.setAttribute('aria-busy', 'true');
        }
        try {
            const url = buildDeleteDuplicateUrl(id);
            if (!url) {
                showToast('Duplicate removal endpoint is not configured.', 'warning');
                return;
            }
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
        state.activeModalId = null;
        state.modalSelections = new Map();
        state.modalFieldNames = [];
        state.applyingUpdate = false;
        setModalInputsDisabled(false);
        updateApplyButtonState();
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

    function ensureModalActions() {
        if (!modal.body) {
            return;
        }
        if (modal.actions && modal.applyButton) {
            return;
        }
        const actions = document.createElement('div');
        actions.className = 'modal-actions';
        actions.hidden = true;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-blue';
        button.textContent = modal.applyDefaultLabel || 'Apply';
        button.setAttribute('aria-disabled', 'true');
        actions.appendChild(button);
        modal.body.appendChild(actions);
        modal.actions = actions;
        modal.applyButton = button;
        modal.applyDefaultLabel = button.textContent;
        button.addEventListener('click', handleApplySelections);
    }

    function slugifyFieldName(field) {
        return String(field || '')
            .trim()
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '-')
            .replace(/^-+|-+$/g, '')
            .slice(0, 40) || 'field';
    }

    function normalizeFieldSelection(field, value) {
        const action = typeof value === 'string' ? value.trim().toLowerCase() : '';
        if (action === 'from_cache_mapped') {
            return MAPPED_FIELD_NAMES.has(field) ? 'from_cache_mapped' : 'from_cache';
        }
        if (action === 'from_cache') {
            return 'from_cache';
        }
        if (action === 'keep_current') {
            return 'keep_current';
        }
        return 'keep_current';
    }

    function buildFieldSelection(field, index, selection) {
        const container = document.createElement('div');
        container.className = 'diff-selection';
        container.setAttribute('role', 'radiogroup');
        container.setAttribute('aria-label', `Update preference for ${field}`);
        const controlName = `diff-choice-${state.activeModalId || 'item'}-${slugifyFieldName(field)}-${index}`;
        const options = [
            {
                value: 'keep_current',
                label: 'Keep current value',
            },
            {
                value: MAPPED_FIELD_NAMES.has(field) ? 'from_cache_mapped' : 'from_cache',
                label: MAPPED_FIELD_NAMES.has(field)
                    ? 'Use IGDB value (mapped)'
                    : 'Use IGDB value',
            },
        ];
        options.forEach((option) => {
            const optionId = `${controlName}-${option.value}`;
            const label = document.createElement('label');
            label.className = 'diff-option';
            const input = document.createElement('input');
            input.type = 'radio';
            input.name = controlName;
            input.id = optionId;
            input.value = option.value;
            input.dataset.fieldName = field;
            input.checked = selection === option.value;
            input.disabled = state.applyingUpdate;
            input.addEventListener('change', () => {
                if (input.checked) {
                    state.modalSelections.set(field, normalizeFieldSelection(field, option.value));
                    updateApplyButtonState();
                }
            });
            const text = document.createElement('span');
            text.textContent = option.label;
            label.appendChild(input);
            label.appendChild(text);
            container.appendChild(label);
        });
        return container;
    }

    function buildDiffSection(field, payload, index, selection) {
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
        section.appendChild(buildFieldSelection(field, index, selection));
        return section;
    }

    function updateApplyButtonState() {
        if (!modal.actions || !modal.applyButton) {
            return;
        }
        const hasFields = state.activeModalId !== null && state.modalFieldNames.length > 0;
        modal.actions.hidden = !hasFields;
        const button = modal.applyButton;
        if (!hasFields) {
            button.disabled = true;
            button.setAttribute('aria-disabled', 'true');
            button.removeAttribute('aria-busy');
            button.textContent = modal.applyDefaultLabel || 'Apply';
            return;
        }
        let hasSelection = false;
        state.modalFieldNames.forEach((field) => {
            const value = state.modalSelections.get(field) || 'keep_current';
            if (value !== 'keep_current') {
                hasSelection = true;
            }
        });
        const shouldDisable = !hasSelection || state.applyingUpdate;
        button.disabled = shouldDisable;
        button.setAttribute('aria-disabled', shouldDisable ? 'true' : 'false');
        if (state.applyingUpdate) {
            button.textContent = 'Applying…';
            button.setAttribute('aria-busy', 'true');
        } else {
            button.textContent = modal.applyDefaultLabel || 'Apply';
            button.removeAttribute('aria-busy');
        }
    }

    function setModalInputsDisabled(disabled) {
        if (!modal.diffList) {
            return;
        }
        const inputs = modal.diffList.querySelectorAll('input[type="radio"]');
        inputs.forEach((input) => {
            input.disabled = disabled;
        });
    }

    function renderDiff(diff) {
        ensureModalActions();
        if (!modal.diffList || !modal.empty) {
            return;
        }
        modal.diffList.innerHTML = '';
        modal.empty.textContent = modalEmptyDefaultText;
        const entries = Object.entries(diff || {});
        if (!entries.length) {
            modal.empty.hidden = false;
            state.modalSelections = new Map();
            state.modalFieldNames = [];
            updateApplyButtonState();
            return;
        }
        modal.empty.hidden = true;
        entries.sort((a, b) => a[0].localeCompare(b[0]));
        const previousSelections =
            state.modalSelections instanceof Map ? state.modalSelections : new Map();
        const nextSelections = new Map();
        const fragment = document.createDocumentFragment();
        entries.forEach(([field, payload], index) => {
            const selection = normalizeFieldSelection(field, previousSelections.get(field));
            nextSelections.set(field, selection);
            fragment.appendChild(buildDiffSection(field, payload, index, selection));
        });
        state.modalSelections = nextSelections;
        state.modalFieldNames = Array.from(nextSelections.keys());
        modal.diffList.appendChild(fragment);
        updateApplyButtonState();
        setModalInputsDisabled(state.applyingUpdate);
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

    function applyDetailToUpdate(detail) {
        if (!detail || typeof detail !== 'object') {
            return null;
        }
        const normalizedId = normalizeId(detail.processed_game_id);
        const lookupId = normalizedId !== null ? normalizedId : detail.processed_game_id;
        const update = getUpdateById(lookupId);
        if (!update) {
            return null;
        }
        if (Object.prototype.hasOwnProperty.call(detail, 'processed_game_id')) {
            update.processed_game_id = detail.processed_game_id;
        }
        if (Object.prototype.hasOwnProperty.call(detail, 'igdb_id')) {
            update.igdb_id = detail.igdb_id;
        }
        if (Object.prototype.hasOwnProperty.call(detail, 'igdb_updated_at')) {
            update.igdb_updated_at = detail.igdb_updated_at;
        }
        if (Object.prototype.hasOwnProperty.call(detail, 'local_last_edited_at')) {
            update.local_last_edited_at = detail.local_last_edited_at;
        }
        if (Object.prototype.hasOwnProperty.call(detail, 'refreshed_at')) {
            update.refreshed_at = detail.refreshed_at;
        }
        if (Object.prototype.hasOwnProperty.call(detail, 'name')) {
            update.name = detail.name || update.name;
        }
        if (Object.prototype.hasOwnProperty.call(detail, 'cover_url')) {
            update.cover_url = detail.cover_url || null;
        }
        if (Object.prototype.hasOwnProperty.call(detail, 'cover_available')) {
            update.cover_available = detail.cover_available !== false;
        }
        if (Object.prototype.hasOwnProperty.call(detail, 'diff')) {
            const diffPayload = detail.diff || {};
            update.has_diff = Boolean(diffPayload && Object.keys(diffPayload).length);
        }
        update.detail_available = detail.detail_available !== false;
        return update;
    }

    function applyDetailToModal(detail, fallbackUpdate = null) {
        if (!detail || typeof detail !== 'object') {
            if (modal.diffList) {
                modal.diffList.innerHTML = '';
            }
            if (modal.empty) {
                modal.empty.textContent = 'Failed to load changes for this game.';
                modal.empty.hidden = false;
            }
            state.modalSelections = new Map();
            state.modalFieldNames = [];
            updateApplyButtonState();
            return;
        }
        const normalizedId = normalizeId(detail.processed_game_id);
        if (normalizedId !== null) {
            state.activeModalId = normalizedId;
        }
        const update =
            fallbackUpdate ||
            getUpdateById(normalizedId !== null ? normalizedId : detail.processed_game_id) ||
            null;
        setMetaValue(modal.gameId, detail.processed_game_id);
        setMetaValue(modal.igdbId, detail.igdb_id);
        setMetaValue(modal.igdbUpdated, formatDate(detail.igdb_updated_at));
        setMetaValue(modal.localEdited, formatDate(detail.local_last_edited_at));
        const resolvedName = detail.name || (update && update.name) || '';
        const coverSource = detail.cover_url || (update && update.cover_url) || null;
        setModalCover(coverSource, resolvedName);
        ensureCoverImageById(detail.processed_game_id, modal.cover, update || detail);
        if (modal.subtitle) {
            const refreshed = formatDate(detail.refreshed_at);
            if (resolvedName && refreshed) {
                modal.subtitle.textContent = `${resolvedName} • Refreshed ${refreshed}`;
            } else if (resolvedName) {
                modal.subtitle.textContent = resolvedName;
            } else if (refreshed) {
                modal.subtitle.textContent = `Refreshed ${refreshed}`;
            } else {
                modal.subtitle.textContent = '';
            }
        }
        state.modalSelections = new Map();
        state.modalFieldNames = [];
        renderDiff(detail.diff);
    }

    async function handleApplySelections() {
        if (state.applyingUpdate) {
            return;
        }
        const processedId = state.activeModalId;
        if (processedId === null) {
            showToast('Select a game to apply updates first.', 'warning');
            return;
        }
        if (!Array.isArray(state.modalFieldNames) || !state.modalFieldNames.length) {
            showToast('No IGDB fields are available to apply.', 'warning');
            return;
        }
        const fieldsPayload = {};
        let hasSelection = false;
        state.modalFieldNames.forEach((field) => {
            const value = state.modalSelections.get(field) || 'keep_current';
            fieldsPayload[field] = value;
            if (value !== 'keep_current') {
                hasSelection = true;
            }
        });
        if (!hasSelection) {
            showToast('Select at least one IGDB value to apply.', 'warning');
            return;
        }
        const url = buildApplyUrl(processedId);
        if (!url) {
            showToast('Apply endpoint is not configured.', 'warning');
            return;
        }
        state.applyingUpdate = true;
        updateApplyButtonState();
        setModalInputsDisabled(true);
        try {
            const payload = await fetchJson(url, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fields: fieldsPayload }),
            });
            let lastEdited = null;
            if (payload && typeof payload === 'object') {
                const result = payload.result && typeof payload.result === 'object' ? payload.result : null;
                const candidate =
                    (result && (result.last_edited_at || result.lastEditedAt)) ||
                    payload.last_edited_at ||
                    payload.lastEditedAt ||
                    null;
                if (typeof candidate === 'string' && candidate.trim()) {
                    lastEdited = candidate.trim();
                }
            }
            const update = getUpdateById(processedId);
            if (update && lastEdited) {
                update.local_last_edited_at = lastEdited;
            }
            state.detailCache.delete(processedId);
            state.detailCache.delete(String(processedId));
            const detail = await fetchDiffDetail(processedId);
            const appliedUpdate = applyDetailToUpdate(detail) || update;
            applyDetailToModal(detail, appliedUpdate || update);
            applyFilters();
            persistUpdatesCache();
            showToast('Applied selected IGDB fields.', 'success');
        } catch (error) {
            console.error('Failed to apply IGDB fields', error);
            const message = error && error.message ? error.message : 'Failed to apply selected IGDB values.';
            showToast(message, 'warning');
        } finally {
            state.applyingUpdate = false;
            setModalInputsDisabled(false);
            updateApplyButtonState();
        }
    }

    async function openDiffModal(id) {
        const update = getUpdateById(id);
        if (!update) {
            showToast('Unable to find update details for that game.', 'warning');
            return;
        }
        const normalizedId = normalizeId(id);
        let activeId = normalizedId;
        if (activeId === null) {
            const processedIdentifier = update.processed_game_id;
            const normalizedProcessed = normalizeId(processedIdentifier);
            if (normalizedProcessed !== null) {
                activeId = normalizedProcessed;
            } else if (processedIdentifier !== undefined && processedIdentifier !== null) {
                activeId = processedIdentifier;
            } else if (id !== undefined && id !== null) {
                activeId = id;
            }
        }
        state.activeModalId = activeId;
        state.modalSelections = new Map();
        state.modalFieldNames = [];
        state.applyingUpdate = false;
        updateApplyButtonState();
        setModalInputsDisabled(false);
        showModalShell();
        setModalCover(update.cover_url, update.name);
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
            const appliedUpdate = applyDetailToUpdate(detail) || update;
            applyDetailToModal(detail, appliedUpdate || update);
        } catch (error) {
            console.error(error);
            if (modal.empty) {
                modal.empty.textContent = 'Failed to load changes for this update.';
                modal.empty.hidden = false;
            }
            showToast(error.message, 'warning');
            state.modalSelections = new Map();
            state.modalFieldNames = [];
            updateApplyButtonState();
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

    async function fetchUpdatesBatch({
        after = null,
        limit = DEFAULT_UPDATES_LIMIT,
        since = null,
    } = {}) {
        const resolvedLimit = resolveLimit(limit, DEFAULT_UPDATES_LIMIT);
        const params = new URLSearchParams({ limit: String(resolvedLimit) });
        if (after !== null && after !== undefined) {
            const parsedAfter = Number.parseInt(after, 10);
            if (!Number.isNaN(parsedAfter) && parsedAfter >= 0) {
                params.set('after', String(parsedAfter));
            }
        }
        if (since && typeof since === 'string' && since.trim()) {
            params.set('since', since.trim());
        }
        const url = `/api/updates?${params.toString()}`;
        logFetch(url);
        const requestHeaders = {};
        if (typeof state.updatesEtag === 'string' && state.updatesEtag.trim()) {
            requestHeaders['If-None-Match'] = state.updatesEtag;
        }
        let responseEtag = null;
        const payload = await fetchJson(url, {
            logRequest: false,
            headers: requestHeaders,
            onResponse: (response) => {
                if (response && response.headers) {
                    const headerValue = response.headers.get('etag');
                    if (headerValue && headerValue.trim()) {
                        responseEtag = headerValue.trim();
                    }
                }
            },
            onNotModified: () => {
                const cached = loadCachedUpdates();
                if (cached && Array.isArray(cached.items)) {
                    if (!responseEtag) {
                        responseEtag = cached.etag && cached.etag.trim() ? cached.etag.trim() : state.updatesEtag;
                    }
                    return {
                        items: cached.items.slice(),
                        nextAfter: cached.nextAfter ?? null,
                    };
                }
                if (!responseEtag && state.updatesEtag) {
                    responseEtag = state.updatesEtag;
                }
                return {
                    items: Array.isArray(state.updates) ? state.updates.slice() : [],
                    nextAfter: state.nextAfter ?? null,
                };
            },
        });
        if (!payload || typeof payload !== 'object' || !Array.isArray(payload.items)) {
            throw new Error(payload && payload.error ? payload.error : 'Failed to load updates.');
        }
        state.updatesEtag = responseEtag && responseEtag.trim() ? responseEtag.trim() : null;
        const nextAfterRaw = payload.nextAfter ?? payload.next_after ?? null;
        const nextAfter = normalizeNextAfter(nextAfterRaw);
        const hasMore =
            payload.has_more ??
            payload.hasMore ??
            (typeof payload.next_cursor === 'string' && payload.next_cursor ? true : undefined);
        return {
            items: payload.items,
            nextAfter,
            nextCursor:
                typeof payload.next_cursor === 'string' && payload.next_cursor.trim()
                    ? payload.next_cursor.trim()
                    : null,
            limit: payload.limit,
            total: payload.total,
            hasMore,
            etag: state.updatesEtag,
        };
    }

    function rebuildUpdatesState({ resetPage = true } = {}) {
        const mapEntries = [];
        state.updates.forEach((item) => {
            const normalizedId = normalizeId(item && item.processed_game_id);
            const key = normalizedId === null ? item && item.processed_game_id : normalizedId;
            if (key !== undefined && key !== null) {
                mapEntries.push([key, item]);
            }
        });
        state.updateMap = new Map(mapEntries);
        state.totalAvailable = state.updates.length;
        if (resetPage) {
            state.page = 1;
        }
        applyFilters({ resetPage });
        updateSortIndicators();
    }

    async function loadUpdates({ append = false, since = null, preserveExisting = false } = {}) {
        setLoading(true);
        try {
            if (!append && !preserveExisting) {
                state.updates = [];
                state.nextAfter = null;
            }

            const limit = state.pageSize || DEFAULT_UPDATES_LIMIT;
            const after = append ? state.nextAfter : null;
            const payload = await fetchUpdatesBatch({
                after,
                limit,
                since: !append ? since : null,
            });
            const items = Array.isArray(payload.items) ? payload.items : [];

            const existingIds = new Set();
            state.updates.forEach((item) => {
                if (!item) {
                    return;
                }
                const rowId = normalizeId(item.id ?? item.row_id ?? item.processed_game_id);
                if (rowId !== null) {
                    existingIds.add(rowId);
                }
            });

            items.forEach((item) => {
                if (!item) {
                    return;
                }
                const rowId = normalizeId(item.id ?? item.row_id ?? item.processed_game_id);
                if (rowId !== null && existingIds.has(rowId)) {
                    return;
                }
                if (rowId !== null && !Object.prototype.hasOwnProperty.call(item, 'id')) {
                    item.id = rowId;
                }
                if (rowId !== null) {
                    existingIds.add(rowId);
                }
                state.updates.push(item);
            });

            const normalizedNextAfter = normalizeNextAfter(payload.nextAfter);
            if (normalizedNextAfter !== null) {
                state.nextAfter = normalizedNextAfter;
            } else if (!append && !preserveExisting) {
                state.nextAfter = null;
            }

            rebuildUpdatesState({ resetPage: !append });
            persistUpdatesCache();
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
            state.nextAfter = null;
            state.updatesEtag = null;
            if (elements.tableBody) {
                elements.tableBody.innerHTML = '';
            }
            if (elements.emptyState) {
                elements.emptyState.hidden = false;
            }
            updateCount();
            renderPagination();
            updateStatusMessage();
            clearUpdatesCache();
            return;
        }
        setLoading(false);
    }

    function persistUpdatesCache() {
        saveUpdatesCache({
            items: state.updates,
            nextAfter: state.nextAfter,
            etag: state.updatesEtag,
        });
        const latest = getLatestUpdatedAt(state.updates);
        if (latest) {
            saveLastSeenUpdatedAt(latest);
        } else {
            saveLastSeenUpdatedAt('');
        }
    }

    function restoreUpdatesFromCache() {
        const cached = loadCachedUpdates();
        if (!cached) {
            state.updatesEtag = null;
            return { hasItems: false, latestUpdatedAt: null };
        }
        state.updatesEtag = typeof cached.etag === 'string' && cached.etag ? cached.etag : null;
        state.updates = Array.isArray(cached.items) ? cached.items.slice() : [];
        state.nextAfter = normalizeNextAfter(cached.nextAfter);
        rebuildUpdatesState({ resetPage: true });
        return {
            hasItems: state.updates.length > 0,
            latestUpdatedAt: getLatestUpdatedAt(state.updates),
        };
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

    let progressStream = null;
    if (typeof window !== 'undefined' && 'EventSource' in window) {
        progressStream = new ProgressStream('/api/progress/stream', {
            onMessage: handleProgressStreamMessage,
            onError: handleProgressStreamError,
        });
        progressStream.start();
        window.addEventListener('beforeunload', () => {
            if (progressStream) {
                progressStream.stop();
            }
        });
    } else {
        console.debug('EventSource not supported; progress updates will fall back to manual polling.');
    }

    loadExistingJobs();
    loadCacheStatus();
    const cachedState = restoreUpdatesFromCache();
    const storedLastSeen = loadLastSeenUpdatedAt();
    if (!storedLastSeen && cachedState.latestUpdatedAt) {
        saveLastSeenUpdatedAt(cachedState.latestUpdatedAt);
    }
    const initialSince = cachedState.hasItems && storedLastSeen ? storedLastSeen : null;
    loadUpdates({ since: initialSince, preserveExisting: cachedState.hasItems });
})();
