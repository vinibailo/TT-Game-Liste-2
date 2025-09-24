(function () {
    const config = window.lookupsConfig || {};
    const tables = Array.isArray(config.tables) ? config.tables : [];
    const labelMap = new Map();
    const singularMap = new Map();

    tables.forEach((table) => {
        if (!table || typeof table !== 'object') {
            return;
        }
        const type = typeof table.type === 'string' ? table.type : '';
        if (!type) {
            return;
        }
        const label = typeof table.label === 'string' ? table.label : type;
        const singular = typeof table.singular_label === 'string' ? table.singular_label : label;
        labelMap.set(type, label);
        singularMap.set(type, singular);
    });

    let defaultType = typeof config.defaultType === 'string' ? config.defaultType : '';
    if (!defaultType && tables.length > 0) {
        const first = tables[0];
        if (first && typeof first.type === 'string') {
            defaultType = first.type;
        }
    }

    const state = {
        type: defaultType,
        items: [],
        filtered: [],
        search: '',
        editingId: null,
        deletingId: null,
    };

    const elements = {
        typeSelect: document.querySelector('[data-type-select]'),
        searchInput: document.querySelector('[data-search]'),
        newButton: document.querySelector('[data-new-entry]'),
        tableBody: document.querySelector('[data-lookup-rows]'),
        emptyState: document.querySelector('[data-empty-state]'),
        loadingState: document.querySelector('[data-loading-state]'),
        countLabel: document.querySelector('[data-entry-count]'),
        activeLabel: document.querySelector('[data-active-label]'),
        entryModal: document.querySelector('[data-entry-modal]'),
        entryModalTitle: document.getElementById('entry-modal-title'),
        entryModalSubtitle: document.querySelector('[data-entry-modal-subtitle]'),
        entryForm: document.querySelector('[data-entry-form]'),
        entryNameInput: document.querySelector('[data-entry-name]'),
        entrySubmit: document.querySelector('[data-entry-submit]'),
        entryCancel: document.querySelector('[data-cancel-entry]'),
        entryClose: document.querySelector('[data-close-entry-modal]'),
        confirmModal: document.querySelector('[data-confirm-modal]'),
        confirmSubtitle: document.querySelector('[data-delete-modal-subtitle]'),
        confirmName: document.querySelector('[data-confirm-name]'),
        confirmDelete: document.querySelector('[data-confirm-delete]'),
        confirmCancel: document.querySelector('[data-cancel-delete]'),
        confirmClose: document.querySelector('[data-close-confirm-modal]'),
    };

    const toast = document.getElementById('toast');
    let lastFocusedElement = null;

    function getLabelForType(type) {
        if (!type) {
            return '';
        }
        return labelMap.get(type) || type;
    }

    function getSingularLabel(type) {
        if (!type) {
            return 'entry';
        }
        return singularMap.get(type) || labelMap.get(type) || type;
    }

    function formatCount(count) {
        const suffix = count === 1 ? 'entry' : 'entries';
        return `${count} ${suffix}`;
    }

    function showToast(message, type = 'success') {
        if (!toast) {
            return;
        }
        const normalized = type === 'error' ? 'warning' : type;
        toast.textContent = message;
        toast.className = '';
        void toast.offsetWidth;
        toast.classList.add(normalized, 'show');
        window.clearTimeout(showToast.timerId);
        showToast.timerId = window.setTimeout(() => {
            toast.classList.remove('show');
        }, 3200);
    }

    function setActiveLabel() {
        if (!elements.activeLabel) {
            return;
        }
        elements.activeLabel.textContent = getLabelForType(state.type);
    }

    function updateCount() {
        if (!elements.countLabel) {
            return;
        }
        const count = state.filtered.length;
        elements.countLabel.textContent = formatCount(count);
    }

    function setLoading(loading) {
        if (elements.loadingState) {
            elements.loadingState.hidden = !loading;
        }
        if (elements.emptyState && loading) {
            elements.emptyState.hidden = true;
        }
        if (elements.tableBody) {
            elements.tableBody.setAttribute('aria-busy', loading ? 'true' : 'false');
        }
    }

    function applyFilters() {
        const term = state.search.trim().toLocaleLowerCase();
        const filtered = term
            ? state.items.filter((item) => item.name.toLocaleLowerCase().includes(term))
            : state.items.slice();
        state.filtered = filtered;
        updateCount();
        renderList();
    }

    function renderList() {
        if (!elements.tableBody) {
            return;
        }
        elements.tableBody.innerHTML = '';
        if (!state.filtered.length) {
            if (elements.emptyState) {
                elements.emptyState.hidden = false;
            }
            return;
        }
        if (elements.emptyState) {
            elements.emptyState.hidden = true;
        }
        const fragment = document.createDocumentFragment();
        state.filtered.forEach((entry) => {
            const row = document.createElement('tr');
            row.dataset.entryId = String(entry.id);

            const nameCell = document.createElement('td');
            const nameText = document.createElement('span');
            nameText.className = 'lookup-name';
            nameText.textContent = entry.name;
            nameCell.appendChild(nameText);
            row.appendChild(nameCell);

            const actionsCell = document.createElement('td');
            actionsCell.className = 'actions-cell';

            const editButton = document.createElement('button');
            editButton.type = 'button';
            editButton.className = 'icon-button';
            editButton.dataset.action = 'edit';
            editButton.dataset.entryId = String(entry.id);
            editButton.setAttribute('aria-label', `Edit ${entry.name || 'entry'}`);
            const editIcon = document.createElement('span');
            editIcon.className = 'material-symbols-rounded';
            editIcon.textContent = 'edit';
            editButton.appendChild(editIcon);

            const deleteButton = document.createElement('button');
            deleteButton.type = 'button';
            deleteButton.className = 'icon-button';
            deleteButton.dataset.action = 'delete';
            deleteButton.dataset.entryId = String(entry.id);
            deleteButton.setAttribute('aria-label', `Delete ${entry.name || 'entry'}`);
            const deleteIcon = document.createElement('span');
            deleteIcon.className = 'material-symbols-rounded';
            deleteIcon.textContent = 'delete';
            deleteButton.appendChild(deleteIcon);

            actionsCell.append(editButton, deleteButton);
            row.appendChild(actionsCell);
            fragment.appendChild(row);
        });
        elements.tableBody.appendChild(fragment);
    }

    async function fetchEntries() {
        if (!state.type) {
            state.items = [];
            applyFilters();
            return;
        }
        setLoading(true);
        try {
            const response = await fetch(`/api/lookups/${encodeURIComponent(state.type)}`);
            if (!response.ok) {
                throw new Error(`Failed to load entries (${response.status})`);
            }
            const payload = await response.json();
            const items = Array.isArray(payload.items) ? payload.items : [];
            const normalized = items
                .map((item) => {
                    const id = item && typeof item.id !== 'undefined' ? item.id : null;
                    const name =
                        item && typeof item.name === 'string' ? item.name.trim() : '';
                    return { id, name };
                })
                .filter((item) => item.id !== null && item.name !== '');
            normalized.sort((a, b) =>
                a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }),
            );
            state.items = normalized;
        } catch (error) {
            console.error(error);
            state.items = [];
            showToast('Failed to load entries.', 'error');
        } finally {
            setLoading(false);
            applyFilters();
        }
    }

    function openModal(backdrop, focusTarget) {
        if (!backdrop) {
            return;
        }
        lastFocusedElement = document.activeElement;
        backdrop.hidden = false;
        backdrop.classList.remove('hidden');
        document.body.classList.add('modal-open');
        if (focusTarget && typeof focusTarget.focus === 'function') {
            focusTarget.focus();
        }
    }

    function closeModal(backdrop) {
        if (!backdrop) {
            return;
        }
        backdrop.classList.add('hidden');
        backdrop.hidden = true;
        const otherOpenModal = Array.from(
            document.querySelectorAll('.modal-backdrop'),
        ).some((element) => element !== backdrop && !element.hidden);
        if (!otherOpenModal) {
            document.body.classList.remove('modal-open');
        }
        if (lastFocusedElement && typeof lastFocusedElement.focus === 'function') {
            lastFocusedElement.focus();
        }
        lastFocusedElement = null;
    }

    function resetEntryForm() {
        state.editingId = null;
        if (elements.entryForm) {
            elements.entryForm.reset();
        }
        if (elements.entryNameInput) {
            elements.entryNameInput.value = '';
        }
        if (elements.entryModalSubtitle) {
            elements.entryModalSubtitle.textContent = '';
        }
        if (elements.entrySubmit) {
            elements.entrySubmit.disabled = false;
            elements.entrySubmit.textContent = 'Save entry';
        }
    }

    function openEntryModal(entry) {
        if (!elements.entryModal || !elements.entrySubmit) {
            return;
        }
        const singular = getSingularLabel(state.type);
        state.editingId = entry ? entry.id : null;
        if (elements.entryModalTitle) {
            elements.entryModalTitle.textContent = entry
                ? `Edit ${singular}`
                : `New ${singular}`;
        }
        if (elements.entryModalSubtitle) {
            elements.entryModalSubtitle.textContent = entry
                ? `Update this ${singular.toLocaleLowerCase()} entry.`
                : `Create a new ${singular.toLocaleLowerCase()} entry.`;
        }
        if (elements.entrySubmit) {
            elements.entrySubmit.textContent = entry ? 'Save changes' : 'Create entry';
        }
        if (elements.entryNameInput) {
            elements.entryNameInput.value = entry ? entry.name : '';
        }
        openModal(elements.entryModal, elements.entryNameInput || elements.entrySubmit);
    }

    function closeEntryModal() {
        if (!elements.entryModal) {
            return;
        }
        resetEntryForm();
        closeModal(elements.entryModal);
    }

    function openDeleteModal(entry) {
        if (!elements.confirmModal || !elements.confirmDelete) {
            return;
        }
        state.deletingId = entry ? entry.id : null;
        const singular = getSingularLabel(state.type);
        if (elements.confirmSubtitle) {
            elements.confirmSubtitle.textContent = `Remove this ${singular.toLocaleLowerCase()} from the vocabulary.`;
        }
        if (elements.confirmName) {
            elements.confirmName.textContent = entry ? entry.name : '';
        }
        openModal(elements.confirmModal, elements.confirmDelete);
    }

    function closeDeleteModal() {
        if (!elements.confirmModal) {
            return;
        }
        state.deletingId = null;
        if (elements.confirmSubtitle) {
            elements.confirmSubtitle.textContent = '';
        }
        if (elements.confirmName) {
            elements.confirmName.textContent = '';
        }
        closeModal(elements.confirmModal);
    }

    async function handleSaveEntry(event) {
        event.preventDefault();
        if (!elements.entryNameInput || !elements.entrySubmit) {
            return;
        }
        const rawName = elements.entryNameInput.value.trim();
        if (!rawName) {
            elements.entryNameInput.focus();
            return;
        }
        const payload = { name: rawName };
        const submitButton = elements.entrySubmit;
        const originalText = submitButton.textContent;
        submitButton.disabled = true;
        submitButton.textContent = 'Saving…';
        try {
            let response;
            if (state.editingId !== null && state.editingId !== undefined) {
                response = await fetch(
                    `/api/lookups/${encodeURIComponent(state.type)}/${encodeURIComponent(
                        state.editingId,
                    )}`,
                    {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                    },
                );
            } else {
                response = await fetch(`/api/lookups/${encodeURIComponent(state.type)}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
            }
            if (!response || !response.ok) {
                throw new Error('Failed to save entry');
            }
            closeEntryModal();
            await fetchEntries();
            showToast('Entry saved successfully.');
        } catch (error) {
            console.error(error);
            showToast('Unable to save the entry.', 'error');
        } finally {
            submitButton.disabled = false;
            submitButton.textContent = originalText;
        }
    }

    async function handleConfirmDelete() {
        if (state.deletingId === null || state.deletingId === undefined) {
            closeDeleteModal();
            return;
        }
        if (!elements.confirmDelete) {
            return;
        }
        const button = elements.confirmDelete;
        const originalText = button.textContent;
        button.disabled = true;
        button.textContent = 'Deleting…';
        try {
            const response = await fetch(
                `/api/lookups/${encodeURIComponent(state.type)}/${encodeURIComponent(
                    state.deletingId,
                )}`,
                {
                    method: 'DELETE',
                },
            );
            if (!response || !response.ok) {
                throw new Error('Failed to delete entry');
            }
            closeDeleteModal();
            await fetchEntries();
            showToast('Entry deleted.');
        } catch (error) {
            console.error(error);
            showToast('Unable to delete the entry.', 'error');
        } finally {
            button.disabled = false;
            button.textContent = originalText;
        }
    }

    function handleTableClick(event) {
        if (!elements.tableBody) {
            return;
        }
        const button = event.target instanceof Element ? event.target.closest('button[data-action]') : null;
        if (!button) {
            return;
        }
        const entryId = button.dataset.entryId;
        if (!entryId) {
            return;
        }
        const entry = state.items.find((item) => String(item.id) === entryId);
        if (!entry) {
            return;
        }
        if (button.dataset.action === 'edit') {
            openEntryModal(entry);
        } else if (button.dataset.action === 'delete') {
            openDeleteModal(entry);
        }
    }

    function handleTypeChange(event) {
        const target = event.target;
        if (!(target instanceof HTMLSelectElement)) {
            return;
        }
        const { value } = target;
        if (!value || value === state.type) {
            return;
        }
        state.type = value;
        state.search = '';
        if (elements.searchInput) {
            elements.searchInput.value = '';
        }
        setActiveLabel();
        fetchEntries();
    }

    function handleSearchChange(event) {
        const target = event.target;
        if (!(target instanceof HTMLInputElement)) {
            return;
        }
        state.search = target.value;
        applyFilters();
    }

    function handleEscape(event) {
        if (event.key !== 'Escape') {
            return;
        }
        const entryModalOpen = elements.entryModal && !elements.entryModal.hidden;
        const confirmModalOpen = elements.confirmModal && !elements.confirmModal.hidden;
        if (confirmModalOpen) {
            closeDeleteModal();
        } else if (entryModalOpen) {
            closeEntryModal();
        }
    }

    function initialize() {
        if (elements.typeSelect && state.type) {
            elements.typeSelect.value = state.type;
        } else if (elements.typeSelect && !state.type) {
            const currentValue = elements.typeSelect.value;
            if (currentValue) {
                state.type = currentValue;
            }
        }
        setActiveLabel();
        updateCount();
        fetchEntries();

        if (elements.typeSelect) {
            elements.typeSelect.addEventListener('change', handleTypeChange);
        }
        if (elements.searchInput) {
            elements.searchInput.addEventListener('input', handleSearchChange);
        }
        if (elements.newButton) {
            elements.newButton.addEventListener('click', () => openEntryModal(null));
        }
        if (elements.tableBody) {
            elements.tableBody.addEventListener('click', handleTableClick);
        }
        if (elements.entryForm) {
            elements.entryForm.addEventListener('submit', handleSaveEntry);
        }
        if (elements.entryCancel) {
            elements.entryCancel.addEventListener('click', closeEntryModal);
        }
        if (elements.entryClose) {
            elements.entryClose.addEventListener('click', closeEntryModal);
        }
        if (elements.entryModal) {
            elements.entryModal.addEventListener('click', (event) => {
                if (event.target === elements.entryModal) {
                    closeEntryModal();
                }
            });
        }
        if (elements.confirmDelete) {
            elements.confirmDelete.addEventListener('click', handleConfirmDelete);
        }
        if (elements.confirmCancel) {
            elements.confirmCancel.addEventListener('click', closeDeleteModal);
        }
        if (elements.confirmClose) {
            elements.confirmClose.addEventListener('click', closeDeleteModal);
        }
        if (elements.confirmModal) {
            elements.confirmModal.addEventListener('click', (event) => {
                if (event.target === elements.confirmModal) {
                    closeDeleteModal();
                }
            });
        }
        document.addEventListener('keydown', handleEscape);
    }

    initialize();
})();
