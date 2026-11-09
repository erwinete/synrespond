<script lang="ts">
	import { createEventDispatcher, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import Modal from '$lib/components/common/Modal.svelte';
	import SynIcon from '../../synrespond/SynIcon.svelte';
	import SynEmoji from '../../synrespond/SynEmoji.svelte';
	import '../../synrespond/synrespond-tokens.css';
	import {
		createNewIntentRoute,
		deleteIntentRouteById,
		updateIntentRouteById
	} from '$lib/apis/classifiers';

	const i18n: any = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let routes: any[] = [];
	export let models: any[] = [];
	export let knowledgeBases: any[] = [];

	const ACTION_META: Record<string, { label: string; block: string; mood: any }> = {
		RAG_COLLECTION: { label: 'RAG Коллекция', block: 'syn-block-emerald', mood: 'focused' },
		DIRECT_RESPONSE: { label: 'Мгновенный ответ', block: 'syn-block-cobalt', mood: 'calm' },
		WEBHOOK_TOOL: { label: 'REST Вебхук', block: 'syn-block-amber', mood: 'curious' },
		SYSTEM_PROMPT_OVERRIDE: { label: 'Смена промпта', block: 'syn-block-cobalt', mood: 'focused' },
		MCP_HARNESS: { label: 'MCP Харнес', block: 'syn-block-violet', mood: 'optimized' },
		HUMAN_HANDOFF: { label: 'Перевод оператору', block: 'syn-block-coral', mood: 'handoff' }
	};

	const ACTION_TYPES = Object.keys(ACTION_META);
	const ALL_CHANNELS = ['chat', 'webrtc', 'sip'];

	let search = '';
	let channelFilter: 'all' | 'chat' | 'webrtc' | 'sip' = 'all';
	let showDetailsModal = false;
	let detailsRoute: any = null;
	let showCreateModal = false;

	let newRoute = emptyRoute();
	let draft: any = {};
	let slotTestInput = '';

	function emptyRoute() {
		return {
			bot_model_id: '',
			intent_name: '',
			examples: [],
			examplesText: '',
			min_confidence: 0.80,
			tau_low: 0.50,
			action_type: 'RAG_COLLECTION',
			target_collection_ids: [],
			use_hipporag: false,
			hybrid_weights: null,
			system_prompt_template: null,
			direct_response: null,
			webhook_config: null,
			required_slots: null,
			allowed_channels: ['chat', 'webrtc', 'sip'],
			mcp_harness_config: null,
			priority: 0,
			is_active: true
		};
	}

	$: filteredRoutes = routes.filter((r) => {
		const matchesSearch = search
			? (r.intent_name + ' ' + (r.bot_model_id ?? '')).toLowerCase().includes(search.toLowerCase())
			: true;
		const allowed = r.allowed_channels ?? ALL_CHANNELS;
		const matchesChannel = channelFilter === 'all' || allowed.includes(channelFilter);
		return matchesSearch && matchesChannel;
	});

	const openDetails = (route: any) => {
		detailsRoute = { ...route, allowed_channels: [...(route.allowed_channels ?? ALL_CHANNELS)] };
		draft = {
			examples: (route.examples ?? []).join('\n'),
			system_prompt_template: route.system_prompt_template ?? '',
			direct_response: route.direct_response ?? '',
			webhook_url: route.webhook_config?.url ?? '',
			webhook_response_path: route.webhook_config?.response_path ?? 'answer',
			webhook_queue: route.webhook_config?.queue ?? 'support_queue',
			webhook_headers: JSON.stringify(route.webhook_config?.headers ?? {}, null, 2),
			mcp_harness_id: route.mcp_harness_config?.harness_id ?? '',
			mcp_tool_name: route.mcp_harness_config?.tool_name ?? '',
			mcp_fallback_to_llm: Boolean(route.mcp_harness_config?.fallback_to_llm ?? true),
			mcp_args_json: JSON.stringify(route.mcp_harness_config?.argument_mapping ?? {}, null, 2),
			required_slots: (route.required_slots ?? []).map((slot: any) => ({ ...slot }))
		};
		showDetailsModal = true;
	};

	const toggleChannelInDetails = (ch: string) => {
		const cur = detailsRoute.allowed_channels ?? ALL_CHANNELS;
		detailsRoute.allowed_channels = cur.includes(ch)
			? cur.filter((c: string) => c !== ch)
			: [...cur, ch];
	};

	const addSlotRow = () => {
		draft.required_slots = [...(draft.required_slots ?? []), { name: '', pattern: '', question: '' }];
	};

	const removeSlotRow = (index: number) => {
		draft.required_slots = draft.required_slots.filter((_: any, i: number) => i !== index);
	};

	const testRegexMatch = (pattern: string, sample: string): boolean | null => {
		if (!pattern || !sample) return null;
		try {
			const re = new RegExp(`^(?:${pattern})$`);
			return re.test(sample);
		} catch {
			return false;
		}
	};

	const saveDetails = async () => {
		try {
			const token = localStorage.token;
			let headers = {};
			try {
				headers = draft.webhook_headers ? JSON.parse(draft.webhook_headers) : {};
			} catch {
				throw 'Заголовки вебхука должны быть валидным JSON';
			}
			let mcpArgs = {};
			try {
				mcpArgs = draft.mcp_args_json ? JSON.parse(draft.mcp_args_json) : {};
			} catch {
				throw 'Аргументы MCP должны быть валидным JSON';
			}

			const webhook_config =
				draft.webhook_url || detailsRoute.action_type === 'HUMAN_HANDOFF'
					? {
							url: draft.webhook_url || null,
							response_path: draft.webhook_response_path || 'answer',
							queue: draft.webhook_queue || 'support_queue',
							headers
						}
					: null;

			const mcp_harness_config =
				detailsRoute.action_type === 'MCP_HARNESS' && draft.mcp_harness_id
					? {
							harness_id: draft.mcp_harness_id.trim(),
							tool_name: draft.mcp_tool_name.trim(),
							argument_mapping: mcpArgs,
							fallback_to_llm: draft.mcp_fallback_to_llm
						}
					: null;

			const required_slots = (draft.required_slots ?? [])
				.filter((slot: any) => slot.name?.trim())
				.map((slot: any) => ({
					name: slot.name.trim(),
					pattern: slot.pattern?.trim() || null,
					question: slot.question?.trim() || null
				}));

			await updateIntentRouteById(token, detailsRoute.id, {
				...detailsRoute,
				examples: draft.examples
					.split('\n')
					.map((s: string) => s.trim())
					.filter(Boolean),
				system_prompt_template: draft.system_prompt_template || null,
				direct_response: draft.direct_response || null,
				webhook_config,
				mcp_harness_config,
				required_slots: required_slots.length ? required_slots : null
			});
			toast.success('Правило маршрутизации обновлено');
			showDetailsModal = false;
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const saveNewRoute = async () => {
		if (!newRoute.intent_name.trim() || !newRoute.bot_model_id) {
			toast.error('Укажите метку интента и целевую модель бота');
			return;
		}
		try {
			await createNewIntentRoute(localStorage.token, {
				...newRoute,
				intent_name: newRoute.intent_name.trim()
			});
			toast.success('Правило создано');
			showCreateModal = false;
			newRoute = emptyRoute();
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const patchRoute = async (route: any, fields: Record<string, any>) => {
		try {
			await updateIntentRouteById(localStorage.token, route.id, { ...route, ...fields });
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const removeRoute = async (route: any) => {
		try {
			await deleteIntentRouteById(localStorage.token, route.id);
			toast.success('Правило удалено');
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		}
	};
</script>

<div class="space-y-3">
	<!-- Панель поиска, фильтра каналов и создания правила -->
	<div class="flex flex-wrap items-center justify-between gap-2.5">
		<div class="relative flex-1 min-w-[220px]">
			<span class="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--syn-text-tertiary)]">
				<SynIcon name="search" size={15} />
			</span>
			<input
				class="w-full rounded-xl border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] pl-9 pr-4 py-2 text-xs outline-none focus:border-[var(--syn-cobalt)] transition"
				placeholder="Поиск по метке интента или модели бота..."
				bind:value={search}
			/>
		</div>

		<div class="flex items-center gap-1 rounded-xl bg-[var(--syn-bg-elevated)] p-1 text-xs">
			{#each [['all', 'Все каналы'], ['chat', 'Chat'], ['webrtc', 'WebRTC'], ['sip', 'SIP']] as [ch, label]}
				<button
					type="button"
					on:click={() => (channelFilter = ch)}
					class="rounded-lg px-2.5 py-1 transition {channelFilter === ch
						? 'bg-[var(--syn-bg-surface)] font-medium shadow-sm'
						: 'text-[var(--syn-text-tertiary)]'}"
				>
					{label}
				</button>
			{/each}
		</div>

		<button
			type="button"
			class="inline-flex items-center gap-1.5 rounded-xl bg-[var(--syn-text-primary)] px-3.5 py-2 text-xs font-medium text-[var(--syn-bg-surface)] hover:opacity-90 transition"
			on:click={() => (showCreateModal = true)}
		>
			<SynIcon name="plus" size={15} />
			<span>Добавить правило</span>
		</button>
	</div>

	<!-- Список правил в виде чистых минималистичных карточек-строк -->
	<div class="space-y-2">
		{#each filteredRoutes as route (route.id)}
			{@const meta = ACTION_META[route.action_type] ?? ACTION_META.RAG_COLLECTION}
			<div class="syn-color-block flex flex-wrap items-center justify-between gap-3">
				<div class="flex items-center gap-3 min-w-[220px]">
					<SynEmoji mood={route.is_active ? meta.mood : 'sleeping'} size={30} animated={false} />
					<div>
						<div class="flex items-center gap-2">
							<span class="font-mono text-xs font-semibold">{route.intent_name}</span>
							<span class="rounded-md bg-[var(--syn-bg-elevated)] px-2 py-0.5 text-[0.68rem] text-[var(--syn-text-secondary)]">
								{route.bot_model_id}
							</span>
						</div>
						<div class="mt-1 flex flex-wrap items-center gap-1.5 text-[0.68rem] text-[var(--syn-text-tertiary)]">
							<span>Примеров: {route.examples?.length ?? 0}</span>
							<span>·</span>
							<span>Слотов: {route.required_slots?.length ?? 0}</span>
							<span>·</span>
							{#each route.allowed_channels ?? ALL_CHANNELS as ch}
								<span class="rounded bg-[var(--syn-bg-canvas)] px-1.5 py-0.2 font-mono uppercase">{ch}</span>
							{/each}
						</div>
					</div>
				</div>

				<div class="flex flex-wrap items-center gap-4 text-xs">
					<div class="flex items-center gap-2">
						<span class="text-[var(--syn-text-tertiary)]">tau_high:</span>
						<input
							type="range"
							min="0.3"
							max="0.99"
							step="0.05"
							value={route.min_confidence}
							class="w-20 accent-[var(--syn-cobalt)]"
							on:change={(e) => patchRoute(route, { min_confidence: Number(e.currentTarget.value) })}
						/>
						<span class="w-8 font-mono">{route.min_confidence}</span>
					</div>

					<select
						class="rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-canvas)] px-2.5 py-1.5 text-xs font-medium"
						value={route.action_type}
						on:change={(e) => patchRoute(route, { action_type: e.currentTarget.value })}
					>
						{#each ACTION_TYPES as action}
							<option value={action}>{ACTION_META[action].label} ({action})</option>
						{/each}
					</select>

					<label class="inline-flex items-center gap-1.5 text-xs cursor-pointer">
						<input
							type="checkbox"
							class="accent-[var(--syn-emerald)]"
							checked={route.use_hipporag}
							on:change={(e) => patchRoute(route, { use_hipporag: e.currentTarget.checked })}
						/>
						<span>HippoRAG</span>
					</label>

					<label class="inline-flex items-center gap-1.5 text-xs cursor-pointer">
						<input
							type="checkbox"
							class="accent-[var(--syn-cobalt)]"
							checked={route.is_active}
							on:change={(e) => patchRoute(route, { is_active: e.currentTarget.checked })}
						/>
						<span>Активно</span>
					</label>

					<div class="flex items-center gap-1.5">
						<button
							type="button"
							class="inline-flex items-center gap-1 rounded-lg border border-[var(--syn-border-subtle)] px-2.5 py-1.5 text-xs hover:bg-[var(--syn-bg-elevated)] transition"
							on:click={() => openDetails(route)}
						>
							<SynIcon name="sliders" size={14} />
							<span>Настроить</span>
						</button>
						<button
							type="button"
							class="rounded-lg p-1.5 text-[var(--syn-coral)] hover:bg-[var(--syn-block-coral)] transition"
							on:click={() => removeRoute(route)}
							aria-label="Удалить правило"
						>
							<SynIcon name="trash" size={15} />
						</button>
					</div>
				</div>
			</div>
		{:else}
			<div class="syn-color-block py-12 text-center space-y-2">
				<SynEmoji mood="curious" size={38} />
				<div class="text-sm font-medium">Правила маршрутизации еще не созданы</div>
				<p class="text-xs text-[var(--syn-text-tertiary)]">
					Добавьте первое правило, чтобы обрабатывать типовые запросы за 5–15 мс без вызова LLM
				</p>
			</div>
		{/each}
	</div>
</div>

<!-- Модалка создания правила -->
{#if showCreateModal}
	<Modal size="md" title="Новое правило маршрутизации" on:close={() => (showCreateModal = false)}>
		<div class="space-y-3 text-sm">
			<div class="grid grid-cols-2 gap-3">
				<label class="block text-xs">
					<span class="text-[var(--syn-text-secondary)]">Метка интента</span>
					<input
						class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
						bind:value={newRoute.intent_name}
						placeholder="order_status"
					/>
				</label>
				<label class="block text-xs">
					<span class="text-[var(--syn-text-secondary)]">Модель бота</span>
					<select
						class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
						bind:value={newRoute.bot_model_id}
					>
						<option value="" disabled>Выберите модель...</option>
						{#each models as model}
							<option value={model.id}>{model.id}</option>
						{/each}
					</select>
				</label>
			</div>

			<label class="block text-xs">
				<span class="text-[var(--syn-text-secondary)]">Тип действия</span>
				<select
					class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
					bind:value={newRoute.action_type}
				>
					{#each ACTION_TYPES as action}
						<option value={action}>{ACTION_META[action].label} ({action})</option>
					{/each}
				</select>
			</label>

			<label class="block text-xs">
				<span class="text-[var(--syn-text-secondary)]">Примеры фраз (по одной на строку)</span>
				<textarea
					class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
					rows="4"
					bind:value={newRoute.examplesText}
					placeholder={'Где мой заказ?\nКогда придет доставка?'}
				/>
			</label>

			<div class="flex items-center justify-between pt-2">
				<button
					type="button"
					class="rounded-xl px-4 py-2 text-xs text-[var(--syn-text-secondary)] hover:bg-[var(--syn-bg-elevated)] transition"
					on:click={() => (showCreateModal = false)}
				>
					Отмена
				</button>
				<button
					type="button"
					class="inline-flex items-center gap-2 rounded-xl bg-[var(--syn-text-primary)] px-5 py-2.5 text-xs font-medium text-[var(--syn-bg-surface)] hover:opacity-90 transition"
					on:click={() => {
						newRoute = {
							...newRoute,
							examples: (newRoute.examplesText ?? '')
								.split('\n')
								.map((s: string) => s.trim())
								.filter(Boolean)
						};
						saveNewRoute();
					}}
				>
					<SynIcon name="check" size={15} />
					<span>Создать правило</span>
				</button>
			</div>
		</div>
	</Modal>
{/if}

<!-- Модалка детальной настройки -->
{#if showDetailsModal && detailsRoute}
	<Modal size="lg" title="Настройка: {detailsRoute.intent_name}" on:close={() => (showDetailsModal = false)}>
		<div class="space-y-4 text-sm max-h-[70vh] overflow-y-auto pr-1">
			<!-- Каналы действия правила -->
			<div class="syn-color-block syn-block-violet space-y-2">
				<div class="text-xs font-semibold">Каналы, в которых правило активно</div>
				<div class="flex gap-2">
					{#each ALL_CHANNELS as ch}
						<label class="inline-flex items-center gap-1.5 rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-1.5 text-xs cursor-pointer">
							<input
								type="checkbox"
								checked={(detailsRoute.allowed_channels ?? ALL_CHANNELS).includes(ch)}
								on:change={() => toggleChannelInDetails(ch)}
							/>
							<span class="font-mono uppercase">{ch}</span>
						</label>
					{/each}
				</div>
			</div>

			<label class="block text-xs">
				<span class="text-[var(--syn-text-secondary)]">Примеры фраз (по одной на строку)</span>
				<textarea
					class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
					rows="4"
					bind:value={draft.examples}
				/>
			</label>

			<label class="block text-xs">
				<span class="text-[var(--syn-text-secondary)]">Мгновенный ответ (DIRECT_RESPONSE)</span>
				<textarea
					class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
					rows="2"
					bind:value={draft.direct_response}
					placeholder="Ваш заказ №{'{'}slots.order{'}'} уже в пути."
				/>
			</label>

			<label class="block text-xs">
				<span class="text-[var(--syn-text-secondary)]">Шаблон системного промпта (SYSTEM_PROMPT_OVERRIDE)</span>
				<textarea
					class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
					rows="2"
					bind:value={draft.system_prompt_template}
				/>
			</label>

			<!-- Вебхук / HUMAN_HANDOFF -->
			<div class="syn-color-block syn-block-amber space-y-2.5">
				<div class="text-xs font-semibold">REST Вебхук / Перевод оператору</div>
				<label class="block text-xs">
					<span class="text-[var(--syn-text-secondary)]">URL бэкенда (проверяется SSRF-фильтром ядра)</span>
					<input
						class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
						bind:value={draft.webhook_url}
						placeholder="https://crm.example.com/api/ticket"
					/>
				</label>
				<div class="grid grid-cols-2 gap-2.5">
					<label class="block text-xs">
						<span class="text-[var(--syn-text-secondary)]">response_path</span>
						<input
							class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
							bind:value={draft.webhook_response_path}
						/>
					</label>
					<label class="block text-xs">
						<span class="text-[var(--syn-text-secondary)]">Очередь операторов</span>
						<input
							class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
							bind:value={draft.webhook_queue}
						/>
					</label>
				</div>
				<label class="block text-xs">
					<span class="text-[var(--syn-text-secondary)]">Заголовки (JSON)</span>
					<textarea
						class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
						rows="2"
						bind:value={draft.webhook_headers}
					/>
				</label>
			</div>

			<!-- MCP Харнес -->
			<div class="syn-color-block syn-block-violet space-y-2.5">
				<div class="text-xs font-semibold">MCP Харнес</div>
				<div class="grid grid-cols-2 gap-2.5">
					<label class="block text-xs">
						<span class="text-[var(--syn-text-secondary)]">harness_id</span>
						<input
							class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
							bind:value={draft.mcp_harness_id}
						/>
					</label>
					<label class="block text-xs">
						<span class="text-[var(--syn-text-secondary)]">tool_name</span>
						<input
							class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
							bind:value={draft.mcp_tool_name}
						/>
					</label>
				</div>
				<label class="block text-xs">
					<span class="text-[var(--syn-text-secondary)]">Аргументы (JSON mapping {'{{slots.*}}'})</span>
					<textarea
						class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
						rows="3"
						bind:value={draft.mcp_args_json}
					/>
				</label>
				<label class="inline-flex items-center gap-2 text-xs cursor-pointer">
					<input type="checkbox" class="accent-[var(--syn-violet)]" bind:checked={draft.mcp_fallback_to_llm} />
					<span>Fallback на LLM при ошибке инструмента</span>
				</label>
			</div>

			<!-- Обязательные слоты + живой regex-тестер -->
			<div class="syn-color-block syn-block-cobalt space-y-2.5">
				<div class="flex items-center justify-between">
					<div class="text-xs font-semibold">Обязательные слоты (slot-filling)</div>
					<button
						type="button"
						class="inline-flex items-center gap-1 rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-2.5 py-1 text-xs"
						on:click={addSlotRow}
					>
						<SynIcon name="plus" size={13} />
						<span>Слот</span>
					</button>
				</div>

				{#each draft.required_slots ?? [] as slot, i}
					<div class="rounded-xl border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] p-2.5 space-y-2">
						<div class="grid grid-cols-3 gap-2">
							<input
								class="rounded-lg border border-[var(--syn-border-subtle)] px-2 py-1.5 font-mono text-xs"
								bind:value={slot.name}
								placeholder="order"
							/>
							<input
								class="rounded-lg border border-[var(--syn-border-subtle)] px-2 py-1.5 font-mono text-xs"
								bind:value={slot.pattern}
								placeholder="regex-паттерн"
							/>
							<input
								class="rounded-lg border border-[var(--syn-border-subtle)] px-2 py-1.5 text-xs"
								bind:value={slot.question}
								placeholder="Уточните номер заказа?"
							/>
						</div>
						<div class="flex items-center gap-2 text-[0.68rem]">
							<input
								class="flex-1 rounded-lg border border-[var(--syn-border-subtle)] px-2 py-1 text-xs"
								placeholder="Тест строки: 12345"
								bind:value={slotTestInput}
							/>
							{#if testRegexMatch(slot.pattern, slotTestInput) === true}
								<span class="rounded-md bg-[var(--syn-block-emerald)] px-2 py-0.5 text-[var(--syn-emerald-ink)] font-medium">Совпадение</span>
							{:else if testRegexMatch(slot.pattern, slotTestInput) === false}
								<span class="rounded-md bg-[var(--syn-block-coral)] px-2 py-0.5 text-[var(--syn-coral-ink)] font-medium">Не совпадает</span>
							{/if}
							<button
								type="button"
								class="rounded-md p-1 text-[var(--syn-coral)] hover:bg-[var(--syn-block-coral)]"
								on:click={() => removeSlotRow(i)}
								aria-label="Удалить слот"
							>
								<SynIcon name="trash" size={13} />
							</button>
						</div>
					</div>
				{/each}
			</div>

			<div class="flex items-center justify-between pt-2">
				<button
					type="button"
					class="rounded-xl px-4 py-2 text-xs text-[var(--syn-text-secondary)] hover:bg-[var(--syn-bg-elevated)] transition"
					on:click={() => (showDetailsModal = false)}
				>
					Отмена
				</button>
				<button
					type="button"
					class="inline-flex items-center gap-2 rounded-xl bg-[var(--syn-text-primary)] px-5 py-2.5 text-xs font-medium text-[var(--syn-bg-surface)] hover:opacity-90 transition"
					on:click={saveDetails}
				>
					<SynIcon name="check" size={15} />
					<span>Сохранить правило</span>
				</button>
			</div>
		</div>
	</Modal>
{/if}
