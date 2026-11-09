<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import Modal from '$lib/components/common/Modal.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
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

	const ACTION_TYPES = ['RAG_COLLECTION', 'DIRECT_RESPONSE', 'WEBHOOK_TOOL', 'SYSTEM_PROMPT_OVERRIDE'];

	let search = '';
	let showDetailsModal = false;
	let detailsRoute: any = null;
	let showCreateModal = false;

	// Форма нового правила
	let newRoute = emptyRoute();

	// Черновик деталей редактируемого правила
	let draft: any = {};

	function emptyRoute() {
		return {
			bot_model_id: '',
			intent_name: '',
			examples: [],
			min_confidence: 0.8,
			tau_low: 0.5,
			action_type: 'RAG_COLLECTION',
			target_collection_ids: [],
			use_hipporag: false,
			hybrid_weights: null,
			system_prompt_template: null,
			direct_response: null,
			webhook_config: null,
			priority: 0,
			is_active: true
		};
	}

	$: filteredRoutes = routes.filter((r) =>
		(search ? (r.intent_name + ' ' + (r.bot_model_id ?? '')).toLowerCase().includes(search.toLowerCase()) : true)
	);

	const openDetails = (route: any) => {
		detailsRoute = route;
		draft = {
			examples: (route.examples ?? []).join('\n'),
			system_prompt_template: route.system_prompt_template ?? '',
			direct_response: route.direct_response ?? '',
			webhook_url: route.webhook_config?.url ?? '',
			webhook_response_path: route.webhook_config?.response_path ?? 'answer',
			webhook_headers: JSON.stringify(route.webhook_config?.headers ?? {}, null, 2),
			required_slots: (route.required_slots ?? []).map((slot: any) => ({ ...slot }))
		};
		showDetailsModal = true;
	};

	const addSlotRow = () => {
		draft.required_slots = [...(draft.required_slots ?? []), { name: '', pattern: '', question: '' }];
	};

	const removeSlotRow = (index: number) => {
		draft.required_slots = draft.required_slots.filter((_: any, i: number) => i !== index);
	};

	const saveDetails = async () => {
		try {
			const token = localStorage.token;
			let headers = {};
			try {
				headers = draft.webhook_headers ? JSON.parse(draft.webhook_headers) : {};
			} catch {
				throw 'Webhook headers must be valid JSON';
			}
			const webhook_config =
				draft.webhook_url || draft.webhook_response_path !== 'answer'
					? {
							url: draft.webhook_url,
							response_path: draft.webhook_response_path || 'answer',
							headers
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
				required_slots: required_slots.length ? required_slots : null
			});
			toast.success($i18n.t('Route updated'));
			showDetailsModal = false;
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const saveNewRoute = async () => {
		if (!newRoute.intent_name.trim() || !newRoute.bot_model_id) {
			toast.error($i18n.t('Intent name and bot model are required'));
			return;
		}
		try {
			const token = localStorage.token;
			await createNewIntentRoute(token, { ...newRoute, intent_name: newRoute.intent_name.trim() });
			toast.success($i18n.t('Route created'));
			showCreateModal = false;
			newRoute = emptyRoute();
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const patchRoute = async (route: any, fields: Record<string, any>) => {
		try {
			const token = localStorage.token;
			await updateIntentRouteById(token, route.id, { ...route, ...fields });
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const removeRoute = async (route: any) => {
		try {
			const token = localStorage.token;
			await deleteIntentRouteById(token, route.id);
			toast.success($i18n.t('Route deleted'));
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const toggleCollection = (route: any, kbId: string) => {
		const current = route.target_collection_ids ?? [];
		const next = current.includes(kbId) ? current.filter((id: string) => id !== kbId) : [...current, kbId];
		return next;
	};
</script>

<div class="flex items-center justify-between mb-2 gap-2">
	<div class="flex-1">
		<input
			class="w-full text-sm rounded-lg px-4 py-2 bg-gray-50 dark:bg-gray-900 outline-none border border-gray-100 dark:border-gray-850"
			placeholder={$i18n.t('Search intents...')}
			bind:value={search}
		/>
	</div>
	<button
		class="px-3.5 py-2 text-sm font-medium bg-black hover:bg-gray-900 dark:bg-white dark:hover:bg-gray-100 text-white dark:text-black rounded-lg transition"
		on:click={() => (showCreateModal = true)}
	>
		+ {$i18n.t('Add Intent Route')}
	</button>
</div>

<div class="overflow-x-auto rounded-xl border border-gray-100 dark:border-gray-850">
	<table class="w-full text-sm text-left">
		<thead class="text-xs text-gray-500 uppercase bg-gray-50 dark:bg-gray-900">
			<tr>
				<th class="px-3 py-2.5">{$i18n.t('Intent')}</th>
				<th class="px-3 py-2.5">{$i18n.t('Bot Model')}</th>
				<th class="px-3 py-2.5">{$i18n.t('Confidence')}</th>
				<th class="px-3 py-2.5">{$i18n.t('Action')}</th>
				<th class="px-3 py-2.5">{$i18n.t('Knowledge')}</th>
				<th class="px-3 py-2.5">{$i18n.t('HippoRAG')}</th>
				<th class="px-3 py-2.5">{$i18n.t('On')}</th>
				<th class="px-3 py-2.5"></th>
			</tr>
		</thead>
		<tbody>
			{#each filteredRoutes as route (route.id)}
				<tr class="border-t border-gray-100 dark:border-gray-850 hover:bg-gray-50 dark:hover:bg-gray-900/50">
					<td class="px-3 py-2 font-medium">
						{route.intent_name}
						{#if route.examples?.length}
							<div class="text-xs text-gray-400">({route.examples.length} {$i18n.t('examples')})</div>
						{/if}
					</td>
					<td class="px-3 py-2 text-xs">{route.bot_model_id}</td>
					<td class="px-3 py-2">
						<input
							type="range"
							min="0"
							max="1"
							step="0.05"
							value={route.min_confidence}
							class="w-20 accent-blue-500"
							on:change={(e) => patchRoute(route, { min_confidence: Number(e.currentTarget.value) })}
						/>
						<span class="text-xs ml-1">{route.min_confidence}</span>
					</td>
					<td class="px-3 py-2">
						<select
							class="text-xs rounded-md bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 px-2 py-1"
							value={route.action_type}
							on:change={(e) => patchRoute(route, { action_type: e.currentTarget.value })}
						>
							{#each ACTION_TYPES as action}
								<option value={action}>{action}</option>
							{/each}
						</select>
					</td>
					<td class="px-3 py-2">
						<Tooltip content={(route.target_collection_ids ?? []).join(', ') || $i18n.t('All knowledge bases')}>
							<button
								class="text-xs underline decoration-dashed"
								on:click={() => openDetails(route)}>#{(route.target_collection_ids ?? []).length}</button
							>
						</Tooltip>
					</td>
					<td class="px-3 py-2">
						<input
							type="checkbox"
							checked={route.use_hipporag}
							on:change={(e) => patchRoute(route, { use_hipporag: e.currentTarget.checked })}
						/>
					</td>
					<td class="px-3 py-2">
						<input
							type="checkbox"
							checked={route.is_active}
							on:change={(e) => patchRoute(route, { is_active: e.currentTarget.checked })}
						/>
					</td>
					<td class="px-3 py-2 whitespace-nowrap">
						<button class="text-xs mr-2 underline" on:click={() => openDetails(route)}
							>{$i18n.t('Details')}</button
						>
						<button class="text-xs text-red-500 underline" on:click={() => removeRoute(route)}
							>{$i18n.t('Delete')}</button
						>
					</td>
				</tr>
			{:else}
				<tr>
					<td colspan="8" class="px-3 py-8 text-center text-gray-400 text-sm">
						{$i18n.t('No intent routes yet. Add one to start deterministic routing without LLM calls.')}
					</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>

<!-- Модальное окно деталей правила -->
<Modal bind:show={showDetailsModal} size="md">
	<div class="flex flex-col gap-3 text-sm">
		<div class="text-lg font-medium">{$i18n.t('Route Details')}</div>
		{#if detailsRoute}
			<div class="text-xs text-gray-400">{detailsRoute.intent_name} / {detailsRoute.bot_model_id}</div>

			<label class="block">
				<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Example phrases (one per line)')}</div>
				<textarea
					class="w-full rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none h-24"
					bind:value={draft.examples}
				/>
			</label>

			<label class="block">
				<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('System prompt override')}</div>
				<textarea
					class="w-full rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none h-28"
					bind:value={draft.system_prompt_template}
					placeholder={'Use {{user_message}} to reference the question'}
				/>
			</label>

			<label class="block">
				<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Direct response template')}</div>
				<textarea
					class="w-full rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none h-20"
					bind:value={draft.direct_response}
				/>
			</label>

			<div class="grid grid-cols-2 gap-2">
				<label class="block">
					<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Webhook URL')}</div>
					<input
						class="w-full rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none"
						bind:value={draft.webhook_url}
					/>
				</label>
				<label class="block">
					<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Response JSON path')}</div>
					<input
						class="w-full rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none"
						bind:value={draft.webhook_response_path}
					/>
				</label>
			</div>

			<label class="block">
				<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Webhook headers (JSON)')}</div>
				<textarea
					class="w-full rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none h-16 font-mono text-xs"
					bind:value={draft.webhook_headers}
				/>
			</label>

			<div>
				<div class="mb-1 flex items-center justify-between">
					<span class="text-xs font-medium text-gray-500">{$i18n.t('Required slots (slot-filling state machine)')}</span>
					<button type="button" class="text-xs underline" on:click={addSlotRow}>+ {$i18n.t('Add slot')}</button>
				</div>
				{#each draft.required_slots ?? [] as slot, slotIndex}
					<div class="mb-1.5 flex gap-1.5">
						<input
							class="w-28 rounded-lg border border-gray-100 bg-gray-50 px-2 py-1 text-xs outline-none dark:border-gray-850 dark:bg-gray-900"
							placeholder="order_id"
							bind:value={slot.name}
						/>
						<input
							class="w-32 rounded-lg border border-gray-100 bg-gray-50 px-2 py-1 font-mono text-xs outline-none dark:border-gray-850 dark:bg-gray-900"
							placeholder={'regex: \\d{5,8}'}
							bind:value={slot.pattern}
						/>
						<input
							class="flex-1 rounded-lg border border-gray-100 bg-gray-50 px-2 py-1 text-xs outline-none dark:border-gray-850 dark:bg-gray-900"
							placeholder={$i18n.t('Clarifying question')}
							bind:value={slot.question}
						/>
						<button type="button" class="text-xs text-red-500" on:click={() => removeSlotRow(slotIndex)}>×</button>
					</div>
				{/each}
				{#if !(draft.required_slots ?? []).length}
					<div class="text-xs text-gray-400">
						{$i18n.t('Empty: the webhook fires immediately. Add slots to ask deterministic clarifying questions first.')}
					</div>
				{/if}
			</div>

			<div>
				<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Target knowledge bases')}</div>
				<div class="flex flex-wrap gap-2 max-h-32 overflow-y-auto">
					{#each knowledgeBases as kb (kb.id)}
						<label class="flex items-center gap-1 text-xs bg-gray-50 dark:bg-gray-900 rounded-md px-2 py-1 border border-gray-100 dark:border-gray-850">
							<input
								type="checkbox"
								checked={(detailsRoute.target_collection_ids ?? []).includes(kb.id)}
								on:change={(e) => {
									const next = e.currentTarget.checked
										? [...(detailsRoute.target_collection_ids ?? []), kb.id]
										: (detailsRoute.target_collection_ids ?? []).filter((id: string) => id !== kb.id);
									detailsRoute = { ...detailsRoute, target_collection_ids: next };
								}}
							/>
							{kb.name}
						</label>
					{:else}
						<span class="text-xs text-gray-400">{$i18n.t('No knowledge bases available')}</span>
					{/each}
				</div>
			</div>

			<div class="flex justify-end gap-2 pt-2">
				<button
					class="px-3 py-1.5 text-sm rounded-lg bg-gray-100 dark:bg-gray-800 hover:opacity-80"
					on:click={() => (showDetailsModal = false)}>{$i18n.t('Cancel')}</button
				>
				<button
					class="px-3 py-1.5 text-sm rounded-lg bg-black text-white dark:bg-white dark:text-black"
					on:click={saveDetails}>{$i18n.t('Save')}</button
				>
			</div>
		{/if}
	</div>
</Modal>

<!-- Модальное окно нового правила -->
<Modal bind:show={showCreateModal} size="sm">
	<div class="flex flex-col gap-3 text-sm">
		<div class="text-lg font-medium">{$i18n.t('Add Intent Route')}</div>

		<label class="block">
			<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Intent name (encoder class label)')}</div>
			<input
				class="w-full rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none"
				bind:value={newRoute.intent_name}
				placeholder="billing_refund"
			/>
		</label>

		<label class="block">
			<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Bot model')}</div>
			<select
				class="w-full rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none"
				bind:value={newRoute.bot_model_id}
			>
				<option value="">-</option>
				{#each models as model (model.id)}
					<option value={model.id}>{model.name ?? model.id}</option>
				{/each}
			</select>
		</label>

		<div class="grid grid-cols-2 gap-2">
			<label class="block">
				<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Action')}</div>
				<select
					class="w-full rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none"
					bind:value={newRoute.action_type}
				>
					{#each ACTION_TYPES as action}
						<option value={action}>{action}</option>
					{/each}
				</select>
			</label>
			<label class="flex items-center gap-2 mt-6">
				<input type="checkbox" bind:checked={newRoute.use_hipporag} />
				<span class="text-xs">{$i18n.t('Enable HippoRAG for this intent')}</span>
			</label>
		</div>

		<div class="flex justify-end gap-2 pt-1">
			<button
				class="px-3 py-1.5 text-sm rounded-lg bg-gray-100 dark:bg-gray-800 hover:opacity-80"
				on:click={() => (showCreateModal = false)}>{$i18n.t('Cancel')}</button
			>
			<button
				class="px-3 py-1.5 text-sm rounded-lg bg-black text-white dark:bg-white dark:text-black"
				on:click={saveNewRoute}>{$i18n.t('Create')}</button
			>
		</div>
	</div>
</Modal>
