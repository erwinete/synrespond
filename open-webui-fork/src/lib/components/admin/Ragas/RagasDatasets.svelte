<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import Modal from '$lib/components/common/Modal.svelte';
	import { getRagasDatasets, createNewRagasDataset, deleteRagasDatasetById } from '$lib/apis/ragas';

	const i18n: any = getContext('i18n');
	const dispatch = createEventDispatcher();

	let datasets: any[] = [];
	let loading = true;

	let showCreateModal = false;
	let newName = '';
	let newDescription = '';
	let newRaw = '';

	const loadDatasets = async () => {
		loading = true;
		try {
			datasets = await getRagasDatasets(localStorage.token);
		} catch (e) {
			toast.error(`${e}`);
			datasets = [];
		} finally {
			loading = false;
		}
	};

	const parseItems = (raw: string): any[] => {
		const text = raw.trim();
		if (!text) return [];
		if (text.startsWith('[') || text.startsWith('{')) {
			// JSON или JSONL
			if (text.startsWith('[')) return JSON.parse(text);
			return text
				.split('\n')
				.filter((line) => line.trim())
				.map((line) => JSON.parse(line));
		}
		// CSV: question,ground_truth
		const lines = text.split('\n').filter((line) => line.trim());
		const [header, ...rows] = lines;
		const columns = header.split(',').map((c) => c.trim());
		const qIdx = columns.findIndex((c) => c.toLowerCase() === 'question');
		const gIdx = columns.findIndex((c) => c.toLowerCase() === 'ground_truth');
		if (qIdx === -1) throw 'CSV header must contain a "question" column';
		return rows.map((row) => {
			const cells = row.split(',');
			return {
				question: cells[qIdx]?.trim() ?? '',
				ground_truth: gIdx !== -1 ? cells[gIdx]?.trim() ?? '' : ''
			};
		});
	};

	const createDataset = async () => {
		try {
			const items = parseItems(newRaw);
			if (!newName.trim() || !items.length) {
				throw $i18n.t('Name and at least one item are required');
			}
			await createNewRagasDataset(localStorage.token, {
				name: newName.trim(),
				description: newDescription.trim() || null,
				items
			});
			toast.success($i18n.t('Dataset created'));
			showCreateModal = false;
			newName = '';
			newDescription = '';
			newRaw = '';
			await loadDatasets();
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const removeDataset = async (dataset: any) => {
		try {
			await deleteRagasDatasetById(localStorage.token, dataset.id);
			toast.success($i18n.t('Dataset deleted'));
			await loadDatasets();
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	onMount(loadDatasets);
</script>

<div class="mb-2 flex items-center justify-between">
	<div class="text-xs text-gray-500">
		{$i18n.t('CSV (question,ground_truth) or JSONL with "question" and "ground_truth" fields.')}
	</div>
	<button
		class="rounded-lg bg-black px-3.5 py-2 text-sm font-medium text-white transition hover:bg-gray-900 dark:bg-white dark:text-black dark:hover:bg-gray-100"
		on:click={() => (showCreateModal = true)}
	>
		+ {$i18n.t('Upload Golden Dataset')}
	</button>
</div>

{#if loading}
	<div class="py-8 text-center text-sm text-gray-400">{$i18n.t('Loading...')}</div>
{:else}
	<div class="flex flex-col gap-2">
		{#each datasets as dataset (dataset.id)}
			<div class="flex items-center justify-between rounded-xl border border-gray-100 px-4 py-3 dark:border-gray-850">
				<div>
					<div class="text-sm font-medium">{dataset.name}</div>
					<div class="text-xs text-gray-400">
						{dataset.items?.length ?? 0} {$i18n.t('items')}
						{#if dataset.description}
							· {dataset.description}
						{/if}
					</div>
				</div>
				<button class="text-xs text-red-500 underline" on:click={() => removeDataset(dataset)}
					>{$i18n.t('Delete')}</button
				>
			</div>
		{:else}
			<div class="py-8 text-center text-sm text-gray-400">
				{$i18n.t('No golden datasets yet.')}
			</div>
		{/each}
	</div>
{/if}

<Modal bind:show={showCreateModal} size="md">
	<div class="flex flex-col gap-3 text-sm">
		<div class="text-lg font-medium">{$i18n.t('Upload Golden Dataset')}</div>

		<label class="block">
			<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Name')}</div>
			<input
				class="w-full rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 outline-none dark:border-gray-850 dark:bg-gray-900"
				bind:value={newName}
			/>
		</label>

		<label class="block">
			<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Description')}</div>
			<input
				class="w-full rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 outline-none dark:border-gray-850 dark:bg-gray-900"
				bind:value={newDescription}
			/>
		</label>

		<label class="block">
			<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('CSV / JSONL content')}</div>
			<textarea
				class="h-40 w-full rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 font-mono text-xs outline-none dark:border-gray-850 dark:bg-gray-900"
				bind:value={newRaw}
				placeholder={'question,ground_truth\n"How to return an order?","Returns are accepted within 30 days."'}
			/>
		</label>

		<div class="flex justify-end gap-2 pt-1">
			<button
				class="rounded-lg bg-gray-100 px-3 py-1.5 text-sm hover:opacity-80 dark:bg-gray-800"
				on:click={() => (showCreateModal = false)}>{$i18n.t('Cancel')}</button
			>
			<button
				class="rounded-lg bg-black px-3 py-1.5 text-sm text-white dark:bg-white dark:text-black"
				on:click={createDataset}>{$i18n.t('Create')}</button
			>
		</div>
	</div>
</Modal>
