<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import Modal from '$lib/components/common/Modal.svelte';
	import { getModels } from '$lib/apis';
	import {
		getRagasDatasets,
		createRagasRun,
		createOptimizationRun,
		getRagasRunSamples,
		deleteRagasRunById,
		applyBestWeights
	} from '$lib/apis/ragas';

	const i18n: any = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let runs: any[] = [];

	let datasets: any[] = [];
	let models: any[] = [];
	let showCreateModal = false;
	let form = { name: '', dataset_id: '', bot_model_id: '', max_items: 50, mode: 'benchmark', n_trials: 10 };
	let creating = false;

	let expandedRunId: string | null = null;
	let samples: any[] = [];
	let samplesLoading = false;

	$: loadRefs = async () => {
		try {
			const token = localStorage.token;
			[datasets, models] = await Promise.all([
				getRagasDatasets(token).catch(() => []),
				getModels(token).catch(() => [])
			]);
			if (!form.dataset_id && datasets.length) form.dataset_id = datasets[0].id;
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const createRun = async () => {
		if (!form.dataset_id) {
			toast.error($i18n.t('Select a golden dataset'));
			return;
		}
		creating = true;
		try {
			if (form.mode === 'optimize') {
				await createOptimizationRun(localStorage.token, {
					dataset_id: form.dataset_id,
					bot_model_id: form.bot_model_id || null,
					n_trials: form.n_trials,
					max_items: Math.min(form.max_items, 50)
				});
				toast.success($i18n.t('Auto-Tune started. Best weights will appear in the run metrics.'));
			} else {
				await createRagasRun(localStorage.token, form);
				toast.success($i18n.t('Run queued. The worker evaluates it in the background.'));
			}
			showCreateModal = false;
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		} finally {
			creating = false;
		}
	};

	const toggleRun = async (run: any) => {
		if (expandedRunId === run.id) {
			expandedRunId = null;
			return;
		}
		expandedRunId = run.id;
		samplesLoading = true;
		try {
			samples = await getRagasRunSamples(localStorage.token, run.id);
		} catch (e) {
			toast.error(`${e}`);
			samples = [];
		} finally {
			samplesLoading = false;
		}
	};

	const applyWeights = async (run: any) => {
		try {
			const response = await fetch(`/api/v1/ragas/runs/${run.id}/apply-weights`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					authorization: `Bearer ${localStorage.token}`
				},
				body: JSON.stringify({})
			});
			const data = await response.json();
			if (!response.ok) throw data.detail ?? data;
			toast.success($i18n.t('Best weights applied to production'));
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const removeRun = async (run: any) => {
		try {
			await deleteRagasRunById(localStorage.token, run.id);
			toast.success($i18n.t('Run deleted'));
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const statusColor = (status: string) =>
		({
			completed: 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300',
			failed: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300',
			running: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
			pending: 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300'
		})[status] ?? 'bg-gray-100 text-gray-600';

	onMount(() => {
		loadRefs();
	});
</script>

<div class="mb-2 flex justify-end">
	<button
		class="rounded-lg bg-black px-3.5 py-2 text-sm font-medium text-white transition hover:bg-gray-900 dark:bg-white dark:text-black dark:hover:bg-gray-100"
		on:click={() => (showCreateModal = true)}
	>
		+ {$i18n.t('New Evaluation Run')}
	</button>
</div>

<div class="flex flex-col gap-2">
	{#each runs as run (run.id)}
		<div class="rounded-xl border border-gray-100 dark:border-gray-850">
			<div
				role="button"
				tabindex="0"
				class="flex w-full cursor-pointer items-center justify-between px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-900/50"
				on:click={() => toggleRun(run)}
				on:keydown={(e) => e.key === 'Enter' && toggleRun(run)}
			>
				<div class="min-w-0">
					<div class="truncate text-sm font-medium">
						{run.name ?? run.id}
						<span class="ml-2 rounded px-1.5 py-0.5 text-[0.65rem] uppercase {statusColor(run.status)}">
							{run.status}
						</span>
					</div>
					<div class="text-xs text-gray-400">
						{run.mode} · {run.sample_count} {$i18n.t('samples')} · {new Date(run.created_at * 1000).toLocaleString()}
					</div>
				</div>
				<div class="flex items-center gap-3 text-xs text-gray-500">
					{#if run.metrics}
						{#each Object.entries(run.metrics) as [key, value]}
							<span>{key}: {typeof value === 'number' ? value.toFixed(2) : value}</span>
						{/each}
					{/if}
					{#if run.mode === 'optimize' && run.status === 'completed'}
						<button
							type="button"
							class="rounded bg-blue-100 px-1.5 py-0.5 text-blue-700 hover:bg-blue-200 dark:bg-blue-900/40 dark:text-blue-300"
							on:click|stopPropagation={() => applyWeights(run)}
						>
							{$i18n.t('Apply Best Weights')}
						</button>
					{/if}
					<button
						type="button"
						class="text-red-500 underline"
						on:click|stopPropagation={() => removeRun(run)}>{$i18n.t('Delete')}</button
					>
				</div>
			</div>
			{#if expandedRunId === run.id}
				<div class="border-t border-gray-100 dark:border-gray-850">
					{#if samplesLoading}
						<div class="px-4 py-4 text-xs text-gray-400">{$i18n.t('Loading...')}</div>
					{:else}
						<div class="divide-y divide-gray-100 dark:divide-gray-850">
							{#each samples as sample (sample.id)}
								<div class="px-4 py-3 text-xs">
									<div class="flex items-start justify-between gap-4">
										<div class="min-w-0 flex-1">
											<div class="font-medium">{sample.question}</div>
											<div class="mt-1 line-clamp-2 text-gray-500 dark:text-gray-400">{sample.answer}</div>
											{#if sample.retrieval_trace?.length || sample.contexts?.length}
												<div class="mt-1 text-gray-400">
													{$i18n.t('Contexts')}: {sample.contexts?.length ?? 0}
												</div>
											{/if}
										</div>
										<div class="flex shrink-0 gap-2 text-gray-500">
											{#each Object.entries(sample.scores ?? {}) as [key, value]}
												<span class="rounded bg-gray-50 px-1.5 py-0.5 dark:bg-gray-900" class:text-red-600={key === 'faithfulness' && typeof value === 'number' && value < 0.7}>
													{key}: {typeof value === 'number' ? value.toFixed(2) : value}
												</span>
											{/each}
											{#if sample.flag}
												<span class="rounded bg-red-100 px-1.5 py-0.5 text-red-700 dark:bg-red-900/40 dark:text-red-300">
													{sample.flag}
												</span>
											{/if}
										</div>
									</div>
								</div>
							{:else}
								<div class="px-4 py-4 text-xs text-gray-400">{$i18n.t('No samples yet')}</div>
							{/each}
						</div>
					{/if}
				</div>
			{/if}
		</div>
	{:else}
		<div class="py-8 text-center text-sm text-gray-400">
			{$i18n.t('No evaluation runs yet. Upload a golden dataset and start a batch run.')}
		</div>
	{/each}
</div>

<Modal bind:show={showCreateModal} size="sm">
	<div class="flex flex-col gap-3 text-sm">
		<div class="text-lg font-medium">{$i18n.t('New Evaluation Run')}</div>

		<label class="block">
			<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Golden dataset')}</div>
			<select
				class="w-full rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 outline-none dark:border-gray-850 dark:bg-gray-900"
				bind:value={form.dataset_id}
			>
				{#each datasets as dataset (dataset.id)}
					<option value={dataset.id}>{dataset.name} ({dataset.items?.length ?? 0})</option>
				{/each}
			</select>
		</label>

		<label class="block">
			<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Bot model (full RAG pipeline)')}</div>
			<select
				class="w-full rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 outline-none dark:border-gray-850 dark:bg-gray-900"
				bind:value={form.bot_model_id}
			>
				<option value="">-</option>
				{#each models as model (model.id)}
					<option value={model.id}>{model.name ?? model.id}</option>
				{/each}
			</select>
		</label>

		<label class="block">
			<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Run mode')}</div>
			<select
				class="w-full rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 outline-none dark:border-gray-850 dark:bg-gray-900"
				bind:value={form.mode}
			>
				<option value="benchmark">{$i18n.t('Benchmark evaluation (4 Ragas metrics)')}</option>
				<option value="optimize">{$i18n.t('Auto-Optimize Pipeline (Optuna weight search)')}</option>
			</select>
		</label>

		<label class="block">
			<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Max questions')}</div>
			<input
				type="number"
				min="1"
				max="1000"
				class="w-full rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 outline-none dark:border-gray-850 dark:bg-gray-900"
				bind:value={form.max_items}
			/>
		</label>

		{#if form.mode === 'optimize'}
			<label class="block">
				<div class="mb-1 text-xs font-medium text-gray-500">{$i18n.t('Optuna trials')}</div>
				<input
					type="number"
					min="2"
					max="60"
					class="w-full rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 outline-none dark:border-gray-850 dark:bg-gray-900"
					bind:value={form.n_trials}
				/>
			</label>
		{/if}

		<div class="flex justify-end gap-2 pt-1">
			<button
				class="rounded-lg bg-gray-100 px-3 py-1.5 text-sm hover:opacity-80 dark:bg-gray-800"
				on:click={() => (showCreateModal = false)}>{$i18n.t('Cancel')}</button
			>
			<button
				class="rounded-lg bg-black px-3 py-1.5 text-sm text-white dark:bg-white dark:text-black disabled:opacity-50"
				disabled={creating}
				on:click={createRun}
			>
				{creating ? $i18n.t('Starting...') : $i18n.t('Start Run')}
			</button>
		</div>
	</div>
</Modal>
