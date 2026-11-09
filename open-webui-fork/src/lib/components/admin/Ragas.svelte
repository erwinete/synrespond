<script lang="ts">
	import { onMount } from 'svelte';
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { getRagasConfig, updateRagasConfig, getRagasRuns } from '$lib/apis/ragas';

	import RagasRuns from './Ragas/RagasRuns.svelte';
	import RagasDatasets from './Ragas/RagasDatasets.svelte';
	import RagasAlerts from './Ragas/RagasAlerts.svelte';

	const i18n: any = getContext('i18n');

	let selectedTab = 'overview';
	let loading = true;

	let config: any = null;
	let runs: any[] = [];

	$: latestCompleted = runs.filter((r) => r.status === 'completed' && r.metrics).slice(0, 3);
	$: summaryMetrics = (() => {
		// Усредняем сводные метрики последних завершенных прогонов
		const acc: Record<string, { sum: number; n: number }> = {};
		for (const run of latestCompleted) {
			for (const [key, value] of Object.entries(run.metrics ?? {})) {
				if (typeof value === 'number') {
					acc[key] = acc[key] ?? { sum: 0, n: 0 };
					acc[key].sum += value;
					acc[key].n += 1;
				}
			}
		}
		const result: Record<string, number> = {};
		for (const [key, { sum, n }] of Object.entries(acc)) {
			result[key] = sum / n;
		}
		return result;
	})();

	const loadAll = async () => {
		loading = true;
		try {
			const token = localStorage.token;
			[config, runs] = await Promise.all([
				getRagasConfig(token).catch(() => null),
				getRagasRuns(token).catch(() => [])
			]);
		} catch (e) {
			toast.error(`${e}`);
		} finally {
			loading = false;
		}
	};

	const saveConfig = async () => {
		try {
			config = await updateRagasConfig(localStorage.token, config);
			toast.success($i18n.t('Settings saved successfully!'));
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const METRIC_LABELS: Record<string, string> = {
		faithfulness: 'Faithfulness',
		answer_relevancy: 'Answer Relevancy',
		context_precision: 'Context Precision',
		context_recall: 'Context Recall'
	};

	onMount(loadAll);
</script>

<div class="mt-0.5 mb-2 flex flex-col gap-y-3 justify-between md:flex-row">
	<div class="flex items-center px-1 text-lg font-medium">{$i18n.t('RAG Quality & Ragas')}</div>
</div>

<div
	class="mb-3 flex gap-1 rounded-[0.75rem] bg-gray-50 p-1 text-[0.8rem] font-medium shadow-sm dark:bg-gray-900"
>
	{#each [['overview', 'Overview'], ['runs', 'Runs'], ['datasets', 'Golden Datasets'], ['alerts', 'Hallucination Alerts']] as [tab, label]}
		<a
			href="#/"
			on:click|preventDefault={() => (selectedTab = tab)}
			class="min-w-fit rounded-lg px-3 py-1 transition {selectedTab === tab
				? 'bg-white text-gray-900 shadow dark:bg-gray-800 dark:text-gray-100'
				: 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-200'}"
		>
			{$i18n.t(label)}
		</a>
	{/each}
</div>

{#if loading}
	<div class="flex justify-center py-8">
		<svg class="animate-spin size-5 text-gray-400" viewBox="0 0 24 24" fill="none">
			<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
			<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
		</svg>
	</div>
{:else if selectedTab === 'overview'}
	<div class="flex flex-col gap-4">
		<div class="grid grid-cols-2 gap-3 lg:grid-cols-4">
			{#each Object.entries(METRIC_LABELS) as [key, label]}
				<div class="rounded-xl border border-gray-100 p-4 dark:border-gray-850">
					<div class="text-xs text-gray-500">{label}</div>
					<div class="mt-1 text-2xl font-semibold">
						{summaryMetrics[key] !== undefined ? (summaryMetrics[key] * 100).toFixed(1) + '%' : '—'}
					</div>
				</div>
			{/each}
		</div>
		<div class="text-xs text-gray-400">
			{$i18n.t('Average across the latest completed runs. Live production sampling is appended to daily live runs.')}
		</div>

		{#if config}
			<form
				class="rounded-xl border border-gray-100 p-4 flex flex-col gap-3 dark:border-gray-850"
				on:submit|preventDefault={saveConfig}
			>
				<div class="font-medium">{$i18n.t('Evaluation Settings')}</div>
				<label class="flex items-center gap-2 text-sm">
					<input type="checkbox" bind:checked={config.enable} />
					{$i18n.t('Enable Ragas evaluation')}
				</label>
				<label class="block text-sm">
					<div class="mb-1 text-xs text-gray-500">{$i18n.t('Worker URL')}</div>
					<input
						class="w-full rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none"
						bind:value={config.worker_url}
						placeholder="http://ragas-worker:8002"
					/>
				</label>
				<div class="grid grid-cols-2 gap-3">
					<label class="block text-sm">
						<div class="mb-1 text-xs text-gray-500">{$i18n.t('Live sampling rate (0-1)')}</div>
						<input
							type="number"
							min="0"
							max="1"
							step="0.05"
							class="w-full rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none"
							bind:value={config.live_sample_rate}
						/>
					</label>
					<label class="block text-sm">
						<div class="mb-1 text-xs text-gray-500">{$i18n.t('Faithfulness alert threshold')}</div>
						<input
							type="number"
							min="0"
							max="1"
							step="0.05"
							class="w-full rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none"
							bind:value={config.hallucination_threshold}
						/>
					</label>
				</div>
				<div class="flex justify-end">
					<button
						class="rounded-lg bg-black px-3 py-1.5 text-sm text-white dark:bg-white dark:text-black"
						type="submit">{$i18n.t('Save')}</button
					>
				</div>
			</form>
		{/if}
	</div>
{:else if selectedTab === 'runs'}
	<RagasRuns {runs} on:change={loadAll} />
{:else if selectedTab === 'datasets'}
	<RagasDatasets on:change={loadAll} />
{:else}
	<RagasAlerts />
{/if}
