<script lang="ts">
	import { onMount } from 'svelte';
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { getRagasConfig, updateRagasConfig, getRagasRuns } from '$lib/apis/ragas';

	import SynIcon from '../synrespond/SynIcon.svelte';
	import SynEmoji from '../synrespond/SynEmoji.svelte';
	import '../synrespond/synrespond-tokens.css';

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

	// PERF-UI-06: цвет кольца по значению метрики (порог галлюцинаций 0.7)
	const gaugeColor = (value: number) =>
		value >= 0.85 ? 'var(--syn-emerald)' : value >= 0.7 ? 'var(--syn-amber)' : 'var(--syn-coral)';
	const gaugeMood = (value: number) => (value >= 0.85 ? 'optimized' : value >= 0.7 ? 'curious' : 'hallucination');
	const gaugeDash = (value: number, radius: number) => {
		const circumference = 2 * Math.PI * radius;
		const filled = Math.max(0, Math.min(1, value)) * circumference;
		return `${filled.toFixed(1)} ${(circumference - filled).toFixed(1)}`;
	};

	onMount(loadAll);
</script>

<div class="syn-studio min-h-full rounded-2xl p-4 md:p-5 space-y-4">
	<header class="flex flex-wrap items-center justify-between gap-3">
		<div class="flex items-center gap-3">
			<SynEmoji mood={loading ? 'focused' : 'optimized'} size={34} />
			<div>
				<h1 class="text-base font-semibold tracking-tight">{$i18n.t('RAG Quality & Ragas')}</h1>
				<p class="text-xs text-[var(--syn-text-tertiary)]">
					Непрерывный аудит ответов: галлюцинации, релевантность и полнота контекста
				</p>
			</div>
		</div>
		<span
			class="inline-flex items-center gap-1.5 rounded-full bg-[var(--syn-block-amber)] px-2.5 py-0.5 text-[0.68rem] font-medium text-[var(--syn-amber-ink)]"
		>
			<SynIcon name="ragas" size={13} />
			Ragas & Optuna Auto-Tune
		</span>
	</header>

	<nav
		class="flex flex-wrap items-center gap-1.5 rounded-xl bg-[var(--syn-bg-elevated)] p-1.5 text-xs font-medium"
		aria-label="Разделы Ragas"
	>
		{#each [['overview', 'Overview'], ['runs', 'Runs'], ['datasets', 'Golden Datasets'], ['alerts', 'Hallucination Alerts']] as [tab, label]}
			<a
				href="#/"
				on:click|preventDefault={() => (selectedTab = tab)}
				class="inline-flex items-center gap-2 rounded-lg px-3.5 py-2 transition {selectedTab === tab
					? 'bg-[var(--syn-bg-surface)] text-[var(--syn-text-primary)] shadow-sm'
					: 'text-[var(--syn-text-secondary)] hover:text-[var(--syn-text-primary)]'}"
			>
				{#if tab === 'alerts'}
					<SynIcon name="shield" size={14} />
				{:else if tab === 'runs'}
					<SynIcon name="flywheel" size={14} />
				{:else if tab === 'datasets'}
					<SynIcon name="layers" size={14} />
				{:else}
					<SynIcon name="ragas" size={14} />
				{/if}
				{$i18n.t(label)}
			</a>
		{/each}
	</nav>

	{#if loading}
		<div class="flex flex-col items-center justify-center py-16 gap-3">
			<SynEmoji mood="focused" size={44} />
			<div class="text-xs text-[var(--syn-text-tertiary)]">Загрузка метрик качества...</div>
		</div>
	{:else if selectedTab === 'overview'}
		<div class="flex flex-col gap-4">
			<div class="grid grid-cols-2 gap-3 lg:grid-cols-4">
				{#each Object.entries(METRIC_LABELS) as [key, label]}
					{@const value = summaryMetrics[key]}
					<div class="syn-color-block syn-block-amber flex items-center gap-3">
						{#if value !== undefined}
							<svg width="64" height="64" viewBox="0 0 64 64" role="img" aria-label="{label}: {(value * 100).toFixed(1)}%">
								<circle cx="32" cy="32" r="26" fill="none" stroke="var(--syn-bg-elevated)" stroke-width="6" />
								<circle
									cx="32"
									cy="32"
									r="26"
									fill="none"
									stroke={gaugeColor(value)}
									stroke-width="6"
									stroke-linecap="round"
									stroke-dasharray={gaugeDash(value, 26)}
									transform="rotate(-90 32 32)"
								/>
								<text x="32" y="36" text-anchor="middle" font-size="13" font-weight="600" fill="var(--syn-text-primary)">
									{(value * 100).toFixed(0)}%
								</text>
							</svg>
						{:else}
							<SynEmoji mood="sleeping" size={52} animated={false} />
						{/if}
						<div>
							<div class="text-xs text-[var(--syn-text-secondary)]">{label}</div>
							<div class="mt-0.5 flex items-center gap-1">
								<SynEmoji mood={value !== undefined ? gaugeMood(value) : 'sleeping'} size={16} animated={false} />
								<span class="font-mono text-[0.68rem] text-[var(--syn-text-tertiary)]">
									{value !== undefined ? (value * 100).toFixed(1) + '%' : 'нет прогонов'}
								</span>
							</div>
						</div>
					</div>
				{/each}
			</div>
			<div class="text-xs text-[var(--syn-text-tertiary)]">
				{$i18n.t('Average across the latest completed runs. Live production sampling is appended to daily live runs.')}
			</div>

			{#if config}
				<form
					class="syn-color-block syn-block-amber flex flex-col gap-3"
					on:submit|preventDefault={saveConfig}
				>
					<div class="flex items-center gap-2 text-sm font-semibold">
						<SynIcon name="sliders" size={15} />
						{$i18n.t('Evaluation Settings')}
					</div>
					<label class="flex items-center gap-2 text-sm">
						<input type="checkbox" class="size-4 accent-[var(--syn-amber)]" bind:checked={config.enable} />
						{$i18n.t('Enable Ragas evaluation')}
					</label>
					<label class="block text-xs">
						<span class="text-[var(--syn-text-secondary)]">{$i18n.t('Worker URL')}</span>
						<input
							class="mt-1 w-full rounded-lg px-3 py-2 bg-[var(--syn-bg-surface)] border border-[var(--syn-border-subtle)] outline-none focus:border-[var(--syn-amber)] transition font-mono"
							bind:value={config.worker_url}
							placeholder="http://ragas-worker:8002"
						/>
					</label>
					<div class="grid grid-cols-2 gap-3">
						<label class="block text-xs">
							<div class="mb-1 text-[var(--syn-text-secondary)]">{$i18n.t('Live sampling rate (0-1)')}</div>
							<input
								type="number"
								min="0"
								max="1"
								step="0.05"
								class="w-full rounded-lg px-3 py-2 bg-[var(--syn-bg-surface)] border border-[var(--syn-border-subtle)] outline-none focus:border-[var(--syn-amber)] transition font-mono"
								bind:value={config.live_sample_rate}
							/>
						</label>
						<label class="block text-xs">
							<div class="mb-1 text-[var(--syn-text-secondary)]">{$i18n.t('Faithfulness alert threshold')}</div>
							<input
								type="number"
								min="0"
								max="1"
								step="0.05"
								class="w-full rounded-lg px-3 py-2 bg-[var(--syn-bg-surface)] border border-[var(--syn-border-subtle)] outline-none focus:border-[var(--syn-amber)] transition font-mono"
								bind:value={config.hallucination_threshold}
							/>
						</label>
					</div>
					<div class="flex justify-end">
						<button
							class="inline-flex items-center gap-2 rounded-xl bg-[var(--syn-text-primary)] px-5 py-2 text-xs font-medium text-[var(--syn-bg-surface)] hover:opacity-90 transition"
							type="submit"
						>
							<SynIcon name="check" size={14} />
							{$i18n.t('Save')}
						</button>
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
</div>
