<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import SynIcon from '../../synrespond/SynIcon.svelte';
	import SynEmoji from '../../synrespond/SynEmoji.svelte';
	import '../../synrespond/synrespond-tokens.css';
	import {
		getIntentFeedback,
		labelIntentFeedback,
		dismissIntentFeedback,
		trainFromFeedback,
		getTrainJobStatus
	} from '$lib/apis/classifiers';

	const i18n: any = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let encoders: any[] = [];
	export let routes: any[] = [];

	let items: any[] = [];
	let loading = true;

	// Черновики меток по строкам (по умолчанию — подсказка LLM или предсказание)
	let draftLabels: Record<string, string> = {};
	let labeledCount = 0;

	// Дообучение
	let training = false;
	let trainJobId: string | null = null;
	let trainStatus: any = null;

	// PERF-UI-03: Anchor Dataset против катастрофического забывания
	let useAnchorDataset = true;

	$: knownIntents = [...new Set(routes.map((r) => r.intent_name))];
	// Очередь сортируется по частоте попадания фразы в серую зону (PERF-CORE-05)
	$: sortedItems = [...items].sort(
		(a, b) => (b.meta?.hit_count ?? 1) - (a.meta?.hit_count ?? 1)
	);
	$: gateDelta = trainStatus?.result?.metrics?.delta_f1;

	const loadQueue = async () => {
		loading = true;
		try {
			items = (await getIntentFeedback(localStorage.token, 'pending')) ?? [];
			draftLabels = {};
			for (const item of items) {
				draftLabels[item.id] = item.hint_intent || item.predicted_intent || '';
			}
		} catch (e) {
			toast.error(`${e}`);
			items = [];
		} finally {
			loading = false;
		}
	};

	const confirmLabel = async (item: any) => {
		const intent = (draftLabels[item.id] || '').trim();
		if (!intent) {
			toast.error($i18n.t('Enter an intent label'));
			return;
		}
		try {
			await labelIntentFeedback(localStorage.token, item.id, intent);
			labeledCount += 1;
			items = items.filter((row) => row.id !== item.id);
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const dismiss = async (item: any) => {
		try {
			await dismissIntentFeedback(localStorage.token, item.id);
			items = items.filter((row) => row.id !== item.id);
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const confirmAllWithHints = async () => {
		for (const item of sortedItems) {
			if ((draftLabels[item.id] || '').trim()) {
				await confirmLabel(item);
			}
		}
	};

	// PERF-UI-03: клавиатурный конвейер разметки.
	// 1..3 — быстрый выбор топ-кандидата или LLM-хинта для первой фразы,
	// Enter — подтвердить метку, Backspace — отклонить фразу.
	const handleHotkey = (event: KeyboardEvent) => {
		const target = event.target as HTMLElement;
		const typing =
			target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA' || target?.tagName === 'SELECT';
		const first = sortedItems[0];
		if (!first) return;

		if (event.key === 'Enter') {
			if (typing) return; // Enter в поле ввода обрабатывает сам input
			event.preventDefault();
			confirmLabel(first);
			return;
		}
		if (typing) return;
		if (event.key === 'Backspace') {
			event.preventDefault();
			dismiss(first);
			return;
		}
		if (['1', '2', '3'].includes(event.key)) {
			const idx = Number(event.key) - 1;
			const hint = first.hint_intent;
			const candidates = first.top_candidates ?? [];
			const pick =
				idx === 0
					? hint || candidates[0]?.intent_name
					: candidates[idx]?.intent_name ?? candidates[0]?.intent_name;
			if (pick) {
				draftLabels = { ...draftLabels, [first.id]: pick };
				event.preventDefault();
			}
		}
	};

	const startTraining = async () => {
		const encoder = encoders[0];
		if (!encoder?.model_path) {
			toast.error($i18n.t('Register an encoder first'));
			return;
		}
		training = true;
		trainStatus = null;
		try {
			const job = await trainFromFeedback(localStorage.token, {
				base_model_id: encoder.model_path,
				activate: true,
				epochs: 4,
				use_anchor: useAnchorDataset
			});
			trainJobId = job.job_id;
			toast.success($i18n.t('Fine-tune job started'));
			pollTrainStatus();
		} catch (e) {
			toast.error(`${e}`);
		} finally {
			training = false;
		}
	};

	const pollTrainStatus = async () => {
		if (!trainJobId) return;
		try {
			trainStatus = await getTrainJobStatus(localStorage.token, trainJobId);
			if (trainStatus.status === 'running' || trainStatus.status === 'queued') {
				setTimeout(pollTrainStatus, 3000);
			} else if (trainStatus.status === 'completed') {
				toast.success(
					$i18n.t('Model hot-swapped. F1: {{f1}}', {
						f1: trainStatus.result?.metrics?.macro_f1 ?? 'n/a'
					})
				);
				dispatch('change');
			} else if (trainStatus.status === 'failed') {
				toast.error(`${trainStatus.error}`);
			}
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	onMount(loadQueue);
</script>

<svelte:window on:keydown={handleHotkey} />

<div class="mb-2 flex flex-wrap items-center justify-between gap-2">
	<div class="text-xs text-[var(--syn-text-secondary)]">
		{$i18n.t('Gray-zone utterances (tau_low <= p < tau_high). Confirm labels, then fine-tune and hot-swap the encoder.')}
		<span class="ml-1 rounded bg-[var(--syn-bg-elevated)] px-1.5 py-0.5 font-mono text-[0.65rem]">
			1..3 · Enter · Backspace
		</span>
	</div>
	<div class="flex items-center gap-2">
		<label
			class="inline-flex items-center gap-1.5 rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-2.5 py-1.5 text-xs cursor-pointer"
			title="Эталонная выборка подтвержденных фраз прошлых итераций защищает старые классы от забывания"
		>
			<input type="checkbox" class="size-3.5 accent-[var(--syn-cobalt)]" bind:checked={useAnchorDataset} />
			<span>Anchor Dataset</span>
		</label>
		<button
			class="px-3 py-2 text-xs rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] hover:bg-[var(--syn-bg-elevated)] transition"
			disabled={loading || items.length === 0}
			on:click={confirmAllWithHints}
		>
			{$i18n.t('Confirm all with hints')}
		</button>
		<button
			class="inline-flex items-center gap-1.5 px-3 py-2 text-xs rounded-lg bg-[var(--syn-text-primary)] text-[var(--syn-bg-surface)] hover:opacity-90 disabled:opacity-50 transition"
			disabled={training || items.length > 0}
			on:click={startTraining}
			title={$i18n.t('Labeled examples become the fine-tune dataset')}
		>
			<SynIcon name="flywheel" size={14} />
			{training ? $i18n.t('Training...') : $i18n.t('Fine-Tune & Hot-Swap ONNX')}
		</button>
	</div>
</div>

{#if trainStatus}
	<div
		class="syn-color-block mb-2 px-3.5 py-2.5 text-xs {trainStatus.status === 'failed'
			? 'syn-block-coral'
			: trainStatus.result?.blocked_by_gate
				? 'syn-block-amber'
				: 'syn-block-emerald'}"
	>
		<div class="flex flex-wrap items-center gap-2">
			<span class="font-semibold">{$i18n.t('Train job')} {trainJobId}: {trainStatus.status}</span>
			{#if gateDelta !== undefined && gateDelta !== null}
				<span
					class="inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-mono text-[0.7rem] {gateDelta >= -0.02
						? 'bg-[var(--syn-block-emerald)] text-[var(--syn-emerald-ink)]'
						: 'bg-[var(--syn-block-coral)] text-[var(--syn-coral-ink)]'}"
					title="Rollback Gate: падение macro-F1 больше 0.02 относительно baseline блокирует подмену"
				>
					<SynIcon name={gateDelta >= -0.02 ? 'check' : 'shield'} size={12} />
					Rollback Gate: ΔF1 {gateDelta > 0 ? '+' : ''}{gateDelta}
				</span>
			{/if}
			{#if trainStatus.result?.blocked_by_gate}
				<span class="rounded-full bg-[var(--syn-block-amber)] px-2 py-0.5 text-[0.7rem] text-[var(--syn-amber-ink)]">
					Swap blocked → candidates/
				</span>
			{/if}
		</div>
		{#if trainStatus.progress?.stage === 'training' && trainStatus.progress?.epoch}
			<div class="mt-1 text-[var(--syn-text-secondary)]">
				{$i18n.t('epoch')} {trainStatus.progress.epoch}/{trainStatus.progress.epochs}
				{#if trainStatus.progress.loss !== undefined}
					· loss {trainStatus.progress.loss}
				{/if}
			</div>
		{:else if trainStatus.progress?.stage}
			<div class="mt-1 text-[var(--syn-text-secondary)]">{trainStatus.progress.stage}</div>
		{/if}
		{#if trainStatus.result?.metrics}
			<div class="mt-1 text-[var(--syn-text-secondary)]">
				accuracy {trainStatus.result.metrics.accuracy} · macro-F1 {trainStatus.result.metrics.macro_f1}
				{#if trainStatus.result.metrics.baseline_macro_f1 !== undefined && trainStatus.result.metrics.baseline_macro_f1 !== null}
					· baseline-F1 {trainStatus.result.metrics.baseline_macro_f1}
				{/if}
			</div>
		{/if}
		{#if trainStatus.result?.blocked_by_gate}
			<div class="mt-1 text-[var(--syn-amber-ink)]">{trainStatus.result.warning}</div>
		{/if}
		{#if trainStatus.result?.out_model_id}
			<div class="mt-1 font-mono text-[var(--syn-text-tertiary)]">→ {trainStatus.result.out_model_id}</div>
		{/if}
	</div>
{/if}

{#if loading}
	<div class="flex flex-col items-center justify-center py-10 gap-2">
		<SynEmoji mood="focused" size={38} />
		<div class="text-xs text-[var(--syn-text-tertiary)]">{$i18n.t('Loading...')}</div>
	</div>
{:else}
	<div class="flex flex-col gap-2">
		{#each sortedItems as item, index (item.id)}
			<div class="syn-color-block px-4 py-3 {index === 0 ? 'ring-1 ring-[var(--syn-cobalt)]' : ''}">
				<div class="flex flex-wrap items-center justify-between gap-2">
					<div class="min-w-0 flex-1">
						<div class="flex items-center gap-2">
							<span class="truncate text-sm">{item.text}</span>
							{#if (item.meta?.hit_count ?? 1) > 1}
								<span
									class="inline-flex shrink-0 items-center rounded-full bg-[var(--syn-block-amber)] px-2 py-0.5 font-mono text-[0.65rem] text-[var(--syn-amber-ink)]"
									title="Сколько раз фраза попала в серую зону"
								>
									×{item.meta.hit_count} {$i18n.t('hits')}
								</span>
							{/if}
						</div>
						<div class="mt-0.5 text-xs text-[var(--syn-text-tertiary)]">
							{$i18n.t('encoder:')} {item.predicted_intent ?? '—'} ({((item.confidence ?? 0) * 100).toFixed(0)}%)
							{#if item.hint_intent}
								· LLM: <span class="font-medium">{item.hint_intent}</span> ({((item.hint_confidence ?? 0) * 100).toFixed(0)}%)
							{/if}
						</div>
					</div>
					<div class="flex items-center gap-2">
						<input
							class="w-44 text-xs rounded-lg px-2 py-1.5 bg-[var(--syn-bg-surface)] border border-[var(--syn-border-subtle)] outline-none focus:border-[var(--syn-cobalt)] transition"
							list="known-intents-{item.id}"
							bind:value={draftLabels[item.id]}
							placeholder={$i18n.t('intent label')}
							on:keydown={(e) => e.key === 'Enter' && confirmLabel(item)}
						/>
						<datalist id="known-intents-{item.id}">
							{#each knownIntents as intent}
								<option value={intent} />
							{/each}
						</datalist>
						{#if (item.top_candidates ?? []).length > 0}
							<div class="hidden xl:flex items-center gap-1">
								{#each item.top_candidates.slice(0, 3) as cand, ci}
									<button
										type="button"
										class="rounded-md border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-1.5 py-1 font-mono text-[0.62rem] hover:border-[var(--syn-cobalt)] transition"
										title="Хоткей {ci + 1}: {cand.intent_name} ({(cand.confidence * 100).toFixed(0)}%)"
										on:click={() => (draftLabels = { ...draftLabels, [item.id]: cand.intent_name })}
									>
										{ci + 1}:{cand.intent_name.slice(0, 12)}
									</button>
								{/each}
							</div>
						{/if}
						<button
							class="px-2.5 py-1.5 text-xs rounded-lg bg-[var(--syn-text-primary)] text-[var(--syn-bg-surface)] hover:opacity-90 transition"
							on:click={() => confirmLabel(item)}
						>
							{$i18n.t('Confirm')}
						</button>
						<button
							class="text-xs text-[var(--syn-text-tertiary)] underline hover:text-[var(--syn-coral)] transition"
							on:click={() => dismiss(item)}
						>
							{$i18n.t('Dismiss')}
						</button>
					</div>
				</div>
			</div>
		{:else}
			<div class="syn-color-block py-10 text-center space-y-2">
				<SynEmoji mood="calm" size={36} />
				<div class="text-xs text-[var(--syn-text-tertiary)]">
					{$i18n.t('Queue is empty. Gray-zone utterances appear here automatically.')}
				</div>
			</div>
		{/each}
	</div>
{/if}
