<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

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

	$: knownIntents = [...new Set(routes.map((r) => r.intent_name))];

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
		for (const item of items) {
			if ((draftLabels[item.id] || '').trim()) {
				await confirmLabel(item);
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
				epochs: 4
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

<div class="mb-2 flex items-center justify-between gap-2">
	<div class="text-xs text-gray-500">
		{$i18n.t('Gray-zone utterances (tau_low <= p < tau_high). Confirm labels, then fine-tune and hot-swap the encoder.')}
	</div>
	<div class="flex gap-2">
		<button
			class="px-3 py-2 text-xs rounded-lg bg-gray-100 dark:bg-gray-800 hover:opacity-80"
			disabled={loading || items.length === 0}
			on:click={confirmAllWithHints}
		>
			{$i18n.t('Confirm all with hints')}
		</button>
		<button
			class="px-3 py-2 text-xs rounded-lg bg-black text-white dark:bg-white dark:text-black disabled:opacity-50"
			disabled={training || items.length > 0}
			on:click={startTraining}
			title={$i18n.t('Labeled examples become the fine-tune dataset')}
		>
			{training ? $i18n.t('Training...') : $i18n.t('Fine-Tune & Hot-Swap ONNX')}
		</button>
	</div>
</div>

{#if trainStatus}
	<div
		class="mb-2 rounded-lg border border-gray-100 px-3 py-2 text-xs dark:border-gray-850
		{trainStatus.status === 'failed' ? 'text-red-500' : 'text-gray-500'}"
	>
		{$i18n.t('Train job')} {trainJobId}: {trainStatus.status}
		{#if trainStatus.progress?.stage === 'training' && trainStatus.progress?.epoch}
			· {$i18n.t('epoch')} {trainStatus.progress.epoch}/{trainStatus.progress.epochs}
			{#if trainStatus.progress.loss !== undefined}
				· loss {trainStatus.progress.loss}
			{/if}
		{:else if trainStatus.progress?.stage}
			· {trainStatus.progress.stage}
		{/if}
		{#if trainStatus.result?.metrics}
			· accuracy {trainStatus.result.metrics.accuracy} · macro-F1 {trainStatus.result.metrics.macro_f1}
			{#if trainStatus.result.metrics.baseline_macro_f1 !== undefined && trainStatus.result.metrics.baseline_macro_f1 !== null}
				· baseline-F1 {trainStatus.result.metrics.baseline_macro_f1} · ΔF1 {trainStatus.result.metrics.delta_f1}
			{/if}
		{/if}
		{#if trainStatus.result?.blocked_by_gate}
			<div class="mt-1 text-amber-600 dark:text-amber-400">{trainStatus.result.warning}</div>
		{/if}
		{#if trainStatus.result?.out_model_id}
			· → {trainStatus.result.out_model_id}
		{/if}
	</div>
{/if}

{#if loading}
	<div class="py-8 text-center text-sm text-gray-400">{$i18n.t('Loading...')}</div>
{:else}
	<div class="flex flex-col gap-2">
		{#each items as item (item.id)}
			<div class="rounded-xl border border-gray-100 px-4 py-3 dark:border-gray-850">
				<div class="flex flex-wrap items-center justify-between gap-2">
					<div class="min-w-0 flex-1">
						<div class="truncate text-sm">{item.text}</div>
						<div class="mt-0.5 text-xs text-gray-400">
							{$i18n.t('encoder:')} {item.predicted_intent ?? '—'} ({((item.confidence ?? 0) * 100).toFixed(0)}%)
							{#if item.hint_intent}
								· LLM: <span class="font-medium">{item.hint_intent}</span> ({((item.hint_confidence ?? 0) * 100).toFixed(0)}%)
							{/if}
						</div>
					</div>
					<div class="flex items-center gap-2">
						<input
							class="w-44 text-xs rounded-lg px-2 py-1.5 bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-850 outline-none"
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
						<button
							class="px-2.5 py-1.5 text-xs rounded-lg bg-black text-white dark:bg-white dark:text-black"
							on:click={() => confirmLabel(item)}
						>
							{$i18n.t('Confirm')}
						</button>
						<button class="text-xs text-gray-400 underline" on:click={() => dismiss(item)}
							>{$i18n.t('Dismiss')}</button
						>
					</div>
				</div>
			</div>
		{:else}
			<div class="py-8 text-center text-sm text-gray-400">
				{$i18n.t('Queue is empty. Gray-zone utterances appear here automatically.')}
			</div>
		{/each}
	</div>
{/if}
