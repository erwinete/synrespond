<script lang="ts">
	import { onMount } from 'svelte';
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import Modal from '$lib/components/common/Modal.svelte';
	import SynIcon from '../../synrespond/SynIcon.svelte';
	import { getRagasAlerts, getRagasDatasets, updateRagasDatasetById } from '$lib/apis/ragas';

	const i18n: any = getContext('i18n');

	let alerts: any[] = [];
	let loading = true;

	// PERF-UI-06: конвертация галлюцинации в Golden Dataset
	let showDatasetModal = false;
	let datasets: any[] = [];
	let selectedDatasetId = '';
	let savingToDataset = false;
	let goldenForm: { question: string; ground_truth: string; alert: any } | null = null;

	const loadAlerts = async () => {
		loading = true;
		try {
			alerts = await getRagasAlerts(localStorage.token);
		} catch (e) {
			toast.error(`${e}`);
			alerts = [];
		} finally {
			loading = false;
		}
	};

	const openGoldenModal = async (alert: any) => {
		goldenForm = { question: alert.question ?? '', ground_truth: '', alert };
		try {
			datasets = (await getRagasDatasets(localStorage.token)) ?? [];
			selectedDatasetId = datasets[0]?.id ?? '';
		} catch (e) {
			datasets = [];
			toast.error(`${e}`);
		}
		showDatasetModal = true;
	};

	const saveToGoldenDataset = async () => {
		if (!goldenForm || !selectedDatasetId) {
			toast.error('Выберите датасет и укажите эталонный ответ');
			return;
		}
		if (!goldenForm.ground_truth.trim()) {
			toast.error('Заполните эталонный ответ (ground_truth)');
			return;
		}
		savingToDataset = true;
		try {
			const dataset = datasets.find((d) => d.id === selectedDatasetId);
			const items = [...(dataset?.items ?? [])];
			items.push({
				question: goldenForm.question,
				ground_truth: goldenForm.ground_truth.trim(),
				origin: 'hallucination_alert',
				source_answer: goldenForm.alert?.answer ?? null
			});
			await updateRagasDatasetById(localStorage.token, selectedDatasetId, {
				...dataset,
				items
			});
			toast.success('Инцидент превращен в регрессионный тест Golden Dataset');
			showDatasetModal = false;
			goldenForm = null;
		} catch (e) {
			toast.error(`${e}`);
		} finally {
			savingToDataset = false;
		}
	};

	onMount(loadAlerts);
</script>

{#if loading}
	<div class="flex flex-col items-center justify-center py-10 gap-2">
		<div class="text-xs text-[var(--syn-text-tertiary)]">{$i18n.t('Loading...')}</div>
	</div>
{:else}
	<div class="flex flex-col gap-2">
		{#each alerts as alert (alert.id)}
			<div class="syn-color-block syn-block-coral">
				<div class="flex items-start justify-between gap-4">
					<div class="min-w-0">
						<div class="flex items-start gap-2">
							<span class="mt-0.5 shrink-0 text-[var(--syn-coral)]">
								<svg width="16" height="16" viewBox="0 0 32 32" fill="none" aria-hidden="true">
									<circle cx="11.2" cy="13.5" r="2.3" stroke="currentColor" stroke-width="1.5" />
									<circle cx="11.2" cy="13.5" r="0.8" fill="currentColor" />
									<rect x="18.5" y="11.5" width="4" height="4" rx="1" stroke="currentColor" stroke-width="1.5" />
								</svg>
							</span>
							<div class="font-medium">{alert.question}</div>
						</div>
						<div class="mt-1 line-clamp-2 text-xs text-[var(--syn-text-secondary)]">{alert.answer}</div>
						<div class="mt-1 text-[0.65rem] text-[var(--syn-text-tertiary)]">
							{alert.bot_model_id ?? ''} · {new Date(alert.created_at * 1000).toLocaleString()}
						</div>
					</div>
					<div class="flex shrink-0 flex-col items-end gap-2">
						<div class="text-xs">
							{#each Object.entries(alert.scores ?? {}) as [key, value]}
								<span class="mr-2 font-mono">{key}: {typeof value === 'number' ? value.toFixed(2) : value}</span>
							{/each}
						</div>
						<button
							type="button"
							class="inline-flex items-center gap-1.5 rounded-lg border border-[var(--syn-amber)] px-2.5 py-1 text-xs font-medium text-[var(--syn-amber-ink)] hover:bg-[var(--syn-block-amber)] transition"
							on:click={() => openGoldenModal(alert)}
						>
							<SynIcon name="plus" size={13} />
							В эталонный датасет
						</button>
					</div>
				</div>
			</div>
		{:else}
			<div class="syn-color-block syn-block-emerald py-10 text-center space-y-2">
				<div class="text-sm font-medium text-[var(--syn-emerald-ink)]">
					{$i18n.t('No hallucination alerts. Faithfulness is above the threshold.')}
				</div>
			</div>
		{/each}
	</div>
{/if}

{#if showDatasetModal && goldenForm}
	<Modal size="sm" title="Галлюцинация → регрессионный тест" on:close={() => (showDatasetModal = false)}>
		<div class="space-y-3 text-sm">
			<div class="rounded-lg bg-[var(--syn-block-coral)] px-3 py-2 text-xs text-[var(--syn-coral-ink)]">
				{goldenForm.question}
			</div>
			<label class="block text-xs">
				<span class="text-[var(--syn-text-secondary)]">Датасет</span>
				<select
					class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2"
					bind:value={selectedDatasetId}
				>
					{#each datasets as ds (ds.id)}
						<option value={ds.id}>{ds.name}</option>
					{/each}
				</select>
			</label>
			<label class="block text-xs">
				<span class="text-[var(--syn-text-secondary)]">Эталонный ответ (ground_truth)</span>
				<textarea
					class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2"
					rows="4"
					bind:value={goldenForm.ground_truth}
					placeholder="Правильный ответ, по которому будут мерить faithfulness..."
				/>
			</label>
			<div class="flex items-center justify-between pt-1">
				<button
					type="button"
					class="rounded-xl px-4 py-2 text-xs text-[var(--syn-text-secondary)] hover:bg-[var(--syn-bg-elevated)] transition"
					on:click={() => (showDatasetModal = false)}
				>
					Отмена
				</button>
				<button
					type="button"
					disabled={savingToDataset}
					class="inline-flex items-center gap-2 rounded-xl bg-[var(--syn-text-primary)] px-5 py-2.5 text-xs font-medium text-[var(--syn-bg-surface)] hover:opacity-90 disabled:opacity-50 transition"
					on:click={saveToGoldenDataset}
				>
					<SynIcon name="check" size={14} />
					{savingToDataset ? 'Сохранение...' : 'Сохранить в датасет'}
				</button>
			</div>
		</div>
	</Modal>
{/if}
