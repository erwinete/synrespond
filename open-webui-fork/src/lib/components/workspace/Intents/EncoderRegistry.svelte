<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import Modal from '$lib/components/common/Modal.svelte';
	import SynIcon from '../../synrespond/SynIcon.svelte';
	import SynEmoji from '../../synrespond/SynEmoji.svelte';
	import {
		createNewEncoder,
		deleteEncoderById,
		testEncoderInference,
		reloadEncoderModel,
		rollbackEncoderModel
	} from '$lib/apis/classifiers';

	const i18n: any = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let encoders: any[] = [];

	const TASK_BADGES: Record<string, { title: string; block: string; mood: any; sample: string }> = {
		sequence_classification: {
			title: 'Классификация интентов',
			block: 'syn-block-cobalt',
			mood: 'focused',
			sample: 'Почему с моей карты дважды списали оплату за подписку?'
		},
		zero_shot_nli: {
			title: 'Zero-Shot NLI / Proximity',
			block: 'syn-block-violet',
			mood: 'curious',
			sample: 'Хочу поменять тарифный план на корпоративный'
		},
		token_classification: {
			title: 'Извлечение слотов (BIO NER + Regex)',
			block: 'syn-block-amber',
			mood: 'optimized',
			sample: 'Проверьте заказ № 94821 от 15.05.2026 на имя Иванов Иван, тел +7 913 123-45-67'
		},
		feature_extraction: {
			title: 'Векторные эмбеддинги и Реранкинг',
			block: 'syn-block-emerald',
			mood: 'calm',
			sample: 'Условия возврата товара надлежащего качества в течение 30 дней'
		}
	};

	let showCreateModal = false;
	let newEncoder = { name: '', task: 'sequence_classification', endpoint_url: '', model_path: '' };
	let testInputs: Record<string, string> = {};
	let testingId: string | null = null;
	let testResults: Record<string, any> = {};
	let busyId: string | null = null;

	const createEncoder = async () => {
		if (!newEncoder.name.trim() || !newEncoder.model_path.trim()) {
			toast.error('Укажите название и каталог модели');
			return;
		}
		try {
			await createNewEncoder(localStorage.token, {
				...newEncoder,
				name: newEncoder.name.trim(),
				model_path: newEncoder.model_path.trim()
			});
			toast.success('ONNX-энкодер зарегистрирован');
			showCreateModal = false;
			newEncoder = { name: '', task: 'sequence_classification', endpoint_url: '', model_path: '' };
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const removeEncoder = async (encoder: any) => {
		try {
			await deleteEncoderById(localStorage.token, encoder.id);
			toast.success('Энкодер удален');
			dispatch('change');
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const runTest = async (encoder: any) => {
		const meta = TASK_BADGES[encoder.task] ?? TASK_BADGES.sequence_classification;
		const phrase = (testInputs[encoder.id] || meta.sample).trim();
		testInputs[encoder.id] = phrase;
		testingId = encoder.id;
		try {
			const res = await testEncoderInference(localStorage.token, {
				text: phrase,
				model_path: encoder.model_path,
				task: encoder.task,
				endpoint_url: encoder.endpoint_url
			});
			testResults = { ...testResults, [encoder.id]: res };
		} catch (e) {
			toast.error(`Ошибка теста модели ${encoder.model_path}: ${e}`);
		} finally {
			testingId = null;
		}
	};

	const reloadModel = async (encoder: any) => {
		busyId = encoder.id;
		try {
			await reloadEncoderModel(localStorage.token, encoder.model_path);
			toast.success(`Модель ${encoder.model_path} перезагружена в RAM`);
		} catch (e) {
			toast.error(`Перезагрузка не удалась: ${e}`);
		} finally {
			busyId = null;
		}
	};

	const rollbackModel = async (encoder: any) => {
		busyId = encoder.id;
		try {
			await rollbackEncoderModel(localStorage.token, encoder.model_path);
			toast.success(`Модель ${encoder.model_path} откатилась к стабильной версии`);
			dispatch('change');
		} catch (e) {
			toast.error(`Откат не удался: ${e}`);
		} finally {
			busyId = null;
		}
	};
</script>

<div class="space-y-3">
	<div class="flex flex-wrap items-center justify-between gap-2">
		<p class="text-xs text-[var(--syn-text-secondary)]">
			Локальный каталог ONNX-моделей (`models/encoders/`). Поддерживаются все 4 задачи без вызова внешней LLM.
		</p>
		<button
			type="button"
			class="inline-flex items-center gap-1.5 rounded-xl bg-[var(--syn-text-primary)] px-3.5 py-2 text-xs font-medium text-[var(--syn-bg-surface)] hover:opacity-90 transition"
			on:click={() => (showCreateModal = true)}
		>
			<SynIcon name="plus" size={15} />
			<span>Зарегистрировать энкодер</span>
		</button>
	</div>

	<div class="grid grid-cols-1 gap-3 lg:grid-cols-2">
		{#each encoders as encoder (encoder.id)}
			{@const badge = TASK_BADGES[encoder.task] ?? TASK_BADGES.sequence_classification}
			{@const result = testResults[encoder.id]}
			<div class="syn-color-block {badge.block} flex flex-col justify-between gap-3">
				<div class="flex items-start justify-between gap-3">
					<div class="flex items-center gap-3">
						<SynEmoji mood={badge.mood} size={32} animated={false} />
						<div>
							<div class="flex items-center gap-2">
								<span class="text-sm font-semibold">{encoder.name}</span>
								<span class="rounded-md bg-[var(--syn-bg-surface)] px-2 py-0.5 font-mono text-[0.65rem]">
									{encoder.model_path}
								</span>
							</div>
							<div class="text-xs text-[var(--syn-text-secondary)]">{badge.title} ({encoder.task})</div>
						</div>
					</div>
					<div class="flex items-center gap-1">
						<button
							type="button"
							disabled={busyId === encoder.id}
							class="rounded-lg p-1.5 text-[var(--syn-text-secondary)] hover:bg-[var(--syn-bg-surface)] transition disabled:opacity-40"
							title="Перезагрузить в RAM"
							aria-label="Перезагрузить в RAM"
							on:click={() => reloadModel(encoder)}
						>
							<SynIcon name="refresh" size={15} />
						</button>
						<button
							type="button"
							disabled={busyId === encoder.id}
							class="rounded-lg p-1.5 text-[var(--syn-amber)] hover:bg-[var(--syn-bg-surface)] transition disabled:opacity-40"
							title="Откатить к стабильной версии"
							aria-label="Откатить к стабильной версии"
							on:click={() => rollbackModel(encoder)}
						>
							<SynIcon name="flywheel" size={15} />
						</button>
						<button
							type="button"
							class="rounded-lg p-1.5 text-[var(--syn-coral)] hover:bg-[var(--syn-bg-surface)] transition"
							on:click={() => removeEncoder(encoder)}
							aria-label="Удалить энкодер"
						>
							<SynIcon name="trash" size={15} />
						</button>
					</div>
				</div>

				<div class="flex gap-2">
					<input
						class="flex-1 rounded-xl border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 text-xs outline-none"
						placeholder={badge.sample}
						bind:value={testInputs[encoder.id]}
						on:keydown={(e) => e.key === 'Enter' && runTest(encoder)}
					/>
					<button
						type="button"
						disabled={testingId === encoder.id}
						on:click={() => runTest(encoder)}
						class="inline-flex items-center gap-1.5 rounded-xl bg-[var(--syn-text-primary)] px-3.5 py-2 text-xs font-medium text-[var(--syn-bg-surface)] disabled:opacity-50"
					>
						<SynIcon name="play" size={13} />
						<span>{testingId === encoder.id ? 'Инференс...' : 'Тест'}</span>
					</button>
				</div>

				{#if result}
					<div class="rounded-xl bg-[var(--syn-bg-surface)] p-3 border border-[var(--syn-border-subtle)] space-y-2 text-xs">
						<div class="flex items-center justify-between text-[0.7rem] text-[var(--syn-text-tertiary)]">
							<span>Задача: <b class="font-mono">{result.task ?? encoder.task}</b></span>
							<span>Задержка ONNX: <b class="font-mono text-[var(--syn-emerald)]">{result.latency_ms} мс</b></span>
						</div>

						{#if result.top_k_candidates}
							{#each result.top_k_candidates as cand}
								<div class="flex items-center gap-2">
									<span class="w-32 truncate font-mono text-[0.72rem]">{cand.intent_name}</span>
									<div class="flex-1 h-2 rounded-full bg-[var(--syn-bg-elevated)] overflow-hidden">
										<div
											class="h-full bg-[var(--syn-cobalt)] rounded-full transition-all"
											style="width: {Math.min(100, cand.confidence * 100).toFixed(1)}%"
										/>
									</div>
									<span class="w-12 text-right font-mono text-[0.7rem]">{(cand.confidence * 100).toFixed(1)}%</span>
								</div>
							{/each}
						{:else if result.spans}
							<div class="flex flex-wrap gap-1.5">
								{#each result.spans as span}
									<span class="inline-flex items-center gap-1 rounded-md bg-[var(--syn-block-amber)] px-2 py-0.5 font-mono text-[0.7rem]">
										<b>{span.label}:</b> {span.text} ({span.source ?? 'onnx'})
									</span>
								{:else}
									<span class="text-[var(--syn-text-tertiary)]">Сущности не обнаружены</span>
								{/each}
							</div>
						{:else if result.dimensions}
							<div class="flex items-center justify-between font-mono text-[0.72rem]">
								<span>Размерность: <b>{result.dimensions}d</b></span>
								<span>L2-норма: <b>{result.l2_norm}</b></span>
							</div>
							<div class="flex gap-0.5 h-3 items-end">
								{#each result.vector_preview ?? [] as val}
									<div
										class="flex-1 rounded-xs bg-[var(--syn-emerald)]"
										style="height: {Math.max(15, Math.min(100, (Math.abs(val) / 0.25) * 100))}%"
									/>
								{/each}
							</div>
						{/if}
					</div>
				{/if}
			</div>
		{:else}
			<div class="col-span-2 syn-color-block py-10 text-center text-xs text-[var(--syn-text-tertiary)]">
				Энкодеры еще не зарегистрированы. Запустите <code>scripts/seed_platform.py</code> или добавьте модель вручную.
			</div>
		{/each}
	</div>
</div>

{#if showCreateModal}
	<Modal size="md" title="Новый ONNX-энкодер" on:close={() => (showCreateModal = false)}>
		<div class="space-y-3 text-sm">
			<label class="block text-xs">
				<span class="text-[var(--syn-text-secondary)]">Название</span>
				<input
					class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2"
					bind:value={newEncoder.name}
					placeholder="ruBERT Intents v2"
				/>
			</label>
			<div class="grid grid-cols-2 gap-3">
				<label class="block text-xs">
					<span class="text-[var(--syn-text-secondary)]">Задача</span>
					<select
						class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
						bind:value={newEncoder.task}
					>
						{#each Object.keys(TASK_BADGES) as task}
							<option value={task}>{task}</option>
						{/each}
					</select>
				</label>
				<label class="block text-xs">
					<span class="text-[var(--syn-text-secondary)]">Каталог модели (models/encoders/)</span>
					<input
						class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
						bind:value={newEncoder.model_path}
						placeholder="rubert-intents-v1"
					/>
				</label>
			</div>
			<label class="block text-xs">
				<span class="text-[var(--syn-text-secondary)]">URL сервиса (необязательно — по умолчанию общий)</span>
				<input
					class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
					bind:value={newEncoder.endpoint_url}
					placeholder="http://encoder-classifier:8001"
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
					on:click={createEncoder}
				>
					<SynIcon name="check" size={15} />
					<span>Зарегистрировать</span>
				</button>
			</div>
		</div>
	</Modal>
{/if}
