<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { toast } from 'svelte-sonner';
	import SynIcon from './SynIcon.svelte';
	import SynEmoji from './SynEmoji.svelte';

	export let open = false;
	export let activeModule: 'channels' | 'guard' | 'encoder' | 'rag' | 'ragas' = 'encoder';
	export let settings: Record<string, any> = {};

	const dispatch = createEventDispatcher();
	let saving = false;

	$: wDense = Number(settings['rag.hybrid_weight_dense'] ?? 0.4);
	$: wBm25 = Number(settings['rag.hybrid_weight_bm25'] ?? 0.2);
	$: wHippo = Number(settings['rag.hybrid_weight_hipporag'] ?? 0.4);
	$: wSum = Math.max(wDense + wBm25 + wHippo, 0.001);
	$: normDense = ((wDense / wSum) * 100).toFixed(0);
	$: normBm25 = ((wBm25 / wSum) * 100).toFixed(0);
	$: normHippo = ((wHippo / wSum) * 100).toFixed(0);

	const MODULE_TABS: {
		id: 'channels' | 'guard' | 'encoder' | 'rag' | 'ragas';
		label: string;
		icon: any;
	}[] = [
		{ id: 'channels', label: 'Каналы и Голос', icon: 'channels' },
		{ id: 'guard', label: 'Защита и ПДн', icon: 'shield' },
		{ id: 'encoder', label: 'Интенты и Пороги', icon: 'encoder' },
		{ id: 'rag', label: 'Гибридный RAG', icon: 'hipporag' },
		{ id: 'ragas', label: 'Аудит Ragas', icon: 'ragas' }
	];

	const applyChannelPreset = (preset: 'chat' | 'webrtc' | 'sip') => {
		if (preset === 'chat') {
			settings = {
				...settings,
				'intent_classifier.tau_high': 0.85,
				'intent_classifier.tau_low': 0.50,
				'intent_classifier.disambiguation_delta': 0.15,
				'intent_classifier.timeout': 0.40,
				'hipporag.enable': true,
				'rag.enable_query_rewriting': true,
				'rag.hybrid_weight_dense': 0.40,
				'rag.hybrid_weight_bm25': 0.20,
				'rag.hybrid_weight_hipporag': 0.40
			};
		} else if (preset === 'webrtc') {
			settings = {
				...settings,
				'intent_classifier.tau_high': 0.80,
				'intent_classifier.tau_low': 0.45,
				'intent_classifier.disambiguation_delta': 0.12,
				'intent_classifier.timeout': 0.30,
				'hipporag.enable': true,
				'rag.enable_query_rewriting': false,
				'voice.anger_threshold': 0.75
			};
		} else {
			settings = {
				...settings,
				'intent_classifier.tau_high': 0.80,
				'intent_classifier.tau_low': 0.45,
				'intent_classifier.disambiguation_delta': 0.12,
				'intent_classifier.timeout': 0.25,
				'hipporag.enable': false,
				'rag.enable_query_rewriting': false,
				'voice.anger_threshold': 0.70
			};
		}
		toast.success(`Применен рекомендованный профиль: ${preset.toUpperCase()}`);
	};

	const normalizeWeightsNow = () => {
		const total = wDense + wBm25 + wHippo;
		if (total <= 0) return;
		settings = {
			...settings,
			'rag.hybrid_weight_dense': Number((wDense / total).toFixed(2)),
			'rag.hybrid_weight_bm25': Number((wBm25 / total).toFixed(2)),
			'rag.hybrid_weight_hipporag': Number((wHippo / total).toFixed(2))
		};
	};

	const saveAllSettings = async () => {
		if (Number(settings['intent_classifier.tau_low']) >= Number(settings['intent_classifier.tau_high'])) {
			toast.error('Нижний порог tau_low должен быть строго меньше верхнего порога tau_high');
			return;
		}
		saving = true;
		try {
			const response = await fetch('/api/v1/mcp', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					authorization: `Bearer ${localStorage.token}`
				},
				body: JSON.stringify({
					jsonrpc: '2.0',
					id: Date.now(),
					method: 'tools/call',
					params: {
						name: 'platform_configure_modules',
						arguments: { settings }
					}
				})
			});
			const data = await response.json();
			if (data.error) throw data.error.message || JSON.stringify(data.error);
			toast.success('Настройки модулей сохранены на лету (без перезапуска)');
			dispatch('saved', settings);
			open = false;
		} catch (e) {
			toast.error(`Ошибка сохранения: ${e}`);
		} finally {
			saving = false;
		}
	};
</script>

{#if open}
	<div
		class="fixed inset-0 z-50 flex justify-end bg-black/35 backdrop-blur-[2px] transition-opacity"
		role="dialog"
		aria-modal="true"
		aria-label="Настройки модулей платформы"
	>
		<button
			type="button"
			class="flex-1 cursor-default"
			aria-label="Закрыть панель"
			on:click={() => (open = false)}
		/>

		<div
			class="syn-studio relative flex h-full w-full max-w-xl flex-col justify-between overflow-y-auto border-l border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] p-6 shadow-2xl"
		>
			<div>
				<div class="flex items-center justify-between pb-4 border-b border-[var(--syn-border-subtle)]">
					<div class="flex items-center gap-3">
						<SynEmoji
							mood={activeModule === 'guard'
								? 'shielded'
								: activeModule === 'ragas'
									? 'optimized'
									: 'focused'}
							size={34}
						/>
						<div>
							<h2 class="text-base font-semibold tracking-tight">Конфигуратор модулей synrespond</h2>
							<p class="text-xs text-[var(--syn-text-tertiary)]">
								Горячее применение настроек ядра без перезапуска контейнеров
							</p>
						</div>
					</div>
					<button
						type="button"
						class="rounded-lg p-2 text-[var(--syn-text-secondary)] hover:bg-[var(--syn-bg-elevated)] transition"
						on:click={() => (open = false)}
					>
						<SynIcon name="close" size={18} />
					</button>
				</div>

				<div class="mt-4 grid grid-cols-5 gap-1.5 rounded-xl bg-[var(--syn-bg-elevated)] p-1">
					{#each MODULE_TABS as tab}
						<button
							type="button"
							on:click={() => (activeModule = tab.id)}
							class="flex flex-col items-center gap-1 rounded-lg px-2 py-2 text-[0.7rem] font-medium transition {activeModule ===
							tab.id
								? 'bg-[var(--syn-bg-surface)] text-[var(--syn-text-primary)] shadow-sm'
								: 'text-[var(--syn-text-tertiary)] hover:text-[var(--syn-text-primary)]'}"
						>
							<SynIcon name={tab.icon} size={16} />
							<span class="truncate max-w-full">{tab.label}</span>
						</button>
					{/each}
				</div>

				<div class="mt-4 flex items-center justify-between rounded-xl bg-[var(--syn-bg-canvas)] px-3.5 py-2.5 text-xs">
					<span class="text-[var(--syn-text-secondary)] font-medium">Профиль оптимизации:</span>
					<div class="flex gap-1.5">
						<button
							type="button"
							class="rounded-md border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-2.5 py-1 hover:border-[var(--syn-cobalt)] transition"
							on:click={() => applyChannelPreset('chat')}
						>
							Чат (Точность)
						</button>
						<button
							type="button"
							class="rounded-md border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-2.5 py-1 hover:border-[var(--syn-emerald)] transition"
							on:click={() => applyChannelPreset('webrtc')}
						>
							WebRTC (Баланс)
						</button>
						<button
							type="button"
							class="rounded-md border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-2.5 py-1 hover:border-[var(--syn-amber)] transition"
							on:click={() => applyChannelPreset('sip')}
						>
							SIP (TTFB &lt; 650мс)
						</button>
					</div>
				</div>

				<div class="mt-5 space-y-4">
					{#if activeModule === 'encoder'}
						<div class="syn-color-block syn-block-cobalt space-y-4">
							<div class="flex items-center justify-between">
								<div>
									<div class="text-sm font-semibold">Маршрутизатор интентов (ONNX Runtime)</div>
									<div class="text-xs text-[var(--syn-text-secondary)]">
										Отсекает типовые запросы за 5–15 мс до вызова тяжелой LLM
									</div>
								</div>
								<input
									type="checkbox"
									class="size-4 accent-[var(--syn-cobalt)]"
									bind:checked={settings['intent_classifier.enable']}
								/>
							</div>

							<div class="rounded-xl bg-[var(--syn-bg-surface)] p-3.5 border border-[var(--syn-border-subtle)]">
								<div class="flex justify-between text-[0.7rem] font-medium text-[var(--syn-text-secondary)] mb-1.5">
									<span>Fallback RAG (0..{settings['intent_classifier.tau_low'] ?? 0.5})</span>
									<span class="text-[var(--syn-amber)]">
										Серая зона ({settings['intent_classifier.tau_low'] ?? 0.5}..{settings['intent_classifier.tau_high'] ?? 0.85})
									</span>
									<span class="text-[var(--syn-emerald)]">
										Прямой ответ (&ge;{settings['intent_classifier.tau_high'] ?? 0.85})
									</span>
								</div>
								<div class="h-2.5 w-full rounded-full overflow-hidden flex bg-[var(--syn-bg-elevated)]">
									<div
										style="width: {(Number(settings['intent_classifier.tau_low'] ?? 0.5) * 100).toFixed(0)}%"
										class="bg-gray-400/50 transition-all"
									/>
									<div
										style="width: {((Number(settings['intent_classifier.tau_high'] ?? 0.85) - Number(settings['intent_classifier.tau_low'] ?? 0.5)) * 100).toFixed(0)}%"
										class="bg-[var(--syn-amber)]/75 transition-all"
									/>
									<div class="flex-1 bg-[var(--syn-emerald)] transition-all" />
								</div>

								<div class="mt-4 grid grid-cols-2 gap-3">
									<label class="block text-xs">
										<span class="text-[var(--syn-text-secondary)]">Порог прямого действия (tau_high)</span>
										<div class="mt-1 flex items-center gap-2">
											<input
												type="range"
												min="0.5"
												max="0.99"
												step="0.01"
												bind:value={settings['intent_classifier.tau_high']}
												class="flex-1 accent-[var(--syn-cobalt)]"
											/>
											<span class="w-10 text-right font-mono">{settings['intent_classifier.tau_high'] ?? 0.85}</span>
										</div>
									</label>
									<label class="block text-xs">
										<span class="text-[var(--syn-text-secondary)]">Нижняя граница серой зоны (tau_low)</span>
										<div class="mt-1 flex items-center gap-2">
											<input
												type="range"
												min="0.1"
												max="0.85"
												step="0.01"
												bind:value={settings['intent_classifier.tau_low']}
												class="flex-1 accent-[var(--syn-amber)]"
											/>
											<span class="w-10 text-right font-mono">{settings['intent_classifier.tau_low'] ?? 0.50}</span>
										</div>
									</label>
								</div>
							</div>

							<div class="grid grid-cols-2 gap-3 text-xs">
								<label class="block">
									<span class="text-[var(--syn-text-secondary)]">Дельта дизамбигуации (&Delta;p)</span>
									<input
										type="number"
										step="0.01"
										min="0.01"
										max="0.50"
										bind:value={settings['intent_classifier.disambiguation_delta']}
										class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
									/>
								</label>
								<label class="block">
									<span class="text-[var(--syn-text-secondary)]">Таймаут классификатора (сек)</span>
									<input
										type="number"
										step="0.05"
										min="0.05"
										max="5.0"
										bind:value={settings['intent_classifier.timeout']}
										class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
									/>
								</label>
							</div>

							<div class="grid grid-cols-2 gap-3 text-xs">
								<label class="block">
									<span class="text-[var(--syn-text-secondary)]">Модель интентов по умолчанию</span>
									<input
										type="text"
										bind:value={settings['intent_classifier.model_id']}
										placeholder="rubert-intents-v1"
										class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
									/>
								</label>
								<label class="block">
									<span class="text-[var(--syn-text-secondary)]">Модель NER-слотов</span>
									<input
										type="text"
										bind:value={settings['intent_classifier.slot_model_id']}
										placeholder="rubert-slots-v1"
										class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
									/>
								</label>
							</div>

							<label class="flex items-center justify-between pt-2 border-t border-[var(--syn-border-subtle)] text-xs">
								<span>Автоматический сбор серой зоны в Active Learning Queue</span>
								<input
									type="checkbox"
									class="size-4 accent-[var(--syn-cobalt)]"
									bind:checked={settings['active_learning.enable']}
								/>
							</label>
						</div>

					{:else if activeModule === 'rag'}
						<div class="syn-color-block syn-block-emerald space-y-4">
							<div class="flex items-center justify-between">
								<div>
									<div class="text-sm font-semibold">Трехконтурный поиск (Dense + BM25 + HippoRAG)</div>
									<div class="text-xs text-[var(--syn-text-secondary)]">
										Взвешенный Reciprocal Rank Fusion (k=60) с графовым PageRank
									</div>
								</div>
								<input
									type="checkbox"
									class="size-4 accent-[var(--syn-emerald)]"
									bind:checked={settings['hipporag.enable']}
								/>
							</div>

							<div class="rounded-xl bg-[var(--syn-bg-surface)] p-3.5 border border-[var(--syn-border-subtle)] space-y-3">
								<div class="flex items-center justify-between text-xs">
									<span class="font-medium">Пропорция слияния контуров:</span>
									<button
										type="button"
										class="text-[0.7rem] text-[var(--syn-emerald)] underline"
										on:click={normalizeWeightsNow}
									>
										Нормализовать к 1.00
									</button>
								</div>

								<div class="h-3 w-full rounded-full overflow-hidden flex">
									<div style="width: {normDense}%" class="bg-[var(--syn-cobalt)] transition-all" title="Dense: {normDense}%" />
									<div style="width: {normBm25}%" class="bg-[var(--syn-amber)] transition-all" title="BM25: {normBm25}%" />
									<div style="width: {normHippo}%" class="bg-[var(--syn-emerald)] transition-all" title="HippoRAG: {normHippo}%" />
								</div>

								<div class="space-y-2.5 pt-1 text-xs">
									<label class="block">
										<div class="flex justify-between">
											<span>Векторный контур (Dense BGE-M3 / ruBERT)</span>
											<span class="font-mono">{settings['rag.hybrid_weight_dense'] ?? 0.4} ({normDense}%)</span>
										</div>
										<input
											type="range"
											min="0"
											max="1"
											step="0.05"
											bind:value={settings['rag.hybrid_weight_dense']}
											class="w-full accent-[var(--syn-cobalt)]"
										/>
									</label>
									<label class="block">
										<div class="flex justify-between">
											<span>Лексический контур (BM25 точные артикулы/номера)</span>
											<span class="font-mono">{settings['rag.hybrid_weight_bm25'] ?? 0.2} ({normBm25}%)</span>
										</div>
										<input
											type="range"
											min="0"
											max="1"
											step="0.05"
											bind:value={settings['rag.hybrid_weight_bm25']}
											class="w-full accent-[var(--syn-amber)]"
										/>
									</label>
									<label class="block">
										<div class="flex justify-between">
											<span>Графовый контур (HippoRAG Personalized PageRank)</span>
											<span class="font-mono">{settings['rag.hybrid_weight_hipporag'] ?? 0.4} ({normHippo}%)</span>
										</div>
										<input
											type="range"
											min="0"
											max="1"
											step="0.05"
											bind:value={settings['rag.hybrid_weight_hipporag']}
											class="w-full accent-[var(--syn-emerald)]"
										/>
									</label>
								</div>
							</div>

							<label class="flex items-center justify-between text-xs pt-1">
								<span>Деконтекстуализация и декомпозиция запроса (Query Rewriting)</span>
								<input
									type="checkbox"
									class="size-4 accent-[var(--syn-emerald)]"
									bind:checked={settings['rag.enable_query_rewriting']}
								/>
							</label>
						</div>

					{:else if activeModule === 'guard'}
						<div class="syn-color-block syn-block-coral space-y-4">
							<div class="flex items-center justify-between">
								<div>
									<div class="text-sm font-semibold">Экран безопасности (Prompt Guard & PII Vault)</div>
									<div class="text-xs text-[var(--syn-text-secondary)]">
										Блокировка джейлбрейков до LLM и обратимое маскирование ПДн (152-ФЗ)
									</div>
								</div>
								<SynEmoji mood="shielded" size={28} />
							</div>

							<label class="flex items-center justify-between rounded-xl bg-[var(--syn-bg-surface)] p-3 border border-[var(--syn-border-subtle)] text-xs">
								<div>
									<div class="font-medium">Детектор промпт-инъекций (/v1/guard)</div>
									<div class="text-[var(--syn-text-tertiary)]">Отказ без вызова LLM при попытке перехвата инструкций</div>
								</div>
								<input
									type="checkbox"
									class="size-4 accent-[var(--syn-coral)]"
									bind:checked={settings['guardrails.prompt_guard']}
								/>
							</label>

							<label class="block text-xs">
								<div class="flex justify-between mb-1">
									<span>Порог срабатывания Prompt Guard</span>
									<span class="font-mono">{settings['guardrails.injection_threshold'] ?? 0.75}</span>
								</div>
								<input
									type="range"
									min="0.25"
									max="0.95"
									step="0.05"
									bind:value={settings['guardrails.injection_threshold']}
									class="w-full accent-[var(--syn-coral)]"
								/>
							</label>

							<label class="flex items-center justify-between rounded-xl bg-[var(--syn-bg-surface)] p-3 border border-[var(--syn-border-subtle)] text-xs">
								<div>
									<div class="font-medium">Обратимое маскирование ПДн (Reversible PII Vault)</div>
									<div class="text-[var(--syn-text-tertiary)]">Карты, телефоны, паспорта, ИНН, СНИЛС заменяются на [CARD_1]</div>
								</div>
								<input
									type="checkbox"
									class="size-4 accent-[var(--syn-coral)]"
									bind:checked={settings['guardrails.pii_masking']}
								/>
							</label>
						</div>

					{:else if activeModule === 'channels'}
						<div class="syn-color-block syn-block-violet space-y-4">
							<div class="text-sm font-semibold">Автономные каналы и Эмоциональная деэскалация</div>

							{#each [
								['channels.chat.enabled', 'Текстовый чат и веб-виджет (SSE)', 'Полная поддержка Markdown и цепочек Reasoning Path'],
								['channels.webrtc.enabled', 'Браузерный голосовой канал (WebRTC / WS PCM)', 'Полнодуплексный звук 16/48 кГц с мгновенным barge-in'],
								['channels.sip.enabled', 'Телефония Asterisk / FreeSWITCH (AudioSocket :8765)', 'Поток 8 кГц slin, голосовые филлеры 250 мс и перевод в очередь']
							] as [key, title, subtitle]}
								<label class="flex items-center justify-between rounded-xl bg-[var(--syn-bg-surface)] p-3 border border-[var(--syn-border-subtle)] text-xs">
									<div>
										<div class="font-medium">{title}</div>
										<div class="text-[var(--syn-text-tertiary)]">{subtitle}</div>
									</div>
									<input
										type="checkbox"
										class="size-4 accent-[var(--syn-violet)]"
										bind:checked={settings[key]}
									/>
								</label>
							{/each}

							<div class="grid grid-cols-2 gap-3 pt-2 text-xs">
								<label class="block">
									<span class="text-[var(--syn-text-secondary)]">Порог гнева абонента (0..1)</span>
									<input
										type="number"
										min="0.3"
										max="1.0"
										step="0.05"
										bind:value={settings['voice.anger_threshold']}
										class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
									/>
								</label>
								<label class="block">
									<span class="text-[var(--syn-text-secondary)]">Очередь Asterisk (Handoff)</span>
									<input
										type="text"
										bind:value={settings['voice.handoff_queue']}
										placeholder="support_queue"
										class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
									/>
								</label>
							</div>
						</div>

					{:else if activeModule === 'ragas'}
						<div class="syn-color-block syn-block-amber space-y-4">
							<div class="flex items-center justify-between">
								<div>
									<div class="text-sm font-semibold">Непрерывный аудит качества Ragas</div>
									<div class="text-xs text-[var(--syn-text-secondary)]">
										Фоновая проверка живых ответов на галлюцинации и полноту контекста
									</div>
								</div>
								<input
									type="checkbox"
									class="size-4 accent-[var(--syn-amber)]"
									bind:checked={settings['ragas.enable']}
								/>
							</div>

							<div class="grid grid-cols-2 gap-3 text-xs">
								<label class="block">
									<span class="text-[var(--syn-text-secondary)]">Доля живого сэмплирования (0..1)</span>
									<input
										type="number"
										min="0"
										max="1"
										step="0.05"
										bind:value={settings['ragas.live_sample_rate']}
										class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
									/>
								</label>
								<label class="block">
									<span class="text-[var(--syn-text-secondary)]">Порог Hallucination Alert</span>
									<input
										type="number"
										min="0.1"
										max="0.95"
										step="0.05"
										bind:value={settings['ragas.hallucination_threshold']}
										class="mt-1 w-full rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-2 font-mono"
									/>
								</label>
							</div>
						</div>
					{/if}
				</div>
			</div>

			<div class="mt-6 flex items-center justify-between border-t border-[var(--syn-border-subtle)] pt-4">
				<button
					type="button"
					class="rounded-xl px-4 py-2 text-xs font-medium text-[var(--syn-text-secondary)] hover:bg-[var(--syn-bg-elevated)] transition"
					on:click={() => (open = false)}
				>
					Отмена
				</button>
				<button
					type="button"
					disabled={saving}
					on:click={saveAllSettings}
					class="inline-flex items-center gap-2 rounded-xl bg-[var(--syn-text-primary)] px-5 py-2.5 text-xs font-medium text-[var(--syn-bg-surface)] hover:opacity-90 disabled:opacity-50 transition"
				>
					<SynIcon name="check" size={15} />
					<span>{saving ? 'Применение...' : 'Сохранить настройки модуля'}</span>
				</button>
			</div>
		</div>
	</div>
{/if}
