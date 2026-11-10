<script lang="ts">
	import { toast } from 'svelte-sonner';
	import SynIcon from './SynIcon.svelte';
	import SynEmoji from './SynEmoji.svelte';

	export let botModelId: string = 'support-bot';

	let channel: 'chat' | 'webrtc' | 'sip' = 'chat';
	let text = 'Где мой заказ № 94821? Мой телефон +7 913 123-45-67';
	let anger = 0.0;
	let running = false;
	let trace: any = null;

	const runSimulation = async () => {
		if (!text.trim()) return;
		running = true;
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
						name: 'bot_simulate_conversation_turn',
						arguments: {
							bot_model_id: botModelId || 'support-bot',
							text: text.trim(),
							channel,
							anger_score: anger
						}
					}
				})
			});
			const data = await response.json();
			if (data.error) throw data.error.message || JSON.stringify(data.error);
			trace = data.result?.structuredContent ?? null;
		} catch (e) {
			toast.error(`Ошибка симуляции: ${e}`);
		} finally {
			running = false;
		}
	};
</script>

<div class="syn-color-block space-y-3">
	<div class="flex flex-wrap items-center gap-2">
		<span class="inline-flex items-center gap-1.5 rounded-lg bg-[var(--syn-bg-elevated)] px-2.5 py-1.5 text-[0.7rem] font-semibold">
			<SynIcon name="spark" size={14} />
			<span>ЖИВОЙ ТЕСТ КОНВЕЙЕРА</span>
		</span>

		<select
			bind:value={channel}
			class="rounded-xl border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-canvas)] px-2.5 py-1.5 text-xs font-medium"
		>
			<option value="chat">Канал: Chat</option>
			<option value="webrtc">Канал: WebRTC</option>
			<option value="sip">Канал: SIP</option>
		</select>

		<input
			type="text"
			bind:value={text}
			placeholder="Введите реплику клиента для сквозной трассировки..."
			class="flex-1 min-w-[240px] rounded-xl border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-canvas)] px-3.5 py-1.5 text-xs outline-none focus:border-[var(--syn-cobalt)]"
			on:keydown={(e) => e.key === 'Enter' && runSimulation()}
		/>

		<label class="inline-flex items-center gap-1.5 text-xs text-[var(--syn-text-secondary)]">
			<span>Гнев:</span>
			<input type="range" min="0" max="1" step="0.05" bind:value={anger} class="w-16 accent-[var(--syn-coral)]" />
			<span class="w-7 font-mono text-[0.7rem]">{anger.toFixed(2)}</span>
		</label>

		<button
			type="button"
			disabled={running}
			on:click={runSimulation}
			class="inline-flex items-center gap-1.5 rounded-xl bg-[var(--syn-text-primary)] px-3.5 py-1.5 text-xs font-medium text-[var(--syn-bg-surface)] hover:opacity-90 disabled:opacity-50 transition"
		>
			<SynIcon name="play" size={13} />
			<span>{running ? 'Трассировка...' : 'Прогнать ход'}</span>
		</button>
	</div>

	{#if trace}
		<div class="flex flex-wrap items-center gap-2 pt-2 border-t border-[var(--syn-border-subtle)] text-xs">
			<div class="inline-flex items-center gap-2 rounded-lg bg-[var(--syn-block-cobalt)] px-2.5 py-1">
				<SynEmoji
					mood={trace.classification?.action === 'direct'
						? 'calm'
						: trace.classification?.action === 'disambiguate'
							? 'curious'
							: 'focused'}
					size={20}
					animated={false}
				/>
				<span>
					Интент: <b class="font-mono">{trace.classification?.intent ?? '—'}</b>
					({((trace.classification?.confidence ?? 0) * 100).toFixed(1)}% · {trace.classification?.action ?? 'fallback'})
				</span>
			</div>

			<SynIcon name="arrow-right" size={14} className="text-[var(--syn-text-tertiary)]" />

			<div class="inline-flex items-center gap-1.5 rounded-lg bg-[var(--syn-block-amber)] px-2.5 py-1">
				<SynIcon name="layers" size={14} />
				<span>
					Слоты:
					{#if trace.slots && Object.keys(trace.slots).length > 0}
						<b class="font-mono">
							{Object.entries(trace.slots)
								.map(([k, v]) => `${k}=${v}`)
								.join(', ')}
						</b>
					{:else}
						<span>нет</span>
					{/if}
				</span>
			</div>

			<SynIcon name="arrow-right" size={14} className="text-[var(--syn-text-tertiary)]" />

			<div class="inline-flex items-center gap-1.5 rounded-lg bg-[var(--syn-block-emerald)] px-2.5 py-1">
				<SynIcon name="check" size={14} />
				<span>
					Решение: <b class="font-mono">{trace.decision?.route ?? 'fallback'}</b>
					{#if trace.decision?.answer}
						— «{trace.decision.answer.slice(0, 90)}»
					{:else if trace.decision?.question}
						— Вопрос слота: «{trace.decision.question}»
					{/if}
				</span>
			</div>

			<button
				type="button"
				on:click={() => (trace = null)}
				class="ml-auto text-[0.7rem] text-[var(--syn-text-tertiary)] hover:text-[var(--syn-text-primary)]"
			>
				Скрыть
			</button>
		</div>
	{/if}
</div>
