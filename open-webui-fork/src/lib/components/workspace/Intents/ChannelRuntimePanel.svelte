<script lang="ts">
	import { onMount, onDestroy, createEventDispatcher } from 'svelte';
	import { toast } from 'svelte-sonner';
	import SynIcon from '../../synrespond/SynIcon.svelte';
	import SynEmoji from '../../synrespond/SynEmoji.svelte';
	import '../../synrespond/synrespond-tokens.css';

	const dispatch = createEventDispatcher();

	let channelState: Record<string, any> = {};
	let loading = true;
	let saving = false;

	// Состояние браузерного голосового тестера
	let testing = false;
	let testLog: { role: 'you' | 'bot' | 'sys'; text: string }[] = [];
	let lastTtfbMs: number | null = null;
	let callerAngerScore = 0.0;

	let ws: WebSocket | null = null;
	let mediaStream: MediaStream | null = null;
	let captureCtx: AudioContext | null = null;
	let playbackCtx: AudioContext | null = null;
	let processor: ScriptProcessorNode | null = null;
	let nextPlayTime = 0;
	let activeSources: AudioBufferSourceNode[] = [];

	const CHANNELS = [
		{
			key: 'channels.chat.enabled',
			name: 'Текстовый чат и виджет',
			tag: 'CHAT / SSE',
			block: 'syn-block-cobalt',
			desc: 'Потоковый SSE-ответ, размаскирование ПДн на лету и визуализация цепочек Reasoning Path.'
		},
		{
			key: 'channels.webrtc.enabled',
			name: 'Браузерный голос WebRTC',
			tag: 'WEBRTC / 16-48 kHz',
			block: 'syn-block-emerald',
			desc: 'Полнодуплексный аудиоканал из браузера с VAD, мгновенным barge-in и fallback без GPU.'
		},
		{
			key: 'channels.sip.enabled',
			name: 'Телефония SIP (Asterisk)',
			tag: 'AUDIOSOCKET :8765',
			block: 'syn-block-violet',
			desc: 'Шлюз телефонии 8 кГц slin, ресемплинг soxr HQ, филлеры 250 мс и перевод в очередь операторов.'
		}
	];

	const loadChannels = async () => {
		loading = true;
		try {
			const res = await fetch('/api/v1/ai-manifest.json', {
				headers: { authorization: `Bearer ${localStorage.token}` }
			});
			const data = res.ok ? await res.json() : null;
			channelState = data?.current_config ?? {
				'channels.chat.enabled': true,
				'channels.webrtc.enabled': true,
				'channels.sip.enabled': true
			};
		} catch (e) {
			toast.error(`${e}`);
		} finally {
			loading = false;
		}
	};

	const toggleChannel = async (key: string) => {
		saving = true;
		try {
			const nextVal = !channelState[key];
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
						arguments: { settings: { [key]: nextVal } }
					}
				})
			});
			const data = await response.json();
			if (data.error) throw data.error.message || JSON.stringify(data.error);
			channelState = { ...channelState, [key]: nextVal };
			toast.success('Состояние канала обновлено');
		} catch (e) {
			toast.error(`${e}`);
		} finally {
			saving = false;
		}
	};

	const clearPlaybackQueue = () => {
		for (const src of activeSources) {
			try {
				src.stop();
			} catch {}
		}
		activeSources = [];
		if (playbackCtx) {
			nextPlayTime = playbackCtx.currentTime;
		}
	};

	const playPcm16kChunk = (buffer: ArrayBuffer) => {
		if (!playbackCtx) {
			playbackCtx = new AudioContext({ sampleRate: 16000 });
		}
		const int16 = new Int16Array(buffer);
		if (int16.length === 0) return;
		const float32 = new Float32Array(int16.length);
		for (let i = 0; i < int16.length; i++) {
			float32[i] = int16[i] / 32768.0;
		}
		const audioBuffer = playbackCtx.createBuffer(1, float32.length, 16000);
		audioBuffer.getChannelData(0).set(float32);

		const source = playbackCtx.createBufferSource();
		source.buffer = audioBuffer;
		source.connect(playbackCtx.destination);

		const startAt = Math.max(playbackCtx.currentTime, nextPlayTime);
		source.start(startAt);
		nextPlayTime = startAt + audioBuffer.duration;
		activeSources.push(source);
		source.onended = () => {
			activeSources = activeSources.filter((s) => s !== source);
		};
	};

	const startTestCall = async () => {
		testing = true;
		testLog = [];
		lastTtfbMs = null;
		try {
			mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000 } });
			const proto = location.protocol === 'https:' ? 'wss' : 'ws';
			const tokenParam = encodeURIComponent(localStorage.token ?? '');
			ws = new WebSocket(`${proto}://${location.host}/api/v1/voice/webrtc/stream?token=${tokenParam}`);
			ws.binaryType = 'arraybuffer';

			ws.onmessage = (event) => {
				if (event.data instanceof ArrayBuffer) {
					playPcm16kChunk(event.data);
					return;
				}
				if (typeof event.data === 'string') {
					const msg = JSON.parse(event.data);
					if (msg.type === 'barge_in_clear') {
						clearPlaybackQueue();
						testLog = [...testLog, { role: 'sys', text: 'Barge-in: речь абонента прервала озвучку бота' }];
					} else if (msg.type === 'transcript') {
						testLog = [...testLog, { role: 'you', text: msg.text }];
					} else if (msg.type === 'answer') {
						testLog = [...testLog, { role: 'bot', text: msg.text }];
					} else if (msg.type === 'ttfb') {
						lastTtfbMs = msg.ms;
					} else if (msg.type === 'anger_score') {
						callerAngerScore = Number(msg.score ?? 0);
					} else if (msg.type === 'error') {
						testLog = [...testLog, { role: 'sys', text: `Ошибка шлюза: ${msg.detail}` }];
					}
				}
			};

			ws.onopen = () => {
				captureCtx = new AudioContext({ sampleRate: 16000 });
				playbackCtx = new AudioContext({ sampleRate: 16000 });
				const source = captureCtx.createMediaStreamSource(mediaStream!);
				processor = captureCtx.createScriptProcessor(2048, 1, 1);
				processor.onaudioprocess = (e) => {
					if (ws?.readyState === WebSocket.OPEN) {
						const pcm = e.inputBuffer.getChannelData(0);
						const buf = new Int16Array(pcm.length);
						let sumSq = 0;
						for (let i = 0; i < pcm.length; i++) {
							buf[i] = Math.max(-32768, Math.min(32767, pcm[i] * 32768));
							sumSq += pcm[i] * pcm[i];
						}
						const rms = Math.sqrt(sumSq / pcm.length);
						callerAngerScore = Math.min(1.0, Math.max(0.0, (rms - 0.04) * 3.2));
						ws.send(buf.buffer);
					}
				};
				source.connect(processor);
				processor.connect(captureCtx.destination);
				testLog = [...testLog, { role: 'sys', text: 'Дуплексный голосовой канал открыт (16 кГц PCM)' }];
			};

			ws.onerror = () => {
				testLog = [...testLog, { role: 'sys', text: 'Соединение с голосовым шлюзом не установлено' }];
			};
		} catch (e) {
			toast.error(`Ошибка микрофона или сокета: ${e}`);
			stopTestCall();
		}
	};

	const stopTestCall = () => {
		clearPlaybackQueue();
		processor?.disconnect();
		mediaStream?.getTracks().forEach((t) => t.stop());
		captureCtx?.close();
		playbackCtx?.close();
		ws?.close();
		processor = null;
		mediaStream = null;
		captureCtx = null;
		playbackCtx = null;
		ws = null;
		testing = false;
	};

	onMount(loadChannels);
	onDestroy(stopTestCall);
</script>

<div class="space-y-4">
	<div class="grid grid-cols-1 gap-3 lg:grid-cols-3">
		{#each CHANNELS as ch}
			{@const isEnabled = Boolean(channelState[ch.key] ?? true)}
			<div class="syn-color-block {ch.block} flex flex-col justify-between gap-3">
				<div class="flex items-start justify-between gap-2">
					<div>
						<span class="rounded-md bg-[var(--syn-bg-surface)] px-2 py-0.5 font-mono text-[0.65rem] font-semibold">
							{ch.tag}
						</span>
						<h3 class="mt-1.5 text-sm font-semibold">{ch.name}</h3>
					</div>
					<SynEmoji mood={isEnabled ? 'calm' : 'sleeping'} size={28} animated={false} />
				</div>

				<p class="text-xs text-[var(--syn-text-secondary)]">{ch.desc}</p>

				<div class="flex items-center justify-between pt-2 border-t border-[var(--syn-border-subtle)]">
					<span class="text-xs font-medium {isEnabled ? 'text-[var(--syn-emerald)]' : 'text-[var(--syn-text-tertiary)]'}">
						{isEnabled ? 'Контур активен' : 'Контур отключен'}
					</span>
					<button
						type="button"
						disabled={saving || loading}
						on:click={() => toggleChannel(ch.key)}
						class="rounded-lg border border-[var(--syn-border-subtle)] bg-[var(--syn-bg-surface)] px-3 py-1 text-xs font-medium hover:border-[var(--syn-border-strong)] transition"
					>
						{isEnabled ? 'Отключить' : 'Включить'}
					</button>
				</div>
			</div>
		{/each}
	</div>

	<!-- Живой полнодуплексный стенд проверки голоса -->
	<div class="syn-color-block space-y-4">
		<div class="flex flex-wrap items-center justify-between gap-3">
			<div class="flex items-center gap-3">
				<SynEmoji
					mood={!testing ? 'calm' : callerAngerScore >= 0.75 ? 'angry' : callerAngerScore >= 0.4 ? 'tense' : 'focused'}
					size={36}
				/>
				<div>
					<div class="text-sm font-semibold">Полнодуплексный голосовой стенд WebRTC (Barge-In & Emotion)</div>
					<div class="text-xs text-[var(--syn-text-tertiary)]">
						Захват микрофона 16 кГц, воспроизведение синтеза речи бота и индикатор эмоции в реальном времени
					</div>
				</div>
			</div>

			<div class="flex items-center gap-3">
				{#if lastTtfbMs !== null}
					<span class="rounded-full bg-[var(--syn-block-emerald)] px-3 py-1 font-mono text-xs text-[var(--syn-emerald)]">
						TTFB: {lastTtfbMs} мс
					</span>
				{/if}
				<button
					type="button"
					on:click={() => (testing ? stopTestCall() : startTestCall())}
					class="inline-flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-medium transition {testing
						? 'bg-[var(--syn-coral)] text-white'
						: 'bg-[var(--syn-text-primary)] text-[var(--syn-bg-surface)]'}"
				>
					<SynIcon name={testing ? 'stop' : 'phone'} size={15} />
					<span>{testing ? 'Завершить звонок' : 'Начать тестовый звонок'}</span>
				</button>
			</div>
		</div>

		{#if testLog.length > 0}
			<div class="max-h-52 overflow-y-auto rounded-xl bg-[var(--syn-bg-canvas)] p-3 space-y-1.5 text-xs">
				{#each testLog as entry}
					<div class="flex items-start gap-2">
						<span class="font-mono uppercase text-[0.68rem] text-[var(--syn-text-tertiary)] w-12 shrink-0">
							{entry.role}:
						</span>
						<span class={entry.role === 'bot' ? 'font-medium text-[var(--syn-cobalt)]' : ''}>
							{entry.text}
						</span>
					</div>
				{/each}
			</div>
		{/if}
	</div>
</div>
